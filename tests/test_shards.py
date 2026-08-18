"""Reading a shard by byte range instead of by mapping it.

The engine used to read every tensor through ``safe_open``, which holds the whole shard memory
-mapped for as long as the handle lives. That is free on Linux, where a read-only file mapping is
page cache and charged to nobody, and it is not free on Windows, where the same mapping is charged
against a commit limit in full the moment it is opened. Measured on the 35B mixture's split
shards: 1.57GB of commit per handle, before a byte was read.

Reading the byte range instead costs the bytes and nothing else, on every system. What has to be
proven is that it costs nothing in correctness either -- so the direct path is checked against
safetensors here, tensor for tensor, on every dtype this build of torch can store, on empty
tensors, on scalars, and on the row slices a mixture streams its experts through.
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import safetensors
import torch
from safetensors.torch import safe_open, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.streaming import shards  # noqa: E402


def storable_dtypes():
    """Every dtype this build of torch and this build of safetensors agree on.

    Determined by trying it, not by listing versions: the set differs between torch builds, and a
    list written today is a skipped test tomorrow.
    """
    found = []
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.safetensors"
        for dtype in dict.fromkeys(shards._DTYPES.values()):
            try:
                tensor = torch.zeros(4, dtype=dtype)
                save_file({"t": tensor}, str(probe))
                with safe_open(str(probe), framework="pt") as handle:
                    handle.get_tensor("t")
            except Exception:  # noqa: BLE001 - the point is to find out which ones work
                continue
            found.append(dtype)
    return found


def sample_for(dtype, shape):
    """A tensor with distinguishable bytes, so a mis-parsed offset cannot pass unnoticed."""
    count = 1
    for dim in shape:
        count *= dim
    if dtype == torch.bool:
        return (torch.arange(count) % 3 == 0).reshape(shape)
    if dtype.is_floating_point or dtype.is_complex:
        return (torch.arange(count, dtype=torch.float32) * 0.5 - 3).reshape(shape).to(dtype)
    info = torch.iinfo(dtype)
    values = torch.arange(count, dtype=torch.int64) % min(97, max(2, info.max))
    return values.reshape(shape).to(dtype)


def as_bytes(tensor):
    """A tensor's raw bytes, for the dtypes ``torch.equal`` will not compare -- fp8 has no eq."""
    if not tensor.numel():
        return tensor
    try:
        return tensor.contiguous().view(torch.uint8)
    except RuntimeError:
        return tensor


class ShardCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(shards.release_all)
        self.root = Path(self._tmp.name)

    def direct(self):
        """A reader forced onto the byte-range route, whatever this machine would choose."""
        return shards.configure_reader(self.root, shard_handle_limit="direct")

    def mapped(self):
        """A reader forced onto the mapping route, which is what an unconstrained host picks."""
        return shards.configure_reader(self.root, shard_handle_limit="unbounded")

    def write(self, name, state_dict, metadata=None):
        path = self.root / f"{name}.safetensors"
        save_file(state_dict, str(path), metadata=metadata)
        return path


# ---- the header ---------------------------------------------------------------------------------

class TestHeader(ShardCase):

    def test_it_reads_what_safetensors_wrote(self):
        path = self.write("m", {"a": torch.zeros(3, 4), "b": torch.ones(5, dtype=torch.int16)})
        index = shards.read_index(path)
        self.assertEqual(sorted(index.keys()), ["a", "b"])
        self.assertEqual(index["a"].shape, (3, 4))
        self.assertEqual(index["a"].dtype, torch.float32)
        self.assertEqual(index["a"].nbytes, 3 * 4 * 4)
        self.assertEqual(index["b"].shape, (5,))
        self.assertEqual(index["b"].nbytes, 10)

    def test_offsets_are_absolute_and_inside_the_file(self):
        path = self.write("m", {"a": torch.zeros(64), "b": torch.zeros(64)})
        index = shards.read_index(path)
        for entry in index.entries.values():
            self.assertGreater(entry.begin, 8)
            self.assertLessEqual(entry.end, path.stat().st_size)

    def test_metadata_is_kept_and_never_mistaken_for_a_tensor(self):
        path = self.write("m", {"a": torch.zeros(2)}, metadata={"format": "pt", "who": "test"})
        index = shards.read_index(path)
        self.assertEqual(index.keys(), ["a"])
        self.assertEqual(index.metadata.get("who"), "test")

    def test_a_scalar_and_an_empty_tensor_survive_the_round_trip(self):
        path = self.write("m", {"scalar": torch.tensor(7.5), "empty": torch.zeros(0, 8)})
        index = shards.read_index(path)
        self.assertEqual(index["scalar"].shape, ())
        self.assertEqual(index["scalar"].nbytes, 4)
        self.assertEqual(index["empty"].shape, (0, 8))
        self.assertEqual(index["empty"].nbytes, 0)

    def test_a_file_that_is_not_a_shard_is_named_as_such(self):
        path = self.root / "junk.safetensors"
        path.write_bytes(b"not a safetensors file at all, not even close")
        with self.assertRaises(shards.ShardFormatError):
            shards.read_index(path)

    def test_a_truncated_header_is_refused_rather_than_half_read(self):
        blob = struct.pack("<Q", 4096) + b"{}"
        with self.assertRaises(shards.ShardFormatError):
            shards.parse_header(blob)

    def test_a_header_that_lies_about_a_tensors_size_is_refused(self):
        body = json.dumps({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}).encode()
        with self.assertRaises(shards.ShardFormatError) as caught:
            shards.parse_header(struct.pack("<Q", len(body)) + body)
        self.assertIn("shape and dtype", str(caught.exception))

    def test_a_tensor_that_runs_past_the_end_of_the_file_is_refused(self):
        body = json.dumps({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
        with self.assertRaises(shards.ShardFormatError) as caught:
            shards.parse_header(struct.pack("<Q", len(body)) + body, size=8 + len(body))
        self.assertIn("past the end", str(caught.exception))

    def test_an_absurd_header_length_is_refused_before_anything_is_allocated(self):
        with self.assertRaises(shards.ShardFormatError):
            shards.parse_header(struct.pack("<Q", 2 ** 62) + b"{}")


# ---- the bytes ----------------------------------------------------------------------------------

class TestDirectReadsMatchSafetensors(ShardCase):
    """The claim the whole change rests on: the same bytes, whichever way they are read."""

    def test_every_dtype_this_build_supports(self):
        dtypes = storable_dtypes()
        self.assertGreaterEqual(len(dtypes), 8, "torch should store more dtypes than this")
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                state = {"vec": sample_for(dtype, (17,)),
                         "mat": sample_for(dtype, (5, 9)),
                         "cube": sample_for(dtype, (3, 4, 2))}
                path = self.write(f"d{str(dtype).replace('.', '_')}", state)
                got = self.direct().read_tensors(path)
                self.assertEqual(sorted(got), sorted(state))
                with safe_open(str(path), framework="pt") as handle:
                    for name in state:
                        want = handle.get_tensor(name)
                        self.assertEqual(got[name].dtype, want.dtype)
                        self.assertEqual(got[name].shape, want.shape)
                        self.assertTrue(torch.equal(as_bytes(got[name]), as_bytes(want)))

    def test_a_subset_reads_only_what_was_asked_for(self):
        state = {f"t{i}": sample_for(torch.float32, (8, 8)) for i in range(6)}
        path = self.write("m", state)
        reader = self.direct()
        got = reader.read_tensors(path, keys=["t4", "t1"])
        self.assertEqual(list(got), ["t4", "t1"])
        self.assertEqual(reader.bytes_read, 2 * 8 * 8 * 4)
        with safe_open(str(path), framework="pt") as handle:
            for name in ("t1", "t4"):
                self.assertTrue(torch.equal(got[name], handle.get_tensor(name)))

    def test_scalars_and_empty_tensors_read_back_identically(self):
        state = {"scalar": torch.tensor(-2.5), "empty": torch.zeros(0, 8),
                 "empty_1d": torch.zeros(0, dtype=torch.int64)}
        path = self.write("m", state)
        got = self.direct().read_tensors(path)
        with safe_open(str(path), framework="pt") as handle:
            for name in state:
                want = handle.get_tensor(name)
                self.assertEqual(got[name].shape, want.shape)
                self.assertTrue(torch.equal(got[name], want))

    def test_rows_read_the_same_bytes_as_slicing_the_mapping(self):
        """The fused-expert read. Row r of a batched tensor is a contiguous range, and this is the
        arithmetic that makes a top-k cost k experts rather than a layer."""
        experts = sample_for(torch.bfloat16, (16, 6, 5))
        path = self.write("moe", {"gate_up_proj": experts, "other": torch.zeros(4)})
        for wanted in ([0], [15], [3, 9, 1], [4, 5, 6], [0, 15], list(range(16))):
            with self.subTest(rows=wanted):
                got = self.direct().read_tensors(
                    path, keys=["gate_up_proj"], rows={"gate_up_proj": wanted})["gate_up_proj"]
                want = torch.cat([experts[r:r + 1] for r in wanted], dim=0)
                self.assertEqual(got.shape, want.shape)
                self.assertTrue(torch.equal(as_bytes(got), as_bytes(want)))

    def test_a_run_of_rows_costs_one_read_and_scattered_rows_cost_one_each(self):
        """Consecutive experts are coalesced into a single range. Not a correctness property, but
        the reason the coalescing exists, and it is silent when it stops working."""
        path = self.write("moe", {"w": sample_for(torch.float32, (12, 4))})
        reader = self.direct()
        before = reader.reads
        reader.read_tensors(path, keys=["w"], rows={"w": [2, 3, 4, 5]})
        self.assertEqual(reader.reads - before, 1)
        before = reader.reads
        reader.read_tensors(path, keys=["w"], rows={"w": [1, 5, 9]})
        self.assertEqual(reader.reads - before, 3)

    def test_only_the_requested_rows_bytes_are_read(self):
        path = self.write("moe", {"w": sample_for(torch.float32, (64, 128))})
        reader = self.direct()
        reader.read_tensors(path, keys=["w"], rows={"w": [7, 8]})
        self.assertEqual(reader.bytes_read, 2 * 128 * 4)

    def test_asking_for_rows_that_are_not_there_is_an_error_not_a_wrong_answer(self):
        """The byte-range route knows the row count and says so, rather than reading past it."""
        path = self.write("moe", {"w": sample_for(torch.float32, (4, 4))})
        with self.assertRaises(IndexError):
            self.direct().read_tensors(path, keys=["w"], rows={"w": [9]})

    def test_an_unknown_tensor_is_named_in_the_error(self):
        path = self.write("m", {"a": torch.zeros(2)})
        with self.assertRaises(KeyError) as caught:
            self.direct().read_tensors(path, keys=["nope"])
        self.assertIn("nope", str(caught.exception))


# ---- the two routes ------------------------------------------------------------------------------

class TestBothRoutesAgree(ShardCase):
    """The engine picks between these from a measurement, so they have to be interchangeable."""

    def state(self):
        return {"w": sample_for(torch.bfloat16, (7, 5)),
                "experts": sample_for(torch.float32, (9, 3)),
                "scalar": torch.tensor(1.25),
                "empty": torch.zeros(0, 4),
                "ints": sample_for(torch.int32, (6,))}

    def test_the_same_tensors_come_back_either_way(self):
        state = self.state()
        path = self.write("m", state)
        for keys, rows in ((None, None), (["w"], None), (["w", "ints"], None),
                           (["experts"], {"experts": [0, 4, 8]}),
                           (["experts"], {"experts": [2, 3]}),
                           (["experts"], {"experts": [5]})):
            with self.subTest(keys=keys, rows=rows):
                a = self.mapped().read_tensors(path, keys=keys, rows=rows)
                b = self.direct().read_tensors(path, keys=keys, rows=rows)
                self.assertEqual(sorted(a), sorted(b))
                for name in a:
                    self.assertEqual(a[name].dtype, b[name].dtype)
                    self.assertEqual(a[name].shape, b[name].shape)
                    self.assertTrue(torch.equal(as_bytes(a[name]), as_bytes(b[name])))

    def test_filling_a_buffer_gives_the_same_bytes_either_way(self):
        """The loader's path, which packs a whole module into one staging buffer."""
        state = self.state()
        path = self.write("m", state)
        results = []
        for reader in (self.mapped(), self.direct()):
            index = reader.index(path)
            buffers = {n: torch.zeros(index[n].nbytes, dtype=torch.uint8) for n in state}
            reader.read_into(path, [shards.ReadRequest(name=n, destination=b)
                                    for n, b in buffers.items()])
            results.append(buffers)
        for name in state:
            self.assertTrue(torch.equal(results[0][name], results[1][name]), name)

    def test_the_mapping_route_aliases_the_file_and_the_range_route_does_not(self):
        """Why the choice is worth making at all, and why it cannot always be the first one.

        On the mapping route the tensors ARE the file's pages: their addresses sit at exactly the
        offsets the header gives, so the read copies nothing -- and the pages stay charged for as
        long as any of those tensors lives, which is what makes the route unaffordable on a machine
        whose commit limit is smaller than the checkpoint.
        """
        state = {f"t{i}": sample_for(torch.float32, (256, 64)) for i in range(4)}
        path = self.write("m", state)
        reader = self.mapped()
        index = reader.index(path)
        got = reader.read_tensors(path)
        base_name = min(state, key=lambda n: index[n].begin)
        base_ptr = got[base_name].data_ptr()
        for name in state:
            self.assertEqual(got[name].data_ptr() - base_ptr,
                             index[name].begin - index[base_name].begin)
        self.assertEqual(reader.handle_opens, 1)

        # The byte-range route holds nothing: no handle was opened, so no page of that file is
        # charged to this process once the read returns.
        reader = self.direct()
        reader.read_tensors(path)
        self.assertEqual(reader.handle_opens, 0)


# ---- what must not happen -----------------------------------------------------------------------

class TestNothingIsMapped(ShardCase):
    """The regression this file exists for: no read may open a mapping on the fast path."""

    def forbid_mapping(self):
        def refuse(*args, **kwargs):
            raise AssertionError("a shard was memory-mapped on the direct read path")

        original = safetensors.safe_open
        safetensors.safe_open = refuse
        self.addCleanup(lambda: setattr(safetensors, "safe_open", original))

    def test_reading_a_shard_never_maps_it(self):
        state = {f"t{i}": sample_for(torch.float32, (8, 8)) for i in range(4)}
        path = self.write("m", state)
        reader = shards.configure_reader(self.root, shard_handle_limit="direct")
        self.assertEqual(reader.mode, shards.ShardReader.DIRECT)
        self.forbid_mapping()
        got = reader.read_tensors(path)
        self.assertEqual(sorted(got), sorted(state))
        self.assertEqual(reader.handle_opens, 0)

    def test_the_engines_own_readers_never_map_a_shard(self):
        """``rocketllm.utils`` is where the prefetch workers enter, and where the bug lived."""
        from rocketllm import utils

        state = {"model.layers.0.w": sample_for(torch.float32, (6, 6)),
                 "model.layers.0.b": sample_for(torch.float32, (6,)),
                 "model.layers.0.experts": sample_for(torch.float32, (8, 3))}
        self.write("model.layers.0", state)
        shards.configure_reader(self.root, shard_handle_limit="direct")
        self.forbid_mapping()
        self.assertEqual(sorted(utils.layer_tensor_names(self.root, "model.layers.0")),
                         sorted(state))
        subset = utils.load_layer_subset(self.root, "model.layers.0", ["model.layers.0.b"])
        self.assertEqual(list(subset), ["model.layers.0.b"])
        rows = utils.load_layer_rows(self.root, "model.layers.0",
                                     {"model.layers.0.experts": [1, 2]})
        self.assertEqual(rows["model.layers.0.experts"].shape, (2, 3))
        whole = utils.load_layer(self.root, "model.layers.0")
        self.assertEqual(sorted(whole), sorted(state))

    def test_the_free_functions_read_what_safetensors_reads(self):
        from rocketllm import utils

        state = {"model.layers.0.w": sample_for(torch.bfloat16, (9, 7)),
                 "model.layers.0.e": sample_for(torch.bfloat16, (5, 4))}
        path = self.write("model.layers.0", state)
        whole = utils.load_layer(self.root, "model.layers.0")
        subset = utils.load_layer_subset(self.root, "model.layers.0", ["model.layers.0.w"])
        rows = utils.load_layer_rows(self.root, "model.layers.0",
                                     {"model.layers.0.e": [3, 0]})
        with safe_open(str(path), framework="pt") as handle:
            for name in state:
                self.assertTrue(torch.equal(as_bytes(whole[name]),
                                            as_bytes(handle.get_tensor(name))))
            self.assertTrue(torch.equal(as_bytes(subset["model.layers.0.w"]),
                                        as_bytes(handle.get_tensor("model.layers.0.w"))))
            experts = handle.get_tensor("model.layers.0.e")
        self.assertTrue(torch.equal(as_bytes(rows["model.layers.0.e"]),
                                    as_bytes(torch.cat([experts[3:4], experts[0:1]]))))


# ---- headers are remembered, not re-read ---------------------------------------------------------

class TestIndexCache(ShardCase):

    def test_a_header_is_parsed_once_per_shard(self):
        path = self.write("m", {"a": torch.zeros(4)})
        reader = shards.reader_for(self.root)
        parses = []
        original = shards.read_index
        shards.read_index = lambda p: (parses.append(p), original(p))[1]
        self.addCleanup(lambda: setattr(shards, "read_index", original))
        for _ in range(5):
            reader.index(path)
        self.assertEqual(len(parses), 1)

    def test_a_rewritten_shard_is_re_read_rather_than_served_stale(self):
        path = self.write("m", {"a": torch.zeros(4)})
        reader = shards.reader_for(self.root)
        self.assertEqual(reader.index(path)["a"].shape, (4,))
        # Written through the same name with a different shape, which is exactly the case a cache
        # keyed on the path alone would get wrong.
        save_file({"a": torch.zeros(9)}, str(path))
        self.assertEqual(reader.index(path)["a"].shape, (9,))

    def test_releasing_drops_the_headers(self):
        path = self.write("m", {"a": torch.zeros(4)})
        reader = shards.reader_for(self.root)
        reader.index(path)
        self.assertTrue(reader._indexes)
        reader.release()
        self.assertFalse(reader._indexes)


# ---- one reader per checkpoint --------------------------------------------------------------------

class TestReaderRegistry(ShardCase):

    def test_the_same_checkpoint_gets_the_same_reader(self):
        self.write("m", {"a": torch.zeros(4)})
        first = shards.reader_for(self.root)
        self.assertIs(shards.reader_for(str(self.root)), first)
        self.assertIs(shards.reader_for(self.root / "." ), first)

    def test_configuring_replaces_the_default_one(self):
        self.write("m", {"a": torch.zeros(4)})
        default = shards.reader_for(self.root)
        configured = shards.configure_reader(self.root, shard_handle_limit=3)
        self.assertIsNot(configured, default)
        self.assertIs(shards.reader_for(self.root), configured)
        self.assertEqual(configured.handle_limit, 3)

    def test_releasing_someone_elses_reader_is_a_no_op(self):
        """A model shutting down must not pull the reader out from under one opened after it."""
        self.write("m", {"a": torch.zeros(4)})
        first = shards.reader_for(self.root)
        second = shards.configure_reader(self.root)
        self.assertIsNone(shards.release_reader(self.root, only=first))
        self.assertIs(shards.reader_for(self.root), second)
        self.assertIs(shards.release_reader(self.root, only=second), second)


# ---- splitting the work ---------------------------------------------------------------------------

class TestPartition(ShardCase):

    def requests(self, path, names):
        index = self.direct().index(path)
        return [shards.ReadRequest(name=name,
                                   destination=torch.empty(index[name].nbytes, dtype=torch.uint8))
                for name in names]

    def test_every_request_lands_in_exactly_one_batch(self):
        state = {f"t{i}": torch.zeros(1 + i * 13, dtype=torch.float32) for i in range(9)}
        path = self.write("m", state)
        reader = self.direct()
        requests = self.requests(path, list(state))
        for ways in (1, 2, 3, 4, 16):
            with self.subTest(ways=ways):
                batches = reader.partition(path, requests, ways)
                self.assertLessEqual(len(batches), min(ways, len(requests)))
                flat = [r for batch in batches for r in batch]
                self.assertEqual(len(flat), len(requests))
                self.assertEqual({id(r) for r in flat}, {id(r) for r in requests})

    def test_each_batch_is_a_forward_sweep_of_the_file(self):
        state = {f"t{i}": torch.zeros(64, dtype=torch.float32) for i in range(8)}
        path = self.write("m", state)
        reader = self.direct()
        index = reader.index(path)
        batches = reader.partition(path, self.requests(path, list(state)), 3)
        for batch in batches:
            offsets = [index[r.name].begin for r in batch]
            self.assertEqual(offsets, sorted(offsets))

    def test_batches_are_balanced_by_bytes_not_by_count(self):
        """One 512KB projection beside thirty norms is the real shape of a decoder layer."""
        state = {"big": torch.zeros(131072, dtype=torch.float32)}
        state.update({f"n{i}": torch.zeros(4, dtype=torch.float32) for i in range(30)})
        path = self.write("m", state)
        reader = self.direct()
        index = reader.index(path)
        batches = reader.partition(path, self.requests(path, list(state)), 2)
        self.assertEqual(len(batches), 2)
        counts = sorted(len(batch) for batch in batches)
        self.assertLess(counts[0], counts[1])
        loads = sorted(sum(index[r.name].nbytes for r in batch) for batch in batches)
        self.assertGreater(loads[0], 0)

    def test_no_requests_is_no_batches(self):
        path = self.write("m", {"a": torch.zeros(4)})
        self.assertEqual(self.direct().partition(path, [], 4), [])


# ---- the mapped fallback ---------------------------------------------------------------------------

class TestMappedFallback(ShardCase):
    """A big-endian host cannot use the container's bytes as stored, and takes the old path."""

    def mapped_reader(self, **kwargs):
        original = shards.direct_reads_available
        shards.direct_reads_available = lambda: False
        try:
            reader = shards.ShardReader(self.root, **kwargs)
        finally:
            shards.direct_reads_available = original
        self.addCleanup(reader.release)
        return reader

    def test_it_reads_exactly_what_the_direct_path_reads(self):
        state = {"w": sample_for(torch.bfloat16, (7, 5)),
                 "experts": sample_for(torch.float32, (9, 3)),
                 "scalar": torch.tensor(1.25)}
        path = self.write("m", state)
        mapped = self.mapped_reader()
        self.assertEqual(mapped.mode, shards.ShardReader.MAPPED)
        direct = shards.reader_for(self.root)
        for keys, rows in ((None, None), (["w"], None),
                           (["experts"], {"experts": [0, 4, 8]}),
                           (["experts"], {"experts": [2, 3]})):
            with self.subTest(keys=keys, rows=rows):
                a = mapped.read_tensors(path, keys=keys, rows=rows)
                b = direct.read_tensors(path, keys=keys, rows=rows)
                self.assertEqual(sorted(a), sorted(b))
                for name in a:
                    self.assertTrue(torch.equal(as_bytes(a[name]), as_bytes(b[name])))

    def test_it_is_the_path_that_gets_bounded(self):
        mapped = self.mapped_reader(shard_handle_limit=2)
        self.assertEqual((mapped.handle_mode, mapped.handle_limit), ("bounded", 2))

    def test_a_shard_the_header_reader_declines_falls_back_on_its_own(self):
        """One unreadable shard must cost that shard its fast path and no others."""
        good = self.write("good", {"a": sample_for(torch.float32, (4, 4))})
        bad = self.write("bad", {"a": sample_for(torch.float32, (4, 4))})
        reader = shards.reader_for(self.root)
        original = shards.read_index

        def refuse_one(path):
            if Path(path).name == "bad.safetensors":
                raise shards.ShardFormatError("a dtype this reader does not know")
            return original(path)

        shards.read_index = refuse_one
        self.addCleanup(lambda: setattr(shards, "read_index", original))

        got = reader.read_tensors(bad)
        self.assertIn(str(bad), reader._mapped_paths)
        reader.read_tensors(good)
        self.assertNotIn(str(good), reader._mapped_paths)
        with safe_open(str(bad), framework="pt") as handle:
            self.assertTrue(torch.equal(got["a"], handle.get_tensor("a")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
