"""Moving weights from storage into the device.

Reading, staging and transferring live here. Nothing in this package decides *what* to move -- that
is the cache's job -- only how to move it without wasting the machine's time.
"""
from .loader import LayerLayout, LayerLoader, LoadedLayer, TensorPlacement
from .staging import BufferLease, HostStagingPool
from .transfer import SyncTransferHandle, TransferHandle, WeightTransfer

__all__ = [
    "HostStagingPool", "BufferLease",
    "LayerLoader", "LayerLayout", "LoadedLayer", "TensorPlacement",
    "WeightTransfer", "TransferHandle", "SyncTransferHandle",
]
