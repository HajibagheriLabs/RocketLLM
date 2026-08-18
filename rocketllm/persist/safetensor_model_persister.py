

import os
from pathlib import Path
from .model_persister import ModelPersister
from safetensors.torch import save_file

from ..streaming import shards




class SafetensorModelPersister(ModelPersister):


    def __init__(self, *args, **kwargs):


        super(SafetensorModelPersister, self).__init__(*args, **kwargs)


    def model_persist_exist(self, layer_name, saving_path):

        safetensor_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors')))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors.done')))

        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))

        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")

        # set done marker
        (saving_path / (layer_name + 'safetensors.done')).touch()


    def load_model(self, layer_name, path):
        # Not load_file: that maps the whole shard and then allocates the tensors beside it, so a
        # 1.5GB layer needs 3GB at once and the mapping half of it is charged against a commit
        # limit on the systems that have one. Reading the byte ranges costs the tensors alone.
        return shards.reader_for(path).read_tensors(Path(path) / (layer_name + ".safetensors"))
