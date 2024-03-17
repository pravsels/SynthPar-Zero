import argparse
from easydict import EasyDict as edict
import yaml
import traceback
import sys 
from torch.utils.data import Dataset

class_to_race_map = {
    0: 'IndianFemale',
    1: 'BlackFemale',
    2: 'AsianFemale',
    3: 'AsianMale',
    4: 'WhiteMale',
    5: 'IndianMale',
    6: 'BlackMale',
    7: 'WhiteFemale',
}

class CustomDataset(Dataset):
    def __init__(self, no_of_identities=100):
        self.w_seeds = list(range(no_of_identities))

    def __len__(self):
        return len(self.w_seeds)

    def __getitem__(self, idx):
        return self.w_seeds[idx]

#----------------------------------------------------------------------------
class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)
#----------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Experiments for NetGAN"
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default="config/0.yaml",
        required=True,
        help="Path of config file")
  
    args = parser.parse_args()

    return args


def get_config(config_file):
  config = edict(yaml.load(open(config_file, 'r'), 
                                Loader=yaml.FullLoader))

  return config