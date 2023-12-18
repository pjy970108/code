# ref : https://github.com/quantylab/rltrader/blob/master/src/quantylab/rltrader/utils.py
import os

os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch'
print('Enabling PyTorch...')
from network.network import Network, DNN
__all__ = [
    'Network', 'DNN'
]