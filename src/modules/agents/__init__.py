REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .vf_agent import VFAgent
from .new_ree_agent import NormalREEAgent, HyperREEAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["vf"] = VFAgent
REGISTRY["normal_ree_agent"] = NormalREEAgent
REGISTRY["hyper_ree_agent"] = HyperREEAgent