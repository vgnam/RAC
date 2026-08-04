REGISTRY = {}

from .basic_controller import BasicMAC
from .vf_controller import BasicVFMAC
from .dual_ree_controller import DualREEMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["vf_mac"] = BasicVFMAC
REGISTRY["dual_ree_mac"] = DualREEMAC