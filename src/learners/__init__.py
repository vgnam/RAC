from .q_learner import QLearner
from .hysteretic_q_learner import HystereticQLearner
from .dual_episode_ree_q_learner import DualEpisodeREEQLearner
from .ippo_learner import IPPOLearner
from .tremble_iql_learner import TrembleIQLLearner
from .tremble_block_learner import TrembleBlockLearner
from .tremble_gpi_learner import TrembleGPILearner
from .ucvar_q_learner import UCVaRQLearner


REGISTRY = {}


REGISTRY["q_learner"] = QLearner
REGISTRY["hysteretic_q_learner"] = HystereticQLearner
REGISTRY["dual_episode_ree_q_learner"] = DualEpisodeREEQLearner
REGISTRY["ippo_learner"] = IPPOLearner
REGISTRY["tremble_iql_learner"] = TrembleIQLLearner
REGISTRY["tremble_block_learner"] = TrembleBlockLearner
REGISTRY["tremble_gpi_learner"] = TrembleGPILearner
REGISTRY["ucvar_q_learner"] = UCVaRQLearner
