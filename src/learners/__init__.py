from .q_learner import QLearner
from .hysteretic_q_learner import HystereticQLearner
from .dual_episode_ree_q_learner import DualEpisodeREEQLearner
from .ippo_learner import IPPOLearner


REGISTRY = {}


REGISTRY["q_learner"] = QLearner
REGISTRY["hysteretic_q_learner"] = HystereticQLearner
REGISTRY["dual_episode_ree_q_learner"] = DualEpisodeREEQLearner
REGISTRY["ippo_learner"] = IPPOLearner
