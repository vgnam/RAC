import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
# from smac.env import MultiAgentEnv, StarCraft2Env
from .mate_env import MateEnv

from .matrix_game.matrix_game_simple import Matrixgame
from .matrix_game.matrix_game_1 import Matrix_game1Env
from .matrix_game.matrix_game_2 import Matrix_game2Env
from .matrix_game.matrix_game_3 import Matrix_game3Env
# from .matrix_game.mmdp_game_1 import mmdp_game1Env
from .matrix_game.matrix_game_climbing import Matrix_Game_Climbing_Env
from .matrix_game.matrix_game_penalty import Matrix_Game_Penalty_Env
from .matrix_game.matrix_game_partially_stochastic import Matrix_Game_Partially_Stochastic_Env
from .matrix_game.matrix_game_fully_stochastic import Matrix_Game_Fully_Stochastic_Env
from .matrix_game.matrix_game_gaussian import Matrix_Game_Gaussian_Env


# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["matrix_game"] = partial(env_fn, env=Matrixgame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["mate"] = partial(env_fn, env=MateEnv)
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["sc2_rew"] = partial(env_fn, env=StarCraft2Env)

REGISTRY["matrix_game_1"] = partial(env_fn, env=Matrix_game1Env)
REGISTRY["matrix_game_2"] = partial(env_fn, env=Matrix_game2Env)
REGISTRY["matrix_game_3"] = partial(env_fn, env=Matrix_game3Env)
# REGISTRY["mmdp_game_1"] = partial(env_fn, env=mmdp_game1Env)
REGISTRY["matrix_game_climbing"] = partial(env_fn, env=Matrix_Game_Climbing_Env)
REGISTRY["matrix_game_penalty"] = partial(env_fn, env=Matrix_Game_Penalty_Env)
REGISTRY["matrix_game_partially_stochastic"] = partial(env_fn, env=Matrix_Game_Partially_Stochastic_Env)
REGISTRY["matrix_game_fully_stochastic"] = partial(env_fn, env=Matrix_Game_Fully_Stochastic_Env)
REGISTRY["matrix_game_gaussian"] = partial(env_fn, env=Matrix_Game_Gaussian_Env)

# REGISTRY["find_goals"] = partial(env_fn, env=MAFindGoals)
# REGISTRY["find_treasure"] = partial(env_fn, env=MAFindTreasure)
# REGISTRY["go_together"] = partial(env_fn, env=MAGoTogether)
# REGISTRY["com_find_goals"] = partial(env_fn, env=MAComFindGoals)


# REGISTRY["occupy"] = partial(env_fn, env=MA_Wrapper)
