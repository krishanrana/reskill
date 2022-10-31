from gym.envs.registration import register

from reskill.rl.envs.envs import *
import sys
from functools import reduce


register(
        id='FetchStacking-v0',
        entry_point='reskill.rl.envs.envs:FetchStackingEnv',
        kwargs={'reward_info': "place_blue"},
        max_episode_steps=100,
    )

register(
        id='FetchStackingMultiGoal-v0',
        entry_point='reskill.rl.envs.envs:FetchStackingEnv',
        kwargs={'reward_info': "place_blue", 'use_fixed_goal':False},
        max_episode_steps=100,
    )

register(
        id='FetchPyramidStack-v0',
        entry_point='reskill.rl.envs.envs:FetchStackingEnv',
        kwargs={'reward_info': "stack_red", 'use_fixed_goal':False, 'use_force_sensor':True},
        max_episode_steps=50,
    )

register(
        id='FetchPlaceMultiGoal-v0',
        entry_point='reskill.rl.envs.envs:FetchPlaceEnv',
        kwargs={'reward_info': "place", 'use_fixed_goal':False, 'use_force_sensor':True},
        max_episode_steps=50     
    )


register(
        id='FetchCleanUp-v0',
        entry_point='reskill.rl.envs.envs:FetchCleanUpEnv',
        kwargs={'reward_info': "cleanup_1block", 'use_fixed_goal':True, 'use_force_sensor':True},
        max_episode_steps=50  
    )


register(
        id='FetchSlipperyPush-v0',
        entry_point='reskill.rl.envs.envs:FetchSlipperyPushEnv',
        kwargs={'reward_info': "place", 'use_fixed_goal':True},
        max_episode_steps=100  
    )

register(
        id='FetchHook-v0',
        entry_point='reskill.rl.envs.fetch_hook_env:FetchHookEnv',
        timestep_limit=100,
    )

register(
    id='FetchComplexHook-v0',
    entry_point='reskill.rl.envs.complex_hook_env:ComplexHookEnv',
    timestep_limit=100,
)

register(
    id='ComplexHookSingleObject-v0',
    entry_point='reskill.rl.envs.complex_hook_env:ComplexHookSingleObjectEnv',
    timestep_limit=100,
)


