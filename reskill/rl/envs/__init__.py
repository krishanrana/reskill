from gym.envs.registration import register

from reskill.rl.envs.envs import *
import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])

def class_exist(className):
    try:
        cls = str_to_class(class_name)
    except AttributeError:
        cls = None
    return True if cls else False


# Robotics
# ----------------------------------------

for num_blocks in [1, 2, 3, 4, 5, 6]:

    # Default reward type is incremental

    for reward_info in ['sparse', 'incremental', 'dense']:
        if reward_info == 'dense':
            suffix = 'Dense'
        elif reward_info == 'sparse':
            suffix = 'Sparse'
        elif reward_info == 'incremental':
            suffix = ''

        kwargs = {
            'reward_info': reward_info,
        }

        # Fetch
        register(
            id='FetchStack{}{}Stage2-v1'.format(num_blocks, suffix),
            entry_point='reskill.rl.envs.envs:FetchStack{}Env'.format(num_blocks),
            kwargs=kwargs,
            max_episode_steps=50 * num_blocks,
        )

        for trainer_type in ['Easy']:
            class_name = 'FetchStack{}Trainer{}Env'.format(num_blocks, trainer_type)
            if class_exist(class_name):

                register(
                    id='FetchStack{}{}Stage1-v1'.format(num_blocks, suffix, trainer_type),
                    entry_point='reskill.rl.envs.envs:{}'.format(class_name),
                    kwargs=kwargs,
                    max_episode_steps=50 * num_blocks,
                )

        class_name = 'FetchStack{}TestEnv'.format(num_blocks)
        if class_exist(class_name):
            register(
                id='FetchStack{}{}Stage3-v1'.format(num_blocks, suffix),
                entry_point='reskill.rl.envs.envs:{}'.format(class_name),
                kwargs=kwargs,
                max_episode_steps=50 * num_blocks,
            )


# Register our own environment
register(
        id='FetchStackFlat-v0',
        entry_point='reskill.rl.envs.envs:FetchStack4FlatEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

register(
        id='FetchStackElevated-v0',
        entry_point='reskill.rl.envs.envs:FetchStack4ElevatedEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )


# New envs

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
        id='FetchStackRedonBlue-v0',
        entry_point='reskill.rl.envs.envs:FetchStackingEnv',
        kwargs={'reward_info': "stack_red", 'use_fixed_goal':False, 'use_force_sensor':True},
        max_episode_steps=50,
    )

register(
        id='FetchStackingTraining-v0',
        entry_point='reskill.rl.envs.envs:FetchStackingEnv',
        kwargs={'reward_info': "custom"},
        max_episode_steps=100,
    )

register(
        id='FetchPlaceFixedGoal-v0',
        entry_point='reskill.rl.envs.envs:FetchPlaceEnv',
        kwargs={'reward_info': "place", 'use_fixed_goal':True},
        max_episode_steps=50
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
        kwargs={'reward_info': "cleanup", 'use_fixed_goal':True, 'use_force_sensor':True},
        max_episode_steps=100   
    )

register(
        id='FetchCleanUp1Block-v0',
        entry_point='reskill.rl.envs.envs:FetchCleanUp1BlockEnv',
        kwargs={'reward_info': "cleanup_1block", 'use_fixed_goal':True, 'use_force_sensor':True},
        max_episode_steps=50  
    )

register(
        id='FetchBlockStack-v0',
        entry_point='reskill.rl.envs.envs:FetchBlockStackEnv',
        kwargs={'reward_info': "stack_red", 'use_fixed_goal':True, 'use_force_sensor':True},
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
    id='ComplexHook-v0',
    entry_point='reskill.rl.envs.complex_hook_env:ComplexHookEnv',
    timestep_limit=100,
)

register(
    id='ComplexHookSingleObject-v0',
    entry_point='reskill.rl.envs.complex_hook_env:ComplexHookSingleObjectEnv',
    timestep_limit=100,
)


