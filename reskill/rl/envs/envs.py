from reskill.rl.envs import fetch_stack_env
from gym import utils

DISTANCE_THRESHOLD = 0.04

class FetchStack1TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack1.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack2TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack3TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack4TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerOneTenthIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.1)
        utils.EzPickle.__init__(self)


class FetchStack2TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack3TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack4TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack6TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack1Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack1.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack2Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack3Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack4Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack5Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack6Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)


class FetchStack2TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack3TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack4TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack5TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack6TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


# Custom Environments
class FetchStack4FlatEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='dense'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)

class FetchStack4ElevatedEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='dense'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info)
        utils.EzPickle.__init__(self)

# Dense reward
class FetchStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='stack_red', use_fixed_goal=False, use_force_sensor=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=50,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.10, target_range=0.10, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0, use_fixed_goal=use_fixed_goal, use_force_sensor=use_force_sensor) #0.2
        utils.EzPickle.__init__(self)


class FetchCleanUpEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='cleanup', use_fixed_goal=True, use_force_sensor=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/table_cleanup_2_block.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, goals_on_stack_probability=0.0,use_fixed_goal=True,use_force_sensor=True) #0.2
        utils.EzPickle.__init__(self)

class FetchCleanUp1BlockEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='cleanup_1block', use_fixed_goal=True, use_force_sensor=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/table_cleanup_1_block.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, use_fixed_goal=True, use_force_sensor=True) #0.2
        utils.EzPickle.__init__(self)


class FetchPlaceEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental', use_fixed_goal=True, use_force_sensor=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack1.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, use_fixed_goal=use_fixed_goal, use_force_sensor=True)
        utils.EzPickle.__init__(self)

class FetchSlipperyPushEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='place', use_fixed_goal=True ):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/slippery_push.xml', num_blocks=1, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, use_fixed_goal=use_fixed_goal)
        utils.EzPickle.__init__(self)

        for i in range(len(self.sim.model.geom_friction)):
            self.sim.model.geom_friction[i] = [25e-2, 5.e-3, 1e-4] #[1e+00, 5.e-3, 1e-4]

class FetchBlockStackEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_info='incremental', use_fixed_goal=True, use_force_sensor=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/block_stacking.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_info=reward_info, use_fixed_goal=use_fixed_goal, use_force_sensor=True)
        utils.EzPickle.__init__(self)
