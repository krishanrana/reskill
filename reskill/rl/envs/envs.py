from reskill.rl.envs import fetch_stack_env
from gym import utils

DISTANCE_THRESHOLD = 0.04


# Custom Environments

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
