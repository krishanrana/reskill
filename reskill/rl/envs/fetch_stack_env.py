import numpy as np
import os
import pdb

from reskill.rl.envs import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[..., :-3] - goal_b[..., :-3], axis=-1)

# def goal_distance_red(goal_a, goal_b):
#     assert goal_a.shape == goal_b.shape
#     return np.linalg.norm(goal_a[..., :-3] - goal_b[..., :-3], axis=-1)


class FetchStackEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, num_blocks, n_substeps, gripper_extra_height, block_gripper,
        target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_info, goals_on_stack_probability=1.0, allow_blocks_on_stack=True, use_fixed_goal=True,
            all_goals_always_on_stack=False, use_force_sensor=False
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            num_blocks: number of block objects in environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_info ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_info = reward_info
        self.goals_on_stack_probability = goals_on_stack_probability
        self.allow_blocks_on_stack = allow_blocks_on_stack
        self.all_goals_always_on_stack = all_goals_always_on_stack
        self.use_fixed_goal = use_fixed_goal
        self.use_force_sensor = use_force_sensor
        self.position = []

        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000

        super(FetchStackEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # Heatmap Generation
    # ----------------------------

    def set_location_record_name(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_location_record()
        # if not self.record_write_dir:
        #     self.save_heatmap_picture(os.path.join(write_dir,'level.png'))
        self.location_record_write_dir = write_dir
        self.location_record_prefix = prefix
        self.location_record_file_number = 0

        return True

    def flush_location_record(self, create_new_empty_record=True):
        if self.location_record is not None and self.location_record_steps_recorded > 0:
            write_file = os.path.join(self.location_record_write_dir,"{}_{}".format(self.location_record_prefix,
                                                                           self.location_record_file_number))
            np.save(write_file, self.location_record[:self.location_record_steps_recorded])
            self.location_record_file_number += 1
            self.location_record_steps_recorded = 0

        if create_new_empty_record:
            self.location_record = np.empty(shape=(self.location_record_max_steps, 3), dtype=np.float32)

    def log_location(self, location):
        if self.location_record is not None:
            self.location_record[self.location_record_steps_recorded] = location
            self.location_record_steps_recorded += 1

            if self.location_record_steps_recorded >= self.location_record_max_steps:
                self.flush_location_record()

    # def save_heatmap_picture(self, filename):
    #     background_picture_np = self.debug_show_player_at_location(location_x=10000)
    #     im = Image.fromarray(background_picture_np)
    #     im.save(filename)



    # GoalEnv methods
    # ----------------------------

    def sub_goal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        goal_a = goal_a[..., :-3]
        goal_b = goal_b[..., :-3]
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i+1) * 3:(i + 2) * 3].shape

        return [
            np.linalg.norm(goal_a[..., i*3:(i+1)*3] - goal_b[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[..., -3:]
        sub_goals = goal[..., :-3]

        distances = [
            np.linalg.norm(gripper_pos - sub_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]

        return np.all([d > self.distance_threshold*2 for d in distances], axis=0)

    def compute_reward(self, achieved_goal, goal, force_sensor, gripper_pos, gripper_state, obs, info):
        # Compute distance between goal and the achieved goal.
        # print(self.reward_info)
        distances = self.sub_goal_distances(achieved_goal, goal)
        if self.reward_info == 'incremental':
            # Using incremental reward for each block in correct position
            reward = np.sum([-(d > self.distance_threshold).astype(np.float32) for d in distances], axis=0)
            reward = np.asarray(reward)
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_info == 'sparse':
            reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in distances], axis=0)
            reward = np.asarray(reward)
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_info == 'dense':
            d = goal_distance(achieved_goal, goal)
            if min([-(d > self.distance_threshold).astype(np.float32) for d in distances]) == 0:
                return 0
            return -d
        elif self.reward_info == 'custom':
            dist_r = np.linalg.norm(achieved_goal[0:2] - goal[0:2], axis=-1)
            dist_b = np.linalg.norm(achieved_goal[3:5] - goal[0:2], axis=-1)
            return -(dist_r + dist_b)
        elif self.reward_info == 'place_blue':
            dist_b = np.linalg.norm(achieved_goal[3:5] - goal[0:2], axis=-1)
            return np.exp(-20*dist_b)

        elif self.reward_info == 'place':
            dist_b = np.linalg.norm(achieved_goal[0:2] - goal[0:2], axis=-1)
            if dist_b < 0.03: 
                return 1.0
            else:
                return 0.0

        elif self.reward_info == "cleanup":

            if achieved_goal[3] > 1.30 and achieved_goal[3] < 1.5 and achieved_goal[4] > 0.96 and achieved_goal[4] < 1.16:
                if force_sensor > 19625:
                    rew_b = 1.0
                else:
                    rew_b = 0.0
            else:
                rew_b = 0.0
            
            if achieved_goal[0] > 1.30 and achieved_goal[0] < 1.5 and achieved_goal[1] > 0.96 and achieved_goal[1] < 1.16:
                if force_sensor > 19625:
                    rew_r = 1.0
                else:
                    rew_r = 0.0
            else:
                rew_r = 0.0


            return rew_b + rew_r

        elif self.reward_info == "cleanup_1block":
            
            if achieved_goal[0] > 1.30 and achieved_goal[0] < 1.5 and achieved_goal[1] > 0.96 and achieved_goal[1] < 1.16:
                if force_sensor > 19625:
                    return np.clip(np.exp(10*gripper_pos[2] - 5), 0, 1)    
                else:
                    return 0.0
            else:
                return 0.0

        elif self.reward_info == "stack_red":
            target  = obs['observation'][19:22]
            dist = np.linalg.norm(achieved_goal[0:2] - target[0:2], axis=-1)
            if dist < 0.08 and achieved_goal[2]>0.492 and achieved_goal[2]<0.498:
                if abs(np.mean(obs['force_sensor']))>1e-7 and dist < 0.04:
                    return 1.00
                else:
                    return 0.75
            else:
                return 0.0
               

            


    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.log_location(location=self.sim.data.get_site_xpos('robot0:grip'))
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        #rot_ctrl = [ 0.5, -0.5, 0.5, 0.5 ]  # 90 deg rotation of the original end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])


        achieved_goal = []

        for i in range(self.num_blocks):
        # for i in range(1):

            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                #object_i_rot.ravel(),
                object_i_velp.ravel(),
                #object_i_velr.ravel()
            ])


            # This is current location of the blocks
            achieved_goal = np.concatenate([
                achieved_goal, object_i_pos.copy()
            ])

        achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])

        achieved_goal = np.squeeze(achieved_goal)

        if self.use_force_sensor:
            self.sim.data.get_sensor('force_sensor') 
            force_reading = self.sim.data.sensordata # Read force sensor reading from tray
        else:
            force_reading = [0,0,0]



        # achieved_goal = np.squeeze(np.concatenate((object0_pos.copy(), object1_pos.copy())))
        #
        # obs = np.concatenate([
        #     grip_pos,
        #     object0_pos.ravel(), object1_pos.ravel(),
        #     object0_rel_pos.ravel(), object1_rel_pos.ravel(),
        #     gripper_state,
        #     object0_rot.ravel(), object1_rot.ravel(),
        #     object0_velp.ravel(), object1_velp.ravel(),
        #     object0_velr.ravel(), object1_velr.ravel(),
        #     grip_velp,
        #     gripper_vel,
        # ])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'force_sensor': force_reading.copy()
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        #lookat = [1.35268506, 0.74301371, 0.4008681]
        lookat = [1.27998563, 0.68635066, 0.35350562]

        for idx in range(3):
            self.viewer.cam.lookat[idx] = lookat[idx]
        # self.viewer.cam.distance = 0.8420461999474638 #2.5
        # self.viewer.cam.azimuth = 42.48803827751195 #132
        # self.viewer.cam.elevation = -22.612440191387563 #-14
        self.viewer.cam.distance = 0.8547035766991275
        self.viewer.cam.azimuth = 124.95215311004816
        self.viewer.cam.elevation = -22.488038277512022

    
    # def add_visual_capsule(self, scene, point1, point2, radius, rgba):
    #     """Adds one capsule to an mjvScene."""
    #     if scene.ngeom >= scene.maxgeom:
    #         return
    #     scene.ngeom += 1  # increment ngeom
    #     # initialise a new capsule, add it to the scene using mjv_makeConnector
    #     mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
    #                         mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
    #                         np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    #     mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
    #                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
    #                             point1[0], point1[1], point1[2],
    #                             point2[0], point2[1], point2[2])

    def _render_callback(self):

        # append position
        gripper_id = self.sim.model.site_name2id('robot0:grip')
        self.position.append(self.sim.data.geom_xpos[gripper_id].copy())


        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # print("sites offset: {}".format(sites_offset[0]))
        for i in range(self.num_blocks):

            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.goal[i*3:(i+1)*3] - sites_offset[i]

        # Visualise gripper position trajectory
        # if len(self.position) > 1:
        #     for i in range(len(self.position)-1):
        #         self.viewer.add_marker(pos=self.position[i], type=2, size=np.array([.005, .005, .005]), rgba=np.array([0, 1, 0, 0.05]), label="")

        
        self.sim.forward()

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        self.goal, goals, number_of_goals_along_stack = self._sample_goal(return_extra_info=True)

        if number_of_goals_along_stack == 0 or not self.allow_blocks_on_stack:
            number_of_blocks_along_stack = 0
        elif number_of_goals_along_stack < self.num_blocks:
            number_of_blocks_along_stack = np.random.randint(0, number_of_goals_along_stack+1)
        else:
            number_of_blocks_along_stack = np.random.randint(0, number_of_goals_along_stack)

        #TODO remove line
        # number_of_blocks_along_stack = 0

        # print("number_of_goals_along_stack: {} number_of_blocks_along_stack: {}".format(number_of_goals_along_stack, number_of_blocks_along_stack))

        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # prev_x_positions = [goal[:2] for goal in goals]  # Avoids blocks randomly being in goals
        prev_x_positions = [goals[0][:2]]
        for i, obj_name in enumerate(self.object_names):
            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
            assert object_qpos.shape == (7,)
            object_qpos[2] = 0.425 #0.425

            # add noise to angle info
            # object_qpos[3] = np.random.normal(loc=0, scale=0.002, size=1)
            # object_qpos[4] = np.random.normal(loc=0, scale=0.002, size=1)
            # object_qpos[5] = np.random.normal(loc=0, scale=0.002, size=1)


            if i < number_of_blocks_along_stack:
                object_qpos[:3] = goals[i]
                object_qpos[:2] += np.random.normal(loc=0, scale=0.002, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2].copy()

                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1 \
                        or np.any([np.linalg.norm(object_xpos - other_xpos) < 0.05 for other_xpos in prev_x_positions]):
                    object_xpos = self.initial_gripper_xpos[:2] + (self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                         size=2)-0.05) #-0.05) # TODO FOR THE CLEANUP ENV
                object_qpos[:2] = object_xpos


            prev_x_positions.append(object_qpos[:2])
            self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        
        

        self.sim.forward()

        obs = self._get_obs()
        return obs

    def _sample_goal(self, return_extra_info=False):

        max_goals_along_stack = self.num_blocks
        #TODO was 2
        if self.all_goals_always_on_stack:
            min_goals_along_stack = self.num_blocks
        else:
            min_goals_along_stack = 1

        if np.random.uniform() < 1.0 - self.goals_on_stack_probability:
            max_goals_along_stack = 0
            min_goals_along_stack = 0


        number_of_goals_along_stack = np.random.randint(min_goals_along_stack, max_goals_along_stack + 1)

        goal0 = None
        first_goal_is_valid = False
        while not first_goal_is_valid:
            goal0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if self.num_blocks > 4:
                if np.linalg.norm(goal0[:2] - self.initial_gripper_xpos[:2]) < 0.09:
                    continue
            first_goal_is_valid = True

        # goal0[0] = goal0[0] - 0.05
        goal0 += self.target_offset
        goal0[2] = self.height_offset
        goal0[1] += self.np_random.uniform(-0.35, 0.35, size=1)


        goals = [goal0]

        prev_x_positions = [goal0[:2]]
        goal_in_air_used = False
        for i in range(self.num_blocks - 1):
            if i < number_of_goals_along_stack - 1:
                goal_i = goal0.copy()
                goal_i[2] = self.height_offset + (0.05 * (i + 1))
            else:
                goal_i_set = False
                goal_i = None
                while not goal_i_set or np.any([np.linalg.norm(goal_i[:2] - other_xpos) < 0.06 for other_xpos in prev_x_positions]):
                    goal_i = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                    goal_i_set = True

                goal_i += self.target_offset
                goal_i[2] = self.height_offset
                

            prev_x_positions.append(goal_i[:2])
            goals.append(goal_i)
        goals.append([0.0, 0.0, 0.0])
        
        if self.use_fixed_goal:
            # 1 block env
            if self.num_blocks == 1:
                if self.reward_info == "stack_red":
                    goals = [[1.416193226, 0.9074910037, 0.4245288], [0.0, 0.0, 0.0]]
                elif self.reward_info == "cleanup_1block":
                    goals = [[1.416193226, 1.074910037, 0.4245288], [0.0, 0.0, 0.0]]
                else:
                    goals = [[1.316193226, 0.7074910037, 0.4245288], [0.0, 0.0, 0.0]]

            elif self.num_blocks == 2: # 2 block env
                goals = [[1.4, 1.06, 0.42], [1.40, 1.06, 0.42], [0.0, 0.0, 0.0]]

        if not return_extra_info:
            return np.concatenate(goals, axis=0).copy()
        else:
            return np.concatenate(goals, axis=0).copy(), goals, number_of_goals_along_stack

    def _is_success(self, achieved_goal, desired_goal):

        distances = self.sub_goal_distances(achieved_goal, desired_goal)
        if sum([-(d > self.distance_threshold).astype(np.float32) for d in distances]) == 0:
            return True
        else:
            return False


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

            # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]
