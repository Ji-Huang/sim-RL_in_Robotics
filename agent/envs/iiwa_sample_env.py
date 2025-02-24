"""
A sample Env class inheriting from the DART-Unity Env for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment
Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space
(position-only or pose/SE(3)) of the end-effector.

action_by_pd_control method can be called to implement a Proportional-Derivative control law instead of an RL policy.

Note: Coordinates in the Unity simulator are different from the ones in DART which used here:
      The mapping is [X, Y, Z] of Unity is [-y, z, x] of DART
"""

import numpy as np
import os
import math

import cv2
import base64

import dartpy as dart
from gym import spaces
from envs_dart.iiwa_dart_unity import IiwaDartUnityEnv

from .box import BoxObject
from pose_estimation.image2num import Image2Num

import time

# Import when images are used as state representation
# import cv2
# import base64

class IiwaSampleEnv(IiwaDartUnityEnv):
    def __init__(self, localization_config_dict, max_ts, orientation_control, use_ik, ik_by_sns,
                 state_type, enable_render=False, task_monitor=False, 
                 with_objects=False, target_mode="random", target_path="/misc/generated_random_targets/cart_pose_7dof.csv", goal_type="target",
                 joints_safety_limit=0.0, max_joint_vel=20.0, max_ee_cart_vel=10.0, max_ee_cart_acc =3.0, max_ee_rot_vel=4.0, max_ee_rot_acc=1.2,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, 0, 0, 0, 0],
                 robotic_tool="3_gripper", box_dim=[0.2, 0.2, 0.3], end_height=0.3, start_side=None, allow_variable_horizon=False, allow_dead_zone=False, 
                 min_pos_distance=0.06, camera_position=[0.0, 1.6, 2.6], camera_rotation=[150.0, 0.0, 180],
                 rng=None, env_id=0):
        self.use_localization = localization_config_dict["use_localization"]
        if self.use_localization:
            data_cfg = localization_config_dict["data_cfg"]
            model_cfg = localization_config_dict["model_cfg"]
            weightfile = localization_config_dict["weightfile"]

            self.image2num = Image2Num(data_cfg, model_cfg, weightfile)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random

        self.allow_variable_horizon = allow_variable_horizon
        self.allow_dead_zone = allow_dead_zone

        self.stop_count = 0
        self.stop_interval = 1

        if start_side is None:
            self.start_side = np.random.choice([-1, 1])
        else:
            self.start_side = start_side

        if self.start_side == 1:
            initial_positions = [1.1975091695785522, 1.2971241474151611, -0.0100827906280756, -0.32704633474349976, -0.020220449194312096, 1.4551894664764404, 1.1442426443099976]
        else:
            initial_positions = [-1.1839678287506104, 1.326664924621582, -0.06719449162483215, -0.2669602036476135, 0.0842360332608223, 1.454065203666687, -1.1522223949432373]
        
        # range of vertical, horizontal pixels for the DART viewer
        viewport = (0, 0, 500, 500)

        self.goal_type = goal_type # Target or box 

        ##############################################################################
        # Set Limits -> Important: Must be set before calling the super().__init__() #
        ##############################################################################

        # Variables below exist in the parent class, hence the names should not be changed                            #
        # Min distance to declare that the target is reached by the end-effector, adapt the values based on your task #
        self.MIN_POS_DISTANCE = min_pos_distance    # [m]
        self.MIN_ROT_DISTANCE = 0.3                 # [rad]

        self.JOINT_POS_SAFE_LIMIT = np.deg2rad(joints_safety_limit) # Manipulator joints safety limit

        # admissible range for joint positions, velocities, accelerations, # 
        # and torques of the iiwa kinematic chain                          #
        self.MAX_JOINT_POS = np.deg2rad([170, 120, 170, 120, 170, 120, 175]) - self.JOINT_POS_SAFE_LIMIT  # [rad]: based on the specs
        self.MIN_JOINT_POS = -self.MAX_JOINT_POS

        # Joint space #
        self.MAX_JOINT_VEL = np.deg2rad(np.full(7, max_joint_vel))                                        # np.deg2rad([85, 85, 100, 75, 130, 135, 135])  # [rad/s]: based on the specs
        self.MAX_JOINT_ACC = 3.0 * self.MAX_JOINT_VEL                                                     # [rad/s^2]: just approximation due to no existing data
        self.MAX_JOINT_TORQUE = np.array([320, 320, 176, 176, 110, 40, 40])                               # [Nm]: based on the specs

        # admissible range for Cartesian pose translational and rotational velocities, #
        # and accelerations of the end-effector                                        #
        self.MAX_EE_CART_VEL = np.full(3, max_ee_cart_vel)                                                # np.full(3, 10.0) # [m/s] --- not optimized values for sim2real transfer
        self.MAX_EE_CART_ACC = np.full(3, max_ee_cart_acc)                                                # np.full(3, 3.0) # [m/s^2] --- not optimized values
        self.MAX_EE_ROT_VEL = np.full(3, max_ee_rot_vel)                                                  # np.full(3, 4.0) # [rad/s] --- not optimized values
        self.MAX_EE_ROT_ACC = np.full(3, max_ee_rot_acc)                                                  # np.full(3, 1.2) # [rad/s^2] --- not optimized values

        ##################################################################################
        # End set limits -> Important: Must be set before calling the super().__init__() #
        ##################################################################################

        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns,
                         state_type=state_type, robotic_tool=robotic_tool, enable_render=enable_render, task_monitor=task_monitor,
                         with_objects=with_objects, target_mode=target_mode, target_path=target_path, viewport=viewport,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions, env_id=env_id)

        # Collision happened for this env when set to 1. In case the manipulator is spanwed in a different position than the #
        # default vertical, for the remaining of the episode zero velocities are sent to UNITY                               #
        self.collided_env = 0
        self.reset_flag = False

        ################################
        # Set initial joints positions #
        ################################
        self.random_initial_joint_positions = random_initial_joint_positions                                                 # True, False
        self.initial_positions = np.asarray(initial_positions, dtype=float) if initial_positions is not None else None

        # Initial positions flag for the manipulator after reseting. 1 means different than the default vertical position #
        # In that case, the environments should terminate at the same time step due to UNITY synchronization              #
        if((self.initial_positions is None or np.count_nonzero(initial_positions) == 0) and self.random_initial_joint_positions == False):
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1

        # Clip gripper action to this limit #
        if(self.robotic_tool.find("3_gripper") != -1):
            self.gripper_clip = 90
        elif(self.robotic_tool.find("2_gripper") != -1):
            self.gripper_clip = 250
        else:
            self.gripper_clip = 90

        self.transform_ee_initial = None
        self.save_image = False  # change it to True to save images received from Unity into jpg files

        if self.save_image:
            self.save_image_folder = self.get_save_image_folder()

        # Some attributes that are initialized in the parent class:
        # self.reset_counter --> keeps track of the number of resets performed
        # self.reset_state   --> reset vector for initializing the Unity simulator at each episode
        # self.tool_target   --> initial position for the gripper state

        # helper methods that can be called from the parent class:
        # self._convert_vector_dart_to_unity(vector) -- transforms a [x, y, z] position vector
        # self._convert_vector_unity_to_dart(vector)
        # self._convert_rotation_dart_to_unity(matrix) -- transforms a 3*3 rotation matrix
        # self._convert_rotation_unity_to_dart(matrix)
        # self._convert_quaternion_dart_to_unity(quaternion) -- transforms a [w, x, y, z] quaternion vector
        # self._convert_quaternion_unity_to_dart(quaternion)
        # self._convert_angle_axis_dart_to_unity(vector) -- transforms a [rx, ry, rz] logmap representation of an Angle-Axis
        # self._convert_angle_axis_unity_to_dart(vector)
        # self._convert_pose_dart_to_unity(dart_pose, unity_in_deg=True) -- transforms a pose -- DART pose order [rx, ry, rz, x, y, z] -- Unity pose order [X, Y, Z, RX, RY, RZ]
        # self.get_rot_error_from_quaternions(target_quat, current_quat) -- calculate the rotation error in angle-axis given orientation inputs in quaternion format
        # self.get_rot_ee_quat()  -- DART coords -- get the orientation of the ee in quaternion
        # self.get_ee_pos()       -- DART coords -- get the position of the ee
        # self.get_ee_orient()    -- DART coords -- get the orientation of the ee in angle-axis
        # self.get_ee_pose()      -- DART coords -- get the pose of the ee
        # self.get_ee_pos_unity()           -- get the ee position in unity coords
        # self.get_ee_orient_unity()        -- get the ee orientation in angle-axis format in unity coords
        # self.get_ee_orient_euler_unity()  -- get the ee orientation in XYZ euler angles in unity coords

        # the lines below should stay as it is
        self.MAX_EE_VEL = self.MAX_EE_CART_VEL
        self.MAX_EE_ACC = self.MAX_EE_CART_ACC
        if orientation_control:
            self.MAX_EE_VEL = np.concatenate((self.MAX_EE_ROT_VEL, self.MAX_EE_VEL))
            self.MAX_EE_ACC = np.concatenate((self.MAX_EE_ROT_ACC, self.MAX_EE_ACC))

        # the lines below wrt action_space_dimension should stay as it is
        # self.action_space_dimension = self.n_links  # there would be 7 actions in case of joint-level control
        # if use_ik:
        #     # There are three cartesian coordinates x,y,z for inverse kinematic control
        #     self.action_space_dimension = 3
        #     if orientation_control:
        #         # and the three rotations around each of the axis
        #         self.action_space_dimension += 3

        # if self.with_gripper:
        #     self.action_space_dimension += 1  # gripper velocity

        # Variables below exist in the parent class, hence the names should not be changed
        tool_length = 0.2  # [m] allows for some tolerances in maximum observation

        # x,y,z of TCP: maximum reach of arm plus tool length in meters
        ee_pos_high = np.array([0.95 + tool_length, 0.95 + tool_length, 1.31 + tool_length])
        ee_pos_low = -np.array([0.95 + tool_length, 0.95 + tool_length, 0.39 + tool_length])

        self.observation_indices = {'obs_len': 0}

        if orientation_control:
            # rx,ry,rz of TCP: maximum orientation in radians without considering dexterous workspace
            self.observation_indices['ee_rot'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += 3

        # and distance to target position (dx,dy,dz), [m]
        self.observation_indices['ee_pos'] = self.observation_indices['obs_len']
        self.observation_indices['obs_len'] += 3

        # and joint positions [rad] and possibly velocities [rad/s]
        if 'a' in self.state_type:
            self.observation_indices['joint_pos'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += self.n_links
        if 'v' in self.state_type:
            self.observation_indices['joint_vel'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += self.n_links

        # the lines below should stay as it is.                                                                        #
        # Important:        Adapt only if you use images as state representation, or your task is more complicated     #
        # Good practice:    If you need to adapt several methods, inherit from IiwaSampleEnv and define your own class #
        
        self.action_space_dimension = 2
        self.action_space = spaces.Box(-np.ones(self.action_space_dimension, dtype=np.float32), np.ones(self.action_space_dimension, dtype=np.float32), dtype=np.float32)

        high = np.empty(0)
        low = np.empty(0)

        box_rot_high = np.full(3, np.pi)
        high = np.append(high, box_rot_high)
        low = np.append(low, -box_rot_high)

        high = np.append(high, np.ones(3, dtype=np.float32))
        low = np.append(low, -np.ones(3, dtype=np.float32))

        high = np.append(high, box_rot_high)
        low = np.append(low, -box_rot_high)

        high = np.append(high, np.ones(3, dtype=np.float32))
        low = np.append(low, -np.ones(3, dtype=np.float32))

        # high = np.append(high, np.ones(3, dtype=np.float32))
        # low = np.append(low, -np.ones(3, dtype=np.float32))

        high = np.append(high, np.ones(3, dtype=np.float32))
        low = np.append(low, -np.ones(3, dtype=np.float32))

        #target_reached
        high = np.append(high, np.ones(1, dtype=np.float32))
        low = np.append(low, -np.zeros(1, dtype=np.float32))
        
        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

        length, height, width = box_dim
        self.box = BoxObject(width, length, height, center=[0.4, 0.4, height / 2.0 + 0.005], rotation=np.deg2rad([0.0, 0.0, 0.0]))
        self.box_dim = box_dim

        self.reaching_target = [0.0, 0.0]
        self.pushing_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.final_target = self.pushing_target

        self.random_mul = -1
        self.end_height = end_height

        self.best_fraction_reduced_goal_dist = 0.
        
    # Generate a random integer within a given range
    def secure_random_uniform(self, a, b):
        if b < a:
            temp_b = b
            b = a
            a = temp_b

        return self.rng.uniform(a, b)
    
    def _compute_goal_distance(self):
        object_pose = self.box.get_pose()

        xy_block = np.array(object_pose[-3:-1])
        theta_block = np.array(object_pose[2])
        xy_target = np.array(self.final_target[-3:-1])
        theta_target = np.array(self.final_target[2])
        
        abs_error = abs(theta_target - theta_block)
    
        # Take the shorter distance considering the symmetry
        theta_error = min(abs_error, 2*math.pi - abs_error)

        # Block has 2-way symmetry.
        while theta_error > np.pi/2:
            theta_error -= np.pi
        while theta_error < -np.pi/2:
            theta_error += np.pi
            
        diff = np.linalg.norm(xy_block - xy_target)

        return diff, theta_error

    def _is_can_reach(self, target_xy, min=0.6, max=0.7):
        if (np.linalg.norm(target_xy) > min) and (np.linalg.norm(target_xy) < max):
            return True
        return False
    
    def _update_env_flags(self):
        ###########################################################################################
        # collision happened or joints limits overpassed                                          #
        # Important: in case the manipulator is spanwed in a different position than the default  #
        #            vertical, for the remaining of the episode zero velocities are sent to UNITY #
        #            see _send_actions_and_update() method in simulator_vec_env.py                #
        ###########################################################################################
        if self.unity_observation['collision_flag'] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1
            # Reset when we have a collision only when we spawn the robot to the default #
            # vertical position, else wait the episode to finish                         #
            # Important: you may want to reset anyway depending on your task - adapt     #
            if self.flag_zero_initial_positions == 0:
                self.reset_flag = True

    def get_save_image_folder(self):
        path = os.path.dirname(os.path.realpath(__file__))
        env_params_list = [self.max_ts, self.dart_sim.orientation_control, self.dart_sim.use_ik, self.dart_sim.ik_by_sns,
                           self.state_type, self.target_mode, self.goal_type, self.random_initial_joint_positions,
                           self.robotic_tool]
        env_params = '_'.join(map(str, env_params_list))
        idx = 0
        while True:
            save_place = path + '/misc/unity_image_logs/' + self.env_key + '_' + env_params + '_%s' % idx
            if not os.path.exists(save_place):
                save_place += '/'
                os.makedirs(save_place)
                break
            idx += 1
        return save_place

    def create_target(self):
        """
            defines the target to reach per episode, this should be adapted by the task

            should always return rx,ry,rz,x,y,z in order, -> dart coordinates system
            i.e., first target orientation rx,ry,rz in radians, and then target position x,y,z in meters
                in case of orientation_control=False --> rx,ry,rz are irrelevant and can be set to zero

            _random_target_gen_joint_level(): generate a random sample target in the reachable workspace of the iiwa

            :return: Cartesian pose of the target to reach in the task space (dart coordinates)
        """

        # Default behaviour # 
        if(self.target_mode == "None"): 
            rx, ry, rz = 0.0, np.pi, 0.0
            target = None
            while True:
                x, y, z = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0.2

                if 0.4*0.4 < x*x + y*y < 0.8*0.8:
                    target = rx, ry, rz, x, y, z
                    break

        elif self.target_mode == "import":
            target = self._recorded_next_target()

        elif self.target_mode == "random":
            target = self._random_target()

        elif self.target_mode == "random_joint_level":
            target = self._random_target_gen_joint_level() # Sample always a rechable target

        elif self.target_mode == "fixed":
            target = self._fixed_target()

        elif self.target_mode == "fixed_joint_level":
            target = self._fixed_target_gen_joint_level()

        elif self.target_mode == "random_push":
            rotation = np.deg2rad([0.0, -180, 0.0]).tolist() #self.box.get_rotation() 
            height = 0.01

            target = rotation + [self.secure_random_uniform(0.5, 0.8), self.secure_random_uniform(-1*self.start_side*0.15, -1*self.start_side*0.7)] + [height]
        
        elif self.target_mode == "straight_push":
            rotation = self.box.get_rotation()
            height = 0.01

            box_position = self.box.get_position()[:-1]
            targets = self.box.calculate_target_points(distance=self.secure_random_uniform(0.2, 0.5))
            init_pos = [0.6, self.start_side*0.6]

            dist_to_box = np.linalg.norm(np.array(init_pos) - np.array(box_position))
            if np.linalg.norm(np.array(init_pos) - np.array(targets[0])) > dist_to_box:
                target = rotation + targets[0].tolist() + [height]
            else:
                target = rotation + targets[1].tolist() + [height]

                
        return target
    
    def _object_candidate(self):
        length, height, width = self.box_dim
        # depending on your task, positioning the object might be necessary, start from the following sample code
        # sample code to position the object
        z = height / 2.0 + 0.005 # use it for tasks including object such as grasping or pushing
        
        x, y = [self.secure_random_uniform(0.4, 0.65), 
                self.secure_random_uniform(self.start_side*0.05, self.start_side*0.55)]
        
        rx, ry, rz = np.deg2rad([0.0, 0.0, self.secure_random_uniform(-35.0, 35.0)])
        self.box.set_pose(rotation=[rx, ry, rz], center=[x, y, z])
        self.pushing_target =  self.create_target()
        self.reaching_target = self.box.get_backside_center(self.pushing_target[-3:-1]).tolist()

        return rx, ry, rz, x, y, z
    
    def generate_object(self):
        """
            defines the initial box position per episode
            should always return rx,ry,rz,x,y,z in order,
            i.e., first box orientation rx,ry,rz in radians, and then box position x,y,z in meters

            :return: Cartesian pose of the initial box position in the task space
        """
        
        if self.target_mode == "test":
            self.pushing_target = np.deg2rad([0.0, 0.0, 15.0]).tolist() + [0.7, -0.4, 0.01]
            rx, ry, rz, x, y, z = np.deg2rad([0.0, 0.0, 0.0]).tolist() + [0.5, 0.4, 0.1005]
        else:
            rx, ry, rz, x, y, z = self._object_candidate()
            while not (self._is_can_reach(self.reaching_target) and self._is_can_reach(self.pushing_target[-3:-1], 0.5, 0.8)):
                rx, ry, rz, x, y, z = self._object_candidate()
        
        # self.pushing_target = np.deg2rad([0.0, 0.0, 15.0]).tolist() + [0.7, -0.4, 0.01]
        # rx, ry, rz, x, y, z = np.deg2rad([0.0, 0.0, 0.0]).tolist() + [0.5, 0.4, 0.1005]
        # print("box:", self.box.get_pose())
        # print("reaching_target:", self.reaching_target, "| dis:", np.linalg.norm(self.reaching_target))
        # print("pushing_target:", self.pushing_target[-3:-1], "| dis:", np.linalg.norm(self.pushing_target[-3:-1]))
        return rx, ry, rz, x, y, z

    def get_state(self):
        """
           defines the environment state, this should be adapted by the task

           get_pos_error(): returns Euclidean error from the end-effector to the target position
           get_rot_error(): returns Quaternion error from the end-effector to the target orientation

           :return: state for the policy training
        """
        state = np.empty(0)
        state = np.append(state, self.box.get_pose())
        
        state = np.append(state, self.final_target)
        # state = np.append(state, np.concatenate([self.reaching_target, [self.end_height]]))

        ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
        state = np.append(state, ee.getTransform().translation())

        goal_distance, theta_error = self._compute_goal_distance()
        target_reached = (goal_distance < self.MIN_POS_DISTANCE)
        state = np.append(state, int(target_reached))

        # the lines below should stay as it is
        self.observation_state = np.array(state)
        # print(self.observation_state)

        return self.observation_state

    def get_reward(self):
        """
           defines the environment reward, this should be adapted by the task

           :param action: is the current action decided by the RL agent

           :return: reward for the policy training
        """
        goal_distance, theta_error = self._compute_goal_distance()

        fraction_reduced_goal_distance = 1.0 - (
            goal_distance / self._init_goal_distance)
        if fraction_reduced_goal_distance > self.best_fraction_reduced_goal_dist:
            self.best_fraction_reduced_goal_dist = fraction_reduced_goal_distance
        
        self.reward_state = self.best_fraction_reduced_goal_dist

        return self.reward_state

    def get_terminal_reward(self):
        """
           checks if the target is reached

           get_pos_distance(): returns norm of the Euclidean error from the end-effector to the target position
           get_rot_distance(): returns norm of the Quaternion error from the end-effector to the target orientation

           Important: by default a 0.0 value of a terminal reward will be given to the agent. To adapt it please refer to the config.py,
                      in the reward_dict. This terminal reward is given to the agent during reset in the step() function in the simulator_vec_env.py

           :return: a boolean value representing if the target is reached within the defined threshold
        """
        goal_distance, theta_error = self._compute_goal_distance()
        
        target_reached = False
        if goal_distance < self.MIN_POS_DISTANCE:
            # if abs(theta_error) < self.MIN_ROT_DISTANCE:
            target_reached = True

        return target_reached
    
    def get_terminal(self):
        """
           checks the terminal conditions for the episode - for reset

           :return: a boolean value indicating if the episode should be terminated
        """

        if (self.time_step > self.max_ts):
            self.reset_flag = True

        if self.allow_variable_horizon:
            goal_distance, theta_error = self._compute_goal_distance()
            # print("goal_distance: ", goal_distance, ", abs theta_error:", abs(theta_error), ", theta_error (deg):", np.rad2deg(theta_error))
            if goal_distance < self.MIN_POS_DISTANCE: #and abs(theta_error) < self.MIN_ROT_DISTANCE:
                if self.stop_count > self.stop_interval:
                    self.reset_flag = True
                self.stop_count += 1
            else:
                self.stop_count = 0

        if not self.allow_dead_zone:
            ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')
            ee_pos = ee.getTransform().translation()[-3:-1]
            if (np.linalg.norm(ee_pos) > 0.85) or (np.linalg.norm(ee_pos) < 0.42): #dead zone
                self.reset_flag = True

        return self.reset_flag

    def update_action(self, action):
        """
           converts env action to the required unity action by possibly using inverse kinematics from DART

           self.dart_sim.command_from_action() --> changes the action from velocities in the task space (position-only 'x,y,z' or complete pose 'rx,ry,rz,x,y,z')
                                                   to velocities in the joint space of the kinematic chain (j1,...,j7)

           :param action: The action vector decided by the RL agent, acceptable range: [-1,+1]

                          It should be a numpy array with the following shape: [arm_action] or [arm_action, tool_action]
                          in case 'robotic_tool' is a gripper, tool_action is always a dim-1 scalar value representative of the normalized gripper velocity

                          arm_action has different dim based on each case of control level:
                              in case of use_ik=False    -> is dim-7 representative of normalized joint velocities
                              in case of use_ik=True     -> there would be two cases:
                                  orientation_control=False  -> of dim-3: Normalized EE Cartesian velocity in x,y,z DART coord
                                  orientation_control=True   -> of dim-6: Normalized EE Rotational velocity in x,y,z DART coord followed by Normalized EE Cartesian velocity in x,y,z DART coord

           :return: the command to send to the Unity simulator including joint velocities and possibly the gripper position
        """

        ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')
        ee_pos = ee.getTransform().translation()[-3:]

        # the lines below should stay as it is
        self.action_state = action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        z_action = np.clip([0.3 - ee_pos[-1]], -np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32))

        action_ = np.concatenate([[0.0, 0.0, 0.0], action, z_action])

        # use INVERSE KINEMATICS #
        if self.dart_sim.use_ik:
            task_vel = self.MAX_EE_VEL * action_
            joint_vel = self.dart_sim.command_from_action(task_vel, normalize_action=False)
        else:
            joint_vel = self.MAX_JOINT_VEL * action_

        # append tool action #
        unity_action = joint_vel
        if self.with_gripper:
            unity_action = np.append(unity_action, [float(self.tool_target)])

        return unity_action

    def update(self, observation, time_step_update=True):
        """
            converts the unity observation to the required env state defined in get_state()
            with also computing the reward value from the get_reward(...) and done flag,
            it increments the time_step, and outputs done=True when the environment should be reset

            important: it also updates the dart kinematic chain of the robot using the new unity simulator observation.
                       always call this function, once you have send a new command to unity to synchronize the agent environment

            :param observation: is the observation received from the Unity simulator within its [X,Y,Z] coordinate system
                                'joint_values':       indices [0:7],
                                'joint_velocities':   indices [7:14],
                                'ee_position':        indices [14:17],
                                'ee_orientation':     indices [17:21],
                                'target_position':    indices [21:24],
                                'target_orientation': indices [24:28],
                                'object_position':    indices [28:31],
                                'object_orientation': indices [31:35],
                                'gripper_position':   indices [35:36], ---(it is optional, in case a gripper is enabled)
                                'collision_flag':     indices [36:37], ---([35:36] in case of without gripper)

            :param time_step_update: whether to increase the time_step of the agent

            :return: The state, reward, episode termination flag (done), and an info dictionary
        """
        # print("join posisition:", self.dart_sim.chain.getPositions().tolist())
        # ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')
        # ee_pos = ee.getTransform().translation()[-3:]
        # print("end effector:", ee_pos)
        # print("distance:", np.linalg.norm(ee_pos[:-1]))
        # print("")
        start_time = time.time()
        self.current_obs = observation['Observation']
        self.box.update_box(self.current_obs, unity=True)

        if self.use_localization:
            image = self._retrieve_image(observation)
            box_x, box_y, box_rz = self.image2num.get_box_pos(image)

            print("[obs box x,y]:\t\t", self.box.center[:-1])
            print("[predicted box x,y]:\t", box_x, ", ", box_y)

            pred_error = np.array(self.box.center[:-1]) - np.array([box_x, box_y])
            print("[prediction error]:", pred_error)
            self.box.update_position_from_localization(box_x, box_y, box_rz)
        
        if self.save_image:
            self._unity_retrieve_observation_image(observation['ImageData'])

        # the methods below handles synchronizing states of the DART kinematic chain with the observation from Unity
        # hence it should be always called
        self._unity_retrieve_observation_numeric(observation['Observation'])
        self._update_dart_chain()
        self._update_env_flags()

        # Class attributes below exist in the parent class, hence the names should not be changed
        if(time_step_update == True):
            self.time_step += 1

        stop_time = time.time()

        self._state = self.get_state()
        self._reward = self.get_reward()
        self._done = bool(self.get_terminal())
        self._info = {"success": False, "iteration_time": stop_time-start_time}                   # Episode was successful. For now it is set at simulator_vec_env.py before reseting, step() method. Adapt if needed
        
        if self._done:
            self._info["success"] = True
            self._info["terminal_observation"] = self._state

        if self.use_localization:
            self._info["pred_error"] = pred_error

        self.prev_action = self.action_state

        return self._state, self._reward, self._done, self._info

    def reset(self, temp=False):
        """
            resets the DART-gym environment and creates the reset_state vector to be sent to Unity

            :param temp: not relevant

            :return: The initialized state
        """
        # takes care of resetting the DART chain and should stay as it is
        self._state = super().reset(random_initial_joint_positions=self.random_initial_joint_positions, initial_positions=self.initial_positions)

        self.transform_ee_initial = None
        self.collided_env = 0

        # spawn the object in UNITY: by default a green box is spawned
        # mapping from DART to Unity coordinates -- Unity expects orientations in degrees
        object_positions_mapped = self._convert_pose_dart_to_unity(self.generate_object())

        self.final_target = self.pushing_target # draw a red target
        
        # sets the initial reaching target for the current episode,
        # should be always called in the beginning of each episode,
        # you might need to call it even during the episode run to change the reaching target for the IK-P controller
        self.set_target(self.final_target)

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        # the mapping below for the target should stay as it is, unity expects orientations in degrees
        target_positions = self.dart_sim.target.getPositions().tolist()
        target_positions_mapped = self._convert_pose_dart_to_unity(target_positions)
  
        self.reset_state = active_joints + joint_positions + joint_velocities\
                           + target_positions_mapped + object_positions_mapped
 
        if self.with_gripper:
            # initial position for the gripper state, accumulates the tool_action velocity received in update_action
            self.tool_target = 0.0  # should be in range [0.0,90.0] for 3-gripper or [0.0,250.0] for 2-gripper
            self.reset_state = self.reset_state + [self.tool_target]

        self.reset_counter += 1
        self.reset_flag = False
        self.stop_count = 0

        self._init_goal_distance, self._init_theta_error = self._compute_goal_distance()
        self.best_fraction_reduced_goal_dist = 0.

        return self._state

    def _retrieve_image(self, observation):
        """
            if the image observations are enabled at the simulator, it passes image data in the observation in addition
            to the numeric data. This method shows how the image data can be accessed, saved locally, processed, etc.
        """
        # how to access the image data when it is provided by the simulator
        base64_bytes = observation['ImageData'][0].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        image = np.frombuffer(image_bytes, np.uint8)
        # image_bw = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        return image[:, :, :]
    
    ####################
    # Commands related #
    ####################
    def action_by_p_control(self, coeff_kp_lin, coeff_kp_rot):
        """
            computes the task-space velocity commands proportional to the reaching target error

            :param coeff_kp_lin: proportional coefficient for the translational error
            :param coeff_kp_rot: proportional coefficient for the rotational error

            :return: The action in task space
        """

        action_lin = coeff_kp_lin * self.dart_sim.get_pos_error()
        action = action_lin

        if self.dart_sim.orientation_control:
            action_rot = coeff_kp_rot * self.dart_sim.get_rot_error()
            action = np.concatenate(([action_rot, action]))

        if self.with_gripper:
            tool_vel = 0.0                         # zero velocity means no gripper movement - should be adapted for the task
            action = np.append(action, [tool_vel])

        return action
