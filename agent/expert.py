import numpy as np
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch as th
import gymnasium as gym
from imitation.policies import base
from gymnasium import spaces

from envs.box import BoxObject

import pushing_info as pushing_info_module

class PushingPolicy(base.HardCodedPolicy):
    def __init__(self, venv, box_dim=[0.2, 0.2, 0.3]):
        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

        self.experts = np.array([ExpertPushing(box_dim=box_dim) for i in range(venv.nenvs)])

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        
        for idx in np.argwhere(episode_start).flatten():
            self.experts[idx].reset()
        
        return super().predict(observation, state, episode_start, deterministic)
    
    def _predict(self, obs: th.Tensor, deterministic: bool = False):
        np_actions = []
        np_obs = obs.detach().cpu().numpy()
        
        for idx in range(len(np_obs)):
            np_actions.append(self._choose_action(idx, np_obs[idx]))
            
        np_actions = np.stack(np_actions, axis=0)
        th_actions = th.as_tensor(np_actions, device=self.device)
        
        return th_actions
    
    def _choose_action(
        self,
        idx: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""
        
        return self.experts[idx].action(obs)

class ExpertPushing():
    def __init__(self, box_dim=[0.2, 0.2, 0.3]):
        length, height, width = box_dim
        self.box = BoxObject(width, length, height, center=[0.4, 0.4, height / 2.0 + 0.005], rotation=np.deg2rad([0.0, 0.0, 0.0]))
        self.phase = "move_to_pre_block"
        self.target_reached = False
    
    def extract_from_observation(self, observation):
        self.box.update_box(observation)
        self.final_target = observation[6:12]
        # self.reaching_target = observation[12:15]
        self.ee_pos = observation[-4:-1]

        if observation[-1]:
            self.target_reached = True
    
    def reset(self):
        self.phase = "move_to_pre_block"
        self.target_reached = False

    def get_theta_from_vector(self, vector):
        return np.arctan2(vector[1], vector[0])

    def theta_to_rotation2d(self, theta):
        r = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        return r

    def rotate(self, theta, xy_dir_block_to_ee):
        rot_2d = self.theta_to_rotation2d(theta)
        return rot_2d @ xy_dir_block_to_ee

    def _get_action_info(self):
        xy_block = self.box.get_position()[:2]
        theta_block = self.box.get_rotation()[-1]
        xy_target = self.final_target[-3:-1]
        xy_ee = self.ee_pos[:-1]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (
            xy_block_to_target) / np.linalg.norm(xy_block_to_target)
        theta_to_target = self.get_theta_from_vector(xy_dir_block_to_target)

        theta_error_ = theta_to_target - theta_block

        theta_error = theta_error_ + (np.pi/2.)
        # Block has 2-way symmetry.
        while theta_error > np.pi/2:
            theta_error -= np.pi
        while theta_error < -np.pi/2:
            theta_error += np.pi

        # print("theta_error:", theta_error, ", theta_error2:", theta_error2)
        xy_pre_block = xy_block + -xy_dir_block_to_target * ((self.box.length/2.0) + 0.05)
        xy_nexttoblock = xy_block + -xy_dir_block_to_target * ((self.box.length/2.0) + 0.03)
        xy_touchingblock = xy_block + -xy_dir_block_to_target * 0.01
        xy_delta_to_nexttoblock = xy_nexttoblock - xy_ee
        xy_delta_to_touchingblock = xy_touchingblock - xy_ee

        xy_block_to_ee = xy_ee - xy_block
        xy_dir_block_to_ee = xy_block_to_ee / np.linalg.norm(xy_block_to_ee)

        theta_threshold_to_orient = 0.261799 #15 degree
        theta_threshold_flat_enough = 0.03
        return pushing_info_module.PushingInfo(
            xy_block=xy_block,
            xy_ee=xy_ee,
            xy_pre_block=xy_pre_block,
            xy_delta_to_nexttoblock=xy_delta_to_nexttoblock,
            xy_delta_to_touchingblock=xy_delta_to_touchingblock,
            xy_dir_block_to_ee=xy_dir_block_to_ee,
            theta_threshold_to_orient=theta_threshold_to_orient,
            theta_threshold_flat_enough=theta_threshold_flat_enough,
            theta_error=theta_error)

    def _get_move_to_preblock(self, xy_pre_block, xy_ee):
        max_step_velocity = 0.1
        # Go 5 cm away from the block, on the line between the block and target.
        xy_delta_to_preblock = xy_pre_block - xy_ee
        diff = np.linalg.norm(xy_delta_to_preblock)
        # print("[move_to_pre_block]", " xy_delta_to_preblock:", xy_delta_to_preblock, ", xy_pre_block:", xy_pre_block, ", xy_ee: ", xy_ee, ", diff: ", diff)
        if diff < 0.005:
            self.phase = "move_to_block"
        xy_delta = xy_delta_to_preblock
        return xy_delta, max_step_velocity
    
    def _get_move_to_block(
            self, xy_delta_to_nexttoblock, theta_threshold_to_orient, theta_error):
        diff = np.linalg.norm(xy_delta_to_nexttoblock)
        # print("[move_to_block]", " xy_delta_to_nexttoblock:", xy_delta_to_nexttoblock, ", diff: ", diff)
        if diff < 0.005:
            self.phase = "push_block"
        # If need to re-oorient, then re-orient.
        if theta_error > theta_threshold_to_orient:
            self.phase = "orient_block_left"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "orient_block_right"
        # Otherwise, push into the block.
        xy_delta = xy_delta_to_nexttoblock
        return xy_delta
    
    def _get_push_block(
            self, theta_error, theta_threshold_to_orient, xy_delta_to_touchingblock):
        
        # If need to reorient, go back to move_to_pre_block, move_to_block first.
        if theta_error > theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        xy_delta = xy_delta_to_touchingblock

        # print("[push_block]", " xy_delta:", xy_delta, ", theta_error:", theta_error, ", phase: ", self.phase)
        return xy_delta
    
    def _get_orient_block_left(self,
                                xy_dir_block_to_ee,
                                orient_circle_diameter,
                                xy_block,
                                xy_ee,
                                theta_error,
                                theta_threshold_flat_enough):
        xy_dir_block_to_ee = self.rotate(0.15, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee

        # print("xy_dir_block_to_ee:", xy_dir_block_to_ee, ", xy_block_to_ee:", xy_block_to_ee, ", xy_push_left_spot:", xy_push_left_spot, ", xy_delta:", xy_delta)
        if theta_error < theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta

    def _get_orient_block_right(self,
                                xy_dir_block_to_ee,
                                orient_circle_diameter,
                                xy_block,
                                xy_ee,
                                theta_error,
                                theta_threshold_flat_enough):
        xy_dir_block_to_ee = self.rotate(-0.15, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee

        # print("xy_dir_block_to_ee:", xy_dir_block_to_ee, ", xy_block_to_ee:", xy_block_to_ee, ", xy_push_left_spot:", xy_push_left_spot, ", xy_delta:", xy_delta)
        if theta_error > -theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta
        
    def _get_action_for_block_target(self):
        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.01
        info = self._get_action_info()
        # print("xy_ee:", info.xy_ee, ", xy_block:", info.xy_block, ", ee_pos:", self.ee_pos)
        if self.phase == "move_to_pre_block":
            xy_delta, max_step_velocity = self._get_move_to_preblock(
                info.xy_pre_block, info.xy_ee)

        if self.phase == "move_to_block":
            xy_delta = self._get_move_to_block(
                info.xy_delta_to_nexttoblock, info.theta_threshold_to_orient,
                info.theta_error)

        if self.phase == "push_block":
            xy_delta = self._get_push_block(
                info.theta_error, info.theta_threshold_to_orient,
                info.xy_delta_to_touchingblock)

        orient_circle_diameter = 0.12

        if self.phase == "orient_block_left" or self.phase == "orient_block_right":
            max_step_velocity = 0.01

        if self.phase == "orient_block_left":
            xy_delta = self._get_orient_block_left(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough)

        if self.phase == "orient_block_right":
            xy_delta = self._get_orient_block_right(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough)

        length = np.linalg.norm(xy_delta)
        if length > max_step_velocity:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_velocity
        
        if self.target_reached:
            xy_delta = [0.0, 0.0]
            
        return xy_delta

    def action(self, observation):

        self.extract_from_observation(observation)
        out = self._get_action_for_block_target()
        # print("-----> out:", out, ", phase:", self.phase, ", ee dist:", np.linalg.norm(self.ee_pos[:-1]))
        return out