import numpy as np
import os
import math

from scipy.spatial.transform import Rotation as R

class BoxObject():
    def __init__(self, width, length, height, center=[0, 0, 0], rotation=[0, 0, 0]):
        self.width = width
        self.length = length
        self.height = height

        self.center = center        #x, y, z . x -> vertical, y -> horizontal
        self.rotation = rotation    #rx, ry, rz . in radians

        self.face = 0
        self.offset = 0.02
        self.calculate_center_points()
        self.calculate_contact_points()

    @staticmethod
    def _quaternion_to_euler_angle(q):
        r = R.from_quat(q)

        return r.as_euler('zyx', degrees=True)
    
    @staticmethod
    def _rotate_point(point, origin, angle_radians):
        px, py = point
        originx, originy = origin

        # Translate point to origin
        translated_point = (px - originx, py - originy)
        # Perform rotation
        cos_angle, sin_angle = np.cos(angle_radians), np.sin(angle_radians)
        rotated_point_x = translated_point[0] * cos_angle - translated_point[1] * sin_angle
        rotated_point_y = translated_point[0] * sin_angle + translated_point[1] * cos_angle
        # Translate point back
        final_point = (rotated_point_x + originx, rotated_point_y + originy)
        return np.array(final_point)

    def update_pose_from_observation_unity(self, observation):
        """
            return object pose from observasion

            :return: Cartesian pose of the box position in the task space. return rx,ry,rz,x,y,z in order
        """
        X, Y, Z =  observation[28:31]
        RX, RY, RZ = BoxObject._quaternion_to_euler_angle(observation[31:35])

        self.center = [Z, -1*X, Y]
        self.rotation = np.deg2rad([-1*RZ, RX, -1*RY]).tolist()
    
    def update_pose_from_observation(self, observation):
        """
            return object pose from observasion

            :return: Cartesian pose of the box position in the task space. return rx,ry,rz,x,y,z in order
        """
        self.rotation = observation[0:3]
        self.center = observation[3:6]

    def update_position_from_localization(self, x, y, rz):
        self.center = np.concatenate([[x, y], [self.center[-1]]])
        # self.rotation = np.concatenate([self.rotation[:-1], [rz]])
    
    def calculate_center_points(self):
        center_point = self.center[:-1]

        center_points = []

        #initial contact points
        center_points.append([center_point[0], center_point[1]])
        center_points.append([center_point[0] - ((self.width/2.0)-self.offset), center_point[1]])
        center_points.append([center_point[0] + ((self.width/2.0)-self.offset), center_point[1]])

        rotation_radians = self.rotation[-1]
        for i, center_point_ in enumerate(center_points):
            center_points[i] = BoxObject._rotate_point(center_point_, center_point, rotation_radians)
        
        self.center_points = center_points
    
    def calculate_contact_points(self):
        center_point = self.center[:-1]

        contact_points = []

        #initial contact points
        contact_points.append([center_point[0], center_point[1] - ((self.height/2.0)+self.offset)])
        contact_points.append([center_point[0] - ((self.width/2.0)-self.offset), center_point[1] - ((self.height/2.0)+self.offset)])
        contact_points.append([center_point[0] + ((self.width/2.0)-self.offset), center_point[1] - ((self.height/2.0)+self.offset)])

        contact_points.append([center_point[0], center_point[1] + ((self.height/2.0)+self.offset)])
        contact_points.append([center_point[0] - ((self.width/2.0)-self.offset), center_point[1] + ((self.height/2.0)+self.offset)])
        contact_points.append([center_point[0] + ((self.width/2.0)-self.offset), center_point[1] + ((self.height/2.0)+self.offset)])

        rotation_radians = self.rotation[-1]
        for i, contact_point in enumerate(contact_points):
            contact_points[i] = BoxObject._rotate_point(contact_point, center_point, rotation_radians)

        self.contact_points = contact_points

    def calculate_target_points(self, distance):
        center_point = self.center[:-1]
        target_points = []

        #initial contact points
        target_points.append([center_point[0], center_point[1] - ((self.height/2.0)+distance)])

        target_points.append([center_point[0], center_point[1] + ((self.height/2.0)+distance)])

        rotation_radians = self.rotation[-1]
        for i, target_point in enumerate(target_points):
            target_points[i] = BoxObject._rotate_point(target_point, center_point, rotation_radians)

        return target_points
    
    def update_box(self, observation, unity=False):
        if unity:
            self.update_pose_from_observation_unity(observation)
        else:
            # print('observation:', observation)
            self.update_pose_from_observation(observation)
            
        self.calculate_center_points()
        self.calculate_contact_points()

    def set_pose(self, rotation, center):
        self.rotation = rotation
        self.center = center

        self.calculate_center_points()
        self.calculate_contact_points()
    
    def get_pose(self):
        return np.append(self.rotation, self.center)
    
    def get_position(self):
        return self.center
    
    def get_rotation(self):
        return self.rotation
    
    def get_dimension(self):
        return self.width, self.length, self.height
    
    def get_backside_center(self, target):
        
        # print('center:', self.center)
        # print(self.contact_points)
        if np.linalg.norm((target-self.contact_points[0])) > np.linalg.norm((target-self.contact_points[3])):
            self.face = 0
            return self.contact_points[0]
        else:
            self.face = 1
            return self.contact_points[3]