import os
import time
import torch
import argparse
import scipy.io
import warnings
from torchvision import datasets, transforms
import cv2
from torchvision.transforms.functional import to_pil_image, to_tensor
import numpy as np

from .darknet import Darknet
from .utils_ps import *
from .MeshPly import MeshPly


class Image2Num():

    def __init__(self, datacfg, modelcfg, weightfile, camera_position=[0.0, 1.6, 2.6], camera_rotation=[150.0, 0.0, 180]) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # Parse configuration files
        data_options = read_data_cfg(datacfg)
        meshname = data_options['mesh']
        gpus = data_options['gpus']
        fx = float(data_options['fx'])
        fy = float(data_options['fy'])
        u0 = float(data_options['u0'])
        v0 = float(data_options['v0'])
        self.im_width = int(data_options['width'])
        self.im_height = int(data_options['height'])

        # Parameters
        seed = int(time.time())
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
        self.num_classes = 1

        # Read object model information, get 3D bounding box corners
        mesh = MeshPly(meshname)
        vertices = np.array(mesh.vertices, dtype='float32')
        self.corners3D = get_3D_corners(vertices)

        # Read intrinsic camera parameters
        self.intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
        model_args = {'cfgfile': modelcfg, 'datacfg': datacfg}
        self.model = Darknet.load_from_checkpoint(weightfile, map_location=self.device, **model_args)
        # self.model.print_network()
        self.model.to(self.device)
        self.model.eval()

        self.num_kp = self.model.num_keypoints

        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((416, 416))

        self.camera_position = camera_position
        self.camera_rotation = camera_rotation

    def get_box_pos(self, data):
        # Convert the image to PyTorch tensor
        tensor = self.transform(data)
        tensor = self.resize(tensor)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.to(self.device)
        
        # Forward pass
        output = self.model(tensor)

        # Using confidence threshold, eliminate low-confidence predictions
        box_pr = get_region_boxes(output, self.num_classes, self.num_kp)
        box_pr = box_pr[0]
        box_pr18 = box_pr[:self.num_kp * 2]

        # Denormalize the corner predictions
        corners2D_pr = torch.reshape(torch.FloatTensor(box_pr18), (self.num_kp, 2))
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.im_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.im_height

        # Compute box pos wrt. camera by pnp
        R_box2cam, t_box2cam = pnp(np.array(self.corners3D, dtype='float32'), np.array(corners2D_pr, dtype='float32'),
                        np.array(self.intrinsic_calibration, dtype='float32'))

        T_box2cam = np.hstack((R_box2cam, t_box2cam))  # Combine rotation and translation into one matrix
        T_box2cam = np.vstack((T_box2cam, [0, 0, 0, 1]))  # Homogeneous coordinates

        # Inverse of the extrinsic matrix
        R_world2cam, t_world2cam = self.get_camera_extrinsic()

        R_cam2world = R_world2cam.T
        t_cam2world = -np.dot(R_cam2world, t_world2cam)

        T_cam2world = np.eye(4)
        T_cam2world[:3, :3] = R_cam2world
        T_cam2world[:3, 3] = t_cam2world

        # Compute box pos wrt. world frame
        T_box2world = np.dot(T_cam2world, T_box2cam)
        R_box2world = T_box2world[:3, :3]
        t_box2world = T_box2world[:3, 3]

        x_box2world, y_box2world, z_box2world = t_box2world
        rz_box2world = np.arctan2(R_box2world[1, 0], R_box2world[0, 0])

        return x_box2world, y_box2world, rz_box2world

    def get_camera_extrinsic(self):
        # Given rotation angles in degrees
        rx, ry, rz = np.deg2rad(0), np.deg2rad(225), np.deg2rad(90)  # sim camera 1080p
        # Calculate rotation matrices
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
        R = np.dot(Ry, Rz)

        # R = np.array([[-0.029, 0.802, -0.596],
        #                 [1.0, 0.027, -0.012],
        #                 [0.006, -0.596, -0.803]])  # real camera

        t = np.array([1.6, 0, 1.0])  # sim camera 1080p
        # t = np.array([1.083, -0.008, 0.715])  # real camera
        R = R.T
        t = -np.dot(R, t)
        return R, t

    # def get_camera_extrinsic(self):
    #     # Given rotation angles in degrees
    #     rx, ry, rz = (np.deg2rad(r) for r in self.camera_rotation)
        
    #     # Calculate rotation matrices
    #     Rz = np.array([
    #         [np.cos(rz), -np.sin(rz), 0],
    #         [np.sin(rz), np.cos(rz), 0],
    #         [0, 0, 1]
    #     ])
    #     Ry = np.array([
    #         [np.cos(ry), 0, np.sin(ry)],
    #         [0, 1, 0],
    #         [-np.sin(ry), 0, np.cos(ry)]
    #     ])
    #     Rx = np.array([[1, 0, 0],
    #                 [0, np.cos(rx), -np.sin(rx)],
    #                 [0, np.sin(rx), np.cos(rx)]
    #                 ])
    #     R = np.dot(Ry, Rz)

    #     t = np.array(self.camera_position)  # sim camera

    #     R = R.T
    #     t = -np.dot(R, t)
    #     return R, t