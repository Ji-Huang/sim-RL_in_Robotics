import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .singleshotpose.region_loss import RegionLoss
from cfg import *
import pytorch_lightning as pl
from .utils_ps import *
from .MeshPly import MeshPly

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(pl.LightningModule):
    def __init__(self, cfgfile, datacfg, learning_rate=1e-3, decay=0.5):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        data_options = read_data_cfg(datacfg)
        self.max_gt_num = int(data_options['max_ground_truth_num'])

        self.width         = int(self.blocks[0]['width'])
        self.height        = int(self.blocks[0]['height'])
        self.test_width    = int(self.blocks[0]['test_width'])
        self.test_height   = int(self.blocks[0]['test_height'])
        self.im_width = int(data_options['width'])
        self.im_height = int(data_options['height'])
        self.num_keypoints = int(self.blocks[0]['num_keypoints'])
        self.lr = learning_rate
        self.decay = decay

        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.iter = 0

        # parameters for test
        # Get the intrinsic camerea matrix, mesh, vertices and corners of the model
        meshname = data_options['mesh']
        fx = float(data_options['fx'])
        fy = float(data_options['fy'])
        u0 = float(data_options['u0'])
        v0 = float(data_options['v0'])
        vx_threshold = float(data_options['diam']) * 0.1  # threshold for the ADD metric

        mesh = MeshPly(meshname)
        self.vertices = np.array(mesh.vertices, dtype='float32')
        self.corners3D = get_3D_corners(self.vertices)
        self.intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)
        self.vx_threshold = vx_threshold
        self.px_threshold = 5  # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
        self.testing_error_trans = 0.0
        self.testing_error_angle = 0.0
        self.testing_error_pixel = 0.0
        self.testing_samples = 0.0
        self.errs_2d = []
        self.errs_3d = []
        self.errs_trans = []
        self.errs_angle = []
        self.errs_corner2D = []

        # self.save_hyperparameters()
    def forward(self, x):
        ind = -2
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters, eps=1e-4))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                if anchors == ['']:
                    loss.anchors = []
                else:
                    loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                loss.max_gt_num = self.max_gt_num
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=250, gamma=self.decay)
        return [opt], [sched]

    def training_step(self, batch, _):
        data, target = batch

        # Forward pass
        output = self.forward(data)
        self.seen = self.seen + data.data.size(0)

        # Compute loss, grow an array of losses for saving later on
        loss, nGT, nCorrect, nProposals, loss_x, loss_y, loss_conf = self.loss(output, target)
        self.log(f'train/nGT', nGT)
        self.log(f'train/nCorrect', nCorrect)
        self.log(f'train/nProposals', nProposals)
        self.log(f'train/loss_x', loss_x.item())
        self.log(f'train/loss_y', loss_y.item())
        self.log(f'train/loss_conf', loss_conf.item())
        self.log(f'train/loss', loss.item())

        return loss

    def validation_step(self, batch, _):
        data, target = batch

        # Forward pass
        output = self.forward(data)
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, self.num_classes, self.num_keypoints)
        # Iterate through all batch elements
        for box_pr, target in zip(all_boxes, target):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target.view(-1, self.num_keypoints * 2 + 3)
            # Get how many objects are present in the scene
            num_gts = truths.shape[0]
            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2 * self.num_keypoints + 1):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])

                # Denormalize the corner predictions
                corners2D_gt = torch.reshape(torch.FloatTensor(box_gt[:self.num_keypoints * 2]), (self.num_keypoints, 2))
                corners2D_pr = torch.reshape(torch.FloatTensor(box_pr[:self.num_keypoints * 2]), (self.num_keypoints, 2))
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * self.im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * self.im_height
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.im_height

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                self.errs_corner2D.append(corner_dist)

                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(self.corners3D, dtype='float32'), np.array(corners2D_gt, dtype='float32'),
                                 np.array(self.intrinsic_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(self.corners3D, dtype='float32'), np.array(corners2D_pr, dtype='float32'),
                                 np.array(self.intrinsic_calibration, dtype='float32'))

                # Compute errors
                # Compute translation error
                trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                self.errs_trans.append(trans_dist)

                # Compute angle error
                angle_dist = calcAngularDistance(R_gt, R_pr)
                self.errs_angle.append(angle_dist)

                # Compute pixel error
                proj_2d_gt = compute_projection(self.vertices, R_gt, t_gt, self.intrinsic_calibration)
                proj_2d_pred = compute_projection(self.vertices, R_pr, t_pr, self.intrinsic_calibration)
                norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=1)
                pixel_dist = np.mean(norm)
                self.errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt = np.concatenate((R_gt, t_gt), axis=1).dot(
                    np.concatenate((self.vertices, np.ones((8, 1))), axis=1).T)
                transform_3d_pred = np.concatenate((R_pr, t_pr), axis=1).dot(
                    np.concatenate((self.vertices, np.ones((8, 1))), axis=1).T)
                norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist = np.mean(norm3d)
                self.errs_3d.append(vertex_dist)

                # Sum errors
                self.testing_error_trans += trans_dist
                self.testing_error_angle += angle_dist
                self.testing_error_pixel += pixel_dist
                self.testing_samples += 1

        return

    def validation_epoch_end(self, out):
        # Compute 2D projection, 6D pose and 5cm5degree scores
        eps = 1e-5

        accPnP = len(np.where(np.array(self.errs_2d) <= self.px_threshold)[0]) * 100. / (len(self.errs_2d) + eps)
        acc3d = len(np.where(np.array(self.errs_3d) <= self.vx_threshold)[0]) * 100. / (len(self.errs_3d) + eps)
        acc5cm5deg = len(np.where((np.array(self.errs_trans) <= 0.05) & (np.array(self.errs_angle) <= 5))[0]) * 100. / (
                len(self.errs_trans) + eps)
        corner_acc = len(np.where(np.array(self.errs_corner2D) <= self.px_threshold)[0]) * 100. / (len(self.errs_corner2D) + eps)
        mean_err_2d = np.mean(self.errs_2d)
        mean_corner_err_2d = np.mean(self.errs_corner2D)
        nts = float(self.testing_samples)

        # Print test statistics
        self.log('val/pred_corner_err_2d', mean_corner_err_2d)
        logging("   Mean corner error is %f" % (mean_corner_err_2d))
        logging('   Acc using {} px 2D Projection after PnP = {:.2f}%'.format(self.px_threshold, accPnP))
        logging('   Acc using {} vx 3D Transformation = {:.2f}%'.format(self.vx_threshold, acc3d))
        logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        logging('   Translation error: %f, angle error: %f' % (
            self.testing_error_trans / (nts + eps), self.testing_error_angle / (nts + eps)))

        # reset parameter
        self.testing_error_trans = 0.0
        self.testing_error_angle = 0.0
        self.testing_error_pixel = 0.0
        self.testing_samples = 0.0
        self.errs_2d = []
        self.errs_3d = []
        self.errs_trans = []
        self.errs_angle = []
        self.errs_corner2D = []
