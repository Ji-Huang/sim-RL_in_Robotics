import argparse
from torchvision import transforms

import dataset
from darknet import Darknet
from utils_ps import *
from MeshPly import MeshPly

def get_camera_extrinsic():
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

    R = np.array([[-0.029, 0.802, -0.596],
                      [1.0, 0.027, -0.012],
                      [0.006, -0.596, -0.803]])  # real camera

    t = np.array([1.6, 0, 1.0])  # sim camera 1080p
    t = np.array([1.083, -0.008, 0.715])  # real camera
    R = R.T
    t = -np.dot(R, t)
    return R, t

def valid(datacfg, modelcfg, weightfile):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # Parse configuration files
    data_options = read_data_cfg(datacfg)
    train_images = data_options['train']
    valid_images = data_options['valid']
    meshname     = data_options['mesh']
    name         = data_options['name']
    gpus         = data_options['gpus']
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height'])
    max_gt_num    = int(data_options['max_ground_truth_num'])

    # Parameters
    seed = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    num_classes     = 1

    # Read object model information, get 3D bounding box corners
    mesh      = MeshPly(meshname)
    vertices  = np.array(mesh.vertices, dtype='float32')
    corners3D = get_3D_corners(vertices)
    try:
        diam = float(data_options['diam'])
    except:
        diam = calc_pts_diameter(np.array(mesh.vertices))
    # Read intrinsic camera parameters
    intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)
    # Set extrinsic camera parameters
    world_to_camera_R, world_to_camera_t = get_camera_extrinsic()
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model_args = {'cfgfile': modelcfg, 'datacfg': datacfg}
    model = Darknet.load_from_checkpoint(args.weightfile, map_location=device, **model_args)
    model.print_network()
    model.to(device)
    model.eval()
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints

    testing_error_trans = []
    testing_error_angle = []
    testing_error_pixel = []
    errs_3d             = []
    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset.PoseDataset(valid_images,
                                                                  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),]),
                                                                  max_gt_num=max_gt_num), batch_size=1, shuffle=False, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        # Forward pass
        output = model(data)
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)
        # Evaluation
        # Iterate through all batch elements
        for img, box_pr, target in zip(data, all_boxes, target):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            pose = target.view(-1, 6)
            # Get how many objects are present in the scene
            num_gts = pose.shape[0]
            # Iterate through each ground-truth object
            for k in range(num_gts):
                pose_k = pose[k].cpu().numpy()
                # Denormalize the corner predictions
                corners2D_pr = torch.reshape(torch.FloatTensor(box_pr[:num_keypoints * 2]), (num_keypoints, 2))
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                R_pr, t_pr = pnp(np.array(corners3D, dtype='float32'), np.array(corners2D_pr, dtype='float32'),
                                 np.array(intrinsic_calibration, dtype='float32'))
                R_gt = np.array([[np.cos(pose_k[5]), -np.sin(pose_k[5]), 0],
                                 [np.sin(pose_k[5]), np.cos(pose_k[5]), 0],
                                 [0, 0, 1]])
                R_gt = world_to_camera_R @ R_gt
                t_gt = np.array([pose_k[0], pose_k[1], pose_k[2]])
                t_gt = np.expand_dims(world_to_camera_R @ t_gt + world_to_camera_t, 1)

                # Compute translation error
                trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                testing_error_trans.append(trans_dist)

                # Compute angle error
                angle_dist = calcAngularDistance(R_gt, R_pr)
                testing_error_angle.append(angle_dist)

                # Compute 3D distances
                transform_3d_gt = np.concatenate((R_gt, t_gt), axis=1).dot(
                    np.concatenate((vertices, np.ones((8, 1))), axis=1).T)
                transform_3d_pred = np.concatenate((R_pr, t_pr), axis=1).dot(
                    np.concatenate((vertices, np.ones((8, 1))), axis=1).T)
                norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist = np.mean(norm3d)
                errs_3d.append(vertex_dist)
        print(batch_idx)
        print(f'Translation error: {trans_dist} m, angle error: {angle_dist} degree')

    eps          = 1e-5
    acc3d10 = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d) + eps)
    acc5cm5deg = (len(np.where((np.array(testing_error_trans) <= 0.05) & (np.array(testing_error_angle) <= 5))[0]) * 100.
                  / (len(testing_error_trans) + eps))
    acc5cm10deg = (len(np.where((np.array(testing_error_trans) <= 0.05) & (np.array(testing_error_angle) <= 10))[0]) * 100.
                  / (len(testing_error_trans) + eps))
    logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging('   Acc using 5 cm 10 degree metric = {:.2f}%'.format(acc5cm10deg))
    logging("   Mean vertex error is %f" % (np.mean(errs_3d)))
    logging('   Translation error: %f m, angle error: %f degree' % (np.mean(testing_error_trans), np.mean(testing_error_angle)))


if __name__ == '__main__':

    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='cfg/box_real.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--weightfile', type=str, default='data/models/real_best.ckpt') # imagenet initialized weights
    args       = parser.parse_args()
    datacfg    = args.datacfg
    modelcfg   = args.modelcfg
    weightfile = args.weightfile
    valid(datacfg, modelcfg, weightfile)
