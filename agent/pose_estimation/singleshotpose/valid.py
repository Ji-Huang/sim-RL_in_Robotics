import argparse
from torchvision import transforms

import dataset
from darknet import Darknet
from utils_ps import *
from MeshPly import MeshPly

def plot_predicted_points(img,gt_points, pred_points, width, height, batch_idx, color_rgb='red'):
    # Plot predicted keypoints
    # Convert the tensor image back to numpy array
    img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # Convert from RGB to BGR (OpenCV's expected format)
    img_np = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (width, height))
    # Convert points to integers
    gt_points = gt_points.int()
    pred_points = pred_points.int()
    if color_rgb == 'red':
        color = (0, 0, 255)
    elif color_rgb == 'blue':
        color = (255, 0, 0)
    elif color_rgb == 'purple':
        color = (255, 0, 255)

    # Draw points on the image
    for point in pred_points:
        cv2.circle(img_np, point.numpy(), 4, color, -1)
    for point in gt_points:
        cv2.circle(img_np, point.numpy(), 4, (255, 0, 0), -1)

    # Connect the corners to form the bounding box
    lines = [(1, 2), (2, 4), (4, 3), (3, 1),
             (5, 6), (6, 8), (8, 7), (7, 5),
             (1, 5), (2, 6), (3, 7), (4, 8)]
    for line in lines:
        start_point = pred_points[line[0]]
        end_point = pred_points[line[1]]
        cv2.line(img_np, start_point.numpy(), end_point.numpy(), color, 2)
        start_point = gt_points[line[0]]
        end_point = gt_points[line[1]]
        cv2.line(img_np, start_point.numpy(), end_point.numpy(), (255, 0, 0), 2)

    # Display the image with points using OpenCV
    # cv2.imshow('Image with Points', cv2.resize(img_np, (960, 540)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join("../dataset/box_series/PredImages_fromRandom", f"{batch_idx:06d}.jpeg"), cv2.resize(img_np, (960, 540)))

def generate_saliency_maps(model, data, target, batch_idx):
    # Calculate saliency maps
    data_copy = data.detach()
    data_copy.requires_grad_()
    loss, _, _, _, loss_x, loss_y, loss_conf = model.loss(model(data_copy), target)
    map_loss = loss_x + loss_y  # position loss / conf loss
    model.zero_grad()
    map_loss.backward()
    saliency, _ = data_copy.grad.abs().max(dim=1)
    N = data.shape[0]
    for i in range(N):
        saliency_map = saliency.cpu()[i].numpy()
        img_np = (data[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Convert from RGB to BGR (OpenCV's expected format)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Normalize the saliency map
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        # Convert the saliency map to a color map
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
        # Overlay the saliency map on the original image
        result = cv2.addWeighted(img_np, 0.4, heatmap, 0.6, 0)
        # Display the result
        # cv2.imshow('Saliency Map', cv2.resize(result, (960, 540)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join("../dataset/box_real/SaliencyMap_fromRandom", f"{batch_idx:06d}.jpeg"),
                    cv2.resize(result, (960, 540)))


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
    backupdir    = data_options['backup']
    name         = data_options['name']
    gpus         = data_options['gpus'] 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height'])
    max_gt_num    = int(data_options['max_ground_truth_num'])
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    seed = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    num_classes     = 1
    testing_samples = 0.0
    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []

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

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model_args = {'cfgfile': modelcfg, 'datacfg': datacfg}
    model = Darknet.load_from_checkpoint(args.weightfile, map_location=device, **model_args)
    model.print_network()
    model.to(device)
    model.eval()
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints 
    num_labels    = num_keypoints * 2 + 3 # +2 for width, height,  +1 for class label

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(valid_images,
                                                                  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),]),
                                                                  train=False,
                                                                  max_gt_num=max_gt_num), batch_size=1, shuffle=False, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        # Forward pass
        output = model(data)

        # generate_saliency_maps(model, data, target, batch_idx)

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)
        # Evaluation
        # Iterate through all batch elements
        for img, box_pr, target in zip(data, all_boxes, target):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target.view(-1, num_labels)
            # Get how many objects are present in the scene
            num_gts = truths.shape[0]
            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2*num_keypoints+1):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])

                # Denormalize the corner predictions 
                corners2D_gt = torch.reshape(torch.FloatTensor(box_gt[:num_keypoints * 2]), (num_keypoints, 2))
                corners2D_pr = torch.reshape(torch.FloatTensor(box_pr[:num_keypoints * 2]), (num_keypoints, 2))
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height          
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # ploted_img = plot_predicted_points(img, corners2D_gt, corners2D_pr, im_width, im_height, batch_idx, 'red')

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                errs_corner2D.append(corner_dist)

                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(corners3D, dtype='float32'), np.array(corners2D_gt, dtype='float32'),
                                 np.array(intrinsic_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(corners3D, dtype='float32'), np.array(corners2D_pr, dtype='float32'),
                                 np.array(intrinsic_calibration, dtype='float32'))

                # Compute translation error
                trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                errs_trans.append(trans_dist)
                
                # Compute angle error
                angle_dist   = calcAngularDistance(R_gt, R_pr)
                errs_angle.append(angle_dist)
                
                # Compute pixel error
                proj_2d_gt   = compute_projection(vertices, R_gt, t_gt, intrinsic_calibration)
                proj_2d_pred = compute_projection(vertices, R_pr, t_pr, intrinsic_calibration)
                norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=1)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt = np.concatenate((R_gt, t_gt), axis=1).dot(
                    np.concatenate((vertices, np.ones((8, 1))), axis=1).T)
                transform_3d_pred = np.concatenate((R_pr, t_pr), axis=1).dot(
                    np.concatenate((vertices, np.ones((8, 1))), axis=1).T)
                norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist       = np.mean(norm3d)    
                errs_3d.append(vertex_dist)  

                # Sum errors
                testing_error_trans  += trans_dist
                testing_error_angle  += angle_dist
                testing_error_pixel  += pixel_dist
                testing_samples      += 1
        print(batch_idx)
        print(f'Projection error: {pixel_dist} ')
    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    eps          = 1e-5
    acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc3d10      = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d  = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    # Print test statistics
    logging('Results of {}'.format(name))
    logging('   Acc using {} px 2D Projection after PnP = {:.2f}%'.format(px_threshold, acc))
    logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging("   Mean reprojected 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
    logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )

if __name__ == '__main__':

    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='cfg/box_real.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--weightfile', type=str, default='data/models/randomized_best.ckpt') # imagenet initialized weights
    args       = parser.parse_args()
    datacfg    = args.datacfg
    modelcfg   = args.modelcfg
    weightfile = args.weightfile
    valid(datacfg, modelcfg, weightfile)
