from __future__ import print_function
import argparse
from torchvision import transforms
import pytorch_lightning as pl

import dataset
from utils_ps import *
from cfg import parse_cfg
from darknet import Darknet

import warnings
warnings.filterwarnings("ignore")

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

if __name__ == "__main__":

    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='cfg/box_randomized.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default=None) # imagenet initialized weights
    args                = parser.parse_args()
    datacfg             = args.datacfg
    modelcfg            = args.modelcfg

    # Parse configuration files
    data_options  = read_data_cfg(datacfg)
    trainlist     = data_options['train']
    testlist      = data_options['valid']
    gpus          = data_options['gpus']
    meshname      = data_options['mesh']
    backupdir     = data_options['backup']
    max_gt_num    = int(data_options['max_ground_truth_num'])
    vx_threshold  = float(data_options['diam']) * 0.1  # threshold for the ADD metric

    if not os.path.exists(backupdir):
        makedirs(backupdir)

    net_options   = parse_cfg(modelcfg)[0]
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    num_workers   = int(net_options['num_workers'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    nsamples      = file_lines(trainlist)
    nbatches      = nsamples / batch_size

    # Train parameters
    max_epochs    = int(net_options['max_epochs'])
    num_keypoints = int(net_options['num_keypoints'])
    
    # Test parameters
    im_width    = int(data_options['width'])
    im_height   = int(data_options['height'])
    fx          = float(data_options['fx'])
    fy          = float(data_options['fy'])
    u0          = float(data_options['u0'])
    v0          = float(data_options['v0'])
    test_width  = int(net_options['test_width'])
    test_height = int(net_options['test_height'])

    # Specify which gpus to use
    use_cuda      = True
    seed          = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
        device = 'cuda:0'

    # Specifiy the model
    if args.initweightfile:
        model_args = {'cfgfile': modelcfg, 'datacfg': datacfg}
        model = Darknet.load_from_checkpoint(args.initweightfile, map_location=device, **model_args)
    else:
        model = Darknet(modelcfg, datacfg, learning_rate, decay)

    # Model settings
    model.print_network()
    init_width        = model.width
    init_height       = model.height

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(dataset.listDataset(trainlist,
                                                                   shape=(init_width, init_height),
                                                                   shuffle=True,
                                                                   train=True,
                                                                   transform=transforms.Compose([transforms.ToTensor(),]),
                                                                   max_gt_num=max_gt_num),
                                               batch_size=batch_size, shuffle=False, **kwargs)

    # Get the dataloader for test data
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(testlist,
                                                                  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),]),
                                                                  train=False,
                                                                  max_gt_num=max_gt_num),
                                              batch_size=1, shuffle=False, **kwargs)

    # Pass the model to GPU
    if use_cuda:
        model = model.cuda()  # model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # Multiple GPU parallelism

    # Train the model
    model_ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='data/models/', save_last=True, save_on_train_epoch_end=False,
                                                 monitor='val/pred_corner_err_2d', save_top_k=1, mode='min')
    trainer = pl.Trainer(resume_from_checkpoint=args.initweightfile,
                         gpus=[int(gpus)], max_epochs=max_epochs,
                         callbacks=[pl.callbacks.LearningRateMonitor(), model_ckpt_cb,],
                         check_val_every_n_epoch=10,
                         log_every_n_steps=20)
    trainer.fit(model, train_loader, test_loader)

