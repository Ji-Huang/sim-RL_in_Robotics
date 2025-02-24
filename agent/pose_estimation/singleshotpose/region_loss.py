import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_ps import *


def build_targets(pred_corners, target, num_keypoints, num_anchors, num_classes, nH, nW,
                  sil_thresh, max_gt_num):
    device = pred_corners.device
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    conf_mask = torch.ones(nB, nA, nH, nW).to(device)
    coord_mask = torch.zeros(nB, nA, nH, nW).to(device)
    cls_mask = torch.zeros(nB, nA, nH, nW).to(device)
    # Target tensors
    txs = torch.zeros(num_keypoints, nB, nA, nH, nW).to(device)
    tys = torch.zeros(num_keypoints, nB, nA, nH, nW).to(device)
    tconf = torch.zeros(nB, nA, nH, nW).to(device)
    tcls = torch.zeros(nB, nA, nH, nW).to(device)

    num_labels = 2 * num_keypoints + 3  # +2 for width, height and +1 for class within label files
    nAnchors = nA * nH * nW
    nPixels = nH * nW

    nGT = 0
    nCorrect = 0

    for b in range(nB):
        # For each target in ground truth
        for t in range(max_gt_num):
            if target[b][t * num_labels + 1] == 0:
                break
            # Get gt box for the current label
            nGT = nGT + 1
            gx = list()
            gy = list()
            gt_box = list()
            for i in range(num_keypoints):
                gt_box.extend([target[b][t * num_labels + 2 * i + 1], target[b][t * num_labels + 2 * i + 2]])
                gx.append(target[b][t * num_labels + 2 * i + 1] * nW)
                gy.append(target[b][t * num_labels + 2 * i + 2] * nH)
                if i == 0:
                    gt_center_x = int(gx[i])
                    gt_center_y = int(gy[i])
            # Update masks
            best_n = 0  # Using 1 anchor box for single object
            pred_box = pred_corners[b * nAnchors + best_n * nPixels + gt_center_y * nH + gt_center_x]
            conf = corner_confidence(gt_box, pred_box)
            # Mask out cells with no object inside
            coord_mask[b][best_n][gt_center_y][gt_center_x] = 1
            cls_mask[b][best_n][gt_center_y][gt_center_x] = 1
            conf_mask[b][best_n][gt_center_y][gt_center_x] = 1
            # Update targets
            for i in range(num_keypoints):
                txs[i][b][best_n][gt_center_y][gt_center_x] = gx[i] - gt_center_x
                tys[i][b][best_n][gt_center_y][gt_center_x] = gy[i] - gt_center_y
            tconf[b][best_n][gt_center_y][gt_center_x] = 1
            tcls[b][best_n][gt_center_y][gt_center_x] = target[b][t * num_labels]
            # Update recall during training
            if conf > 0.7:
                nCorrect = nCorrect + 1

    nGT = nGT / nB
    nCorrect = nCorrect / nB
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls


class RegionLoss(nn.Module):
    def __init__(self, num_keypoints=9, num_classes=1, anchors=[], num_anchors=1, max_gt_num=5):
        # Define the loss layer
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors  # for single object pose estimation, there is only 1 trivial predictor (anchor)
        self.num_keypoints = num_keypoints
        self.coord_scale = 1
        self.class_scale = 1
        self.thresh = 0.6
        self.max_gt_num = max_gt_num

    def forward(self, output, target):
        # Parameters
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        num_keypoints = self.num_keypoints

        # Activation
        output = output.view(nB, nA, (num_keypoints * 2 + 1 + nC), nH, nW)
        device = output.device
        # Key points' coordinates
        x = torch.sigmoid(output.index_select(2, torch.LongTensor([0]).to(device)).view(nB, nA, nH, nW)).unsqueeze(0)
        y = torch.sigmoid(output.index_select(2, torch.LongTensor([1]).to(device)).view(nB, nA, nH, nW)).unsqueeze(0)
        for i in range(1, num_keypoints):
            x = torch.cat((x, output.index_select(2, torch.LongTensor([2 * i + 0]).to(device)).view(nB, nA, nH, nW).unsqueeze(0)),dim=0)
            y = torch.cat((y, output.index_select(2, torch.LongTensor([2 * i + 1]).to(device)).view(nB, nA, nH, nW).unsqueeze(0)),dim=0)
        # Confidence value
        conf = torch.sigmoid(output.index_select(2, torch.LongTensor([2 * num_keypoints]).to(device))).view(nB, nA, nH, nW)
        # Predicted class
        cls = output.index_select(2, torch.linspace(2 * num_keypoints + 1, 2 * num_keypoints + 1 + nC - 1, nC).long().to(device))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        # Create predicted boxes
        pred_corners = torch.FloatTensor(2 * num_keypoints, nB * nA * nH * nW).to(device)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).to(device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).to(device)
        # Get normalized coordinate for predicted points
        for i in range(num_keypoints):
            pred_corners[2 * i + 0] = (x[i].view_as(grid_x) + grid_x) / nW
            pred_corners[2 * i + 1] = (y[i].view_as(grid_y) + grid_y) / nH
        pred_corners = pred_corners.transpose(0, 1).contiguous().view(-1, 2 * num_keypoints)

        # Build targets
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, txs, tys, tconf, tcls = \
            build_targets(pred_corners, target, num_keypoints, nA, nC, nH, nW,
                          self.thresh, self.max_gt_num)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.5).sum().item()) / nB

        conf_mask = conf_mask.sqrt()
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
        cls = cls[cls_mask].view(-1, nC)

        # Create loss
        loss_xs = list()
        loss_ys = list()
        for i in range(num_keypoints):
            loss_xs.append(
                self.coord_scale * nn.MSELoss(size_average=False)(x[i] * coord_mask, txs[i] * coord_mask))
            loss_ys.append(
                self.coord_scale * nn.MSELoss(size_average=False)(y[i] * coord_mask, tys[i] * coord_mask))
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask)
        loss_x = sum(loss_xs)
        loss_y = sum(loss_ys)

        loss = loss_x + loss_y + loss_conf  # in single object pose estimation, there is no classification loss

        # print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_conf.item(), loss.item()))

        return loss, nGT, nCorrect, nProposals, loss_x, loss_y, loss_conf