import os
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import sys
import argparse
import re
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import itertools

from dataloader.LeftAtrium import LAHeart
from dataloader.pancreas import Pancreas
from dataloader.AortaDissection import AortaDissection

import utils.loss
from utils import statistic, ramps
from utils.loss import DiceLoss, SoftIoULoss, to_one_hot
from utils.losses import FocalLoss
from utils.logger import get_cur_time, checkpoint_save
from utils.Generate_Prototype import *
from utils.train_util import *
from utils.lr import *

from networks.vnet_AMC import VNet_AMC
from test import test_LA_Pancreas, test_AD


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Semi-supervised Training for UPCoL: Uncertainty-informed Prototype Consistency Learning for Semi-supervised Medical Image Segmentation')

    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--exp', type=str, default='LA', help='experiment_name')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default='../results', help='Paths to previous checkpoints')
    parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
    # dataset
    parser.add_argument("--data_dir", type=str, default='../../../Datasets/LA_dataset',
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str, default='../datalist/LA',
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")
    parser.add_argument("--aug_times", type=int, default=5,
                        help="times of augmentation for training.")

    # Optimization options
    parser.add_argument('--lab_batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='maximum epoch number to pretraining')
    parser.add_argument('--save_step', type=int, default=5, help='frequecy of checkpoint save in pretraining')
    parser.add_argument('--consistency_rampup', type=float,
                        default=300.0, help='consistency_rampup')

    parser.add_argument('--beta1', type=float, default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float, default=1, help='multiplier of prototype')

    # Miscs
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')

    return parser.parse_args()


args = get_arguments()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# create logger
resultdir = os.path.join(args.save_path, args.exp)
logdir = os.path.join(resultdir, 'logs')
savedir = os.path.join(resultdir, 'checkpoints')
shotdir = os.path.join(resultdir, 'snapshot')
print('Result path: {}\nLogs path: {}\nCheckpoints path: {}\nSnapshot path: {}'.format(resultdir, logdir, savedir,
                                                                                       shotdir))

os.makedirs(logdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)
os.makedirs(shotdir, exist_ok=True)

writer = SummaryWriter(logdir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(shotdir + '/' + 'snapshot.log', encoding='utf8')
fh.setFormatter(formatter)
logger.addHandler(fh)
logging.info(str(args))


def create_model(ema=False):
    net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(
        img_z * mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    return mask.long(), loss_mask.long()


def create_dataloader():
    if 'LA' in args.data_dir:
        train_labset = LAHeart(args.data_dir, args.list_dir, split='lab', aug_times=args.aug_times)
        train_unlabset = LAHeart(args.data_dir, args.list_dir, split='unlab', aug_times=args.aug_times)
        testset = LAHeart(args.data_dir, args.list_dir, split='test')
    elif 'TBAD' in args.data_dir:
        train_labset = AortaDissection(args.data_dir, args.list_dir, split='lab', aug_times=args.aug_times)
        train_unlabset = AortaDissection(args.data_dir, args.list_dir, split='unlab', aug_times=args.aug_times)
        testset = AortaDissection(args.data_dir, args.list_dir, split='test', aug_times=args.aug_times)
    else:
        train_labset = Pancreas(args.data_dir, args.list_dir, split='lab', aug_times=args.aug_times)
        train_unlabset = Pancreas(args.data_dir, args.list_dir, split='unlab', aug_times=args.aug_times)
        testset = Pancreas(args.data_dir, args.list_dir, split='test', aug_times=args.aug_times)

    trainlab_loader = DataLoader(train_labset, batch_size=args.lab_batch_size, shuffle=True, num_workers=0)
    trainunlab_loader = DataLoader(train_unlabset, batch_size=args.unlab_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    logging.info("{} iterations for lab per epoch.".format(len(trainlab_loader)))
    logging.info("{} iterations for unlab per epoch.".format(len(trainunlab_loader)))
    logging.info("{} samples for test.\n".format(len(test_loader)))
    return trainlab_loader, trainunlab_loader, test_loader


def calculate_similarity_matrix(prototypes_a, prototypes_b):
    """
    Calculate the similarity matrix for the given prototypes.
    
    Args:
        prototypes (list of torch.Tensor): List of prototypes for each class.
        
    Returns:
        torch.Tensor: Similarity matrix.
    """
    num_classes = len(prototypes_a)
    prototype_matrix_a = torch.stack([proto.view(-1) for proto in prototypes_a])  # Ensure shape: [num_classes, features]
    prototype_matrix_b = torch.stack([proto.view(-1) for proto in prototypes_b]) # Ensure shape: [num_classes, features]
    similarity_matrix = torch.mm(prototype_matrix_a, prototype_matrix_b.T)  # Shape: [num_classes, num_classes]
    normalized_similarity_matrix = F.softmax(similarity_matrix, dim=1)
    return normalized_similarity_matrix

def contrastive_loss(similarity_matrix, margin=1.0):
    """
    Calculate the contrastive loss based on the similarity matrix.
    
    Args:
        similarity_matrix (torch.Tensor): Similarity matrix of shape [num_classes, num_classes].
        margin (float): Margin for the contrastive loss.
        
    Returns:
        torch.Tensor: Contrastive loss.
    """
    num_classes = similarity_matrix.size(0)
    identity_matrix = torch.eye(num_classes, device=similarity_matrix.device)
    
    # Positive pairs (diagonal elements of similarity matrix)
    pos_pairs = torch.diag(similarity_matrix)
    
    # Negative pairs (off-diagonal elements of similarity matrix)
    neg_pairs = similarity_matrix[~torch.eye(num_classes, dtype=bool)].view(num_classes, -1)
    
    # Contrastive loss
    pos_loss = torch.mean((1 - pos_pairs) ** 2)
    neg_loss = torch.mean(torch.clamp(neg_pairs - margin, min=0) ** 2)
    loss = pos_loss + neg_loss
    
    return loss

def soft_triple_loss(prototypes, features, labels, margin=0.5):
    """
    prototypes: [2, 128]
    features: [2, 2, 112, 112, 80]
    labels: [2, 112, 112, 80]
    """

    distances = torch.stack([calDist(features, prototype, scaler=args.scaler) 
                             for prototype in prototypes], dim=1)  # [2, 2, 112, 112, 80, 2]
    
    # 将 distances 的维度调整为 [B, N, num_prototypes]
    distances = distances.view(distances.shape[0], -1, len(prototypes))  # [2, 2 * 112 * 112 * 80, 2]

    # 将 labels 的维度调整为与 distances 的前两维相匹配
    labels = labels.view(labels.shape[0], -1)  # [2, 2 * 112 * 112 * 80]

    # 确保 labels 的值在 [0, num_prototypes) 范围内
    labels = torch.clamp(labels, 0, len(prototypes) - 1)

    positive_distances = distances[torch.arange(labels.shape[0])[:, None], 
                                   torch.arange(labels.shape[1]), 
                                   labels]

    # 计算负样本距离
    negative_indices = (labels + 1) % len(prototypes)  # 获取与 labels 值不同的另一个原型的索引
    negative_distances = distances[torch.arange(labels.shape[0])[:, None], 
                                   torch.arange(labels.shape[1]), 
                                   negative_indices]

    loss = torch.mean(F.relu(positive_distances - negative_distances + margin))
    return loss

def sup_loss(predictions, labels, dice_loss, ce_loss, focal_loss, iou_loss, reliability_map=False):
    if torch.is_tensor(reliability_map) == False:
        loss_ce = ce_loss(predictions[0], labels)
        loss_dice = dice_loss(predictions[1], labels)
        loss_focal = focal_loss(predictions[2], labels)
        loss_iou = ce_loss(predictions[3], labels)
        mean_prediction = (predictions[0] + predictions[1] + predictions[2] + predictions[3]) / 4
        loss_ce1 = ce_loss(mean_prediction, labels)
        loss = (loss_ce + loss_dice + loss_focal + loss_iou) / 4 + loss_ce1
    else:
        loss_ce = (ce_loss(predictions[0], labels) * reliability_map).sum()
        loss_dice = dice_loss(predictions[1], labels, reliability_map)
        loss_focal = (focal_loss(predictions[2], labels) * reliability_map).sum() 
        loss_iou = (ce_loss(predictions[3], labels) * reliability_map).sum() 
        mean_prediction = (predictions[0] + predictions[1] + predictions[2] + predictions[3]) / 4
        loss_ce1 = (ce_loss(mean_prediction, labels) * reliability_map).sum()
        loss = (loss_ce + loss_dice + loss_focal + loss_iou) / 4 + loss_ce1

    return loss


def fixmatch(x1, y1, x2, y2, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    mixed_y = mixed_y.long()
    return mixed_x, mixed_y

def mixup_data(x_labeled, y_labeled, x_unlabeled, y_unlabeled, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_labeled.size(0)
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x_labeled + (1 - lam) * x_unlabeled[index, :]
    y_a, y_b = y_labeled, y_unlabeled[index]
    return mixed_x, y_a, y_b, lam

def main():
    save_path = Path(savedir)

    set_random_seed(args.seed)
    net_au = create_model().cuda()
    net = create_model().cuda()
    ema_net = create_model(ema=True).cuda()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer = optim.Adam(itertools.chain(net.parameters(), net_au.parameters()), lr=args.lr,
                           betas=(args.beta1, args.beta2))
    if args.resume:
        load_net_opt(net, optimizer, Path(args.load_path) / 'best.pth')
        load_net_opt(ema_net, optimizer, Path(args.load_path) / 'best_ema.pth')

    trainlab_loader, trainunlab_loader, test_loader = create_dataloader()

    dice_loss = DiceLoss(nclass=args.num_classes)
    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss()
    iou_loss = SoftIoULoss(nclass=args.num_classes)
    pixel_level_ce_loss = CrossEntropyLoss(reduction='none')
    # consistency_loss_fn = ConsistencyLoss()
    maxdice = 0
    iter_num = 0
    cur_threshold = 0.5
    for epoch in tqdm(range(args.num_epochs), ncols=70):
        logging.info('\n')

        net.train()
        net_au.train()
        # adjust_learning_rate_warmup_cosine(optimizer, epoch, args.lr, warmup_epochs=5, total_epochs=args.num_epochs)
        # adjust_learning_rate_step(optimizer, epoch, args.lr, step_size=10, gamma=0.5)
        consistency_weight = get_current_consistency_weight(epoch, args.consistency_rampup)
        for step, (labeled_batch, unlabeled_batch) in enumerate(zip(trainlab_loader, trainunlab_loader)):
            lab_img, lab_lab = labeled_batch['image'].cuda(), labeled_batch['label'].cuda()
            unlab_img = unlabeled_batch['image'].cuda()
            lab_lab_onehot = to_one_hot(lab_lab.unsqueeze(1), args.num_classes)
            
            '''Supervised'''
            lab_out = net(lab_img)

            lab_out_ce = lab_out[0]
            lab_out_dice = lab_out[1]
            lab_out_focal = lab_out[2]
            lab_out_iou = lab_out[3]

            loss_supervised = sup_loss(lab_out, lab_lab, dice_loss, ce_loss, focal_loss, iou_loss) 
            # labeled prototypes
            lab_fts = F.interpolate(lab_out[4], size=lab_lab.shape[-3:], mode='trilinear')
            lab_prototypes = getPrototype(lab_fts, lab_lab_onehot)
            # prototypes_sim_lm = calculate_similarity_matrix(lab_prototypes, mix_prototypes)
        
            # lab_prototypes = [(lab_prototypes[c] + mix_prototypes[c]) / 2 for c in range(args.num_classes)]

            '''Unsupervised'''
            with torch.no_grad():
                unlab_ema_out = ema_net(unlab_img)
                unlab_ema_out_pred = (unlab_ema_out[0] + unlab_ema_out[1] + unlab_ema_out[2] + unlab_ema_out[3]) / 4

                unlab_ema_out_soft = torch.softmax(unlab_ema_out_pred, dim=1)
                unlab_ema_out_soft1 = torch.softmax(unlab_ema_out_pred / 0.5, dim=1)
                # uncertainty assesment
                unlab_ema_out_var = sum((x - unlab_ema_out_pred)**2 for x in unlab_ema_out[:4]) / 4
                entro_uncertainty = -torch.sum(unlab_ema_out_soft * torch.log(unlab_ema_out_soft + 1e-16), dim=1)
                entro_norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in entro_uncertainty], dim=0)
                # reliability_map = (1 - entro_norm_uncertainty) / np.prod(np.array(entro_norm_uncertainty.shape[-3:]))
                norm_uncertainty = torch.stack([var / torch.sum(var) for var in unlab_ema_out_var], dim=0)
                # Uncertainty assessment using variance
                mean_uncertainty = torch.mean(unlab_ema_out_soft, dim=1)  # [batch_size, depth, height, width]
                mean_var_uncertainty = torch.mean(unlab_ema_out_var, dim=1)  # [batch_size, depth, height, width]
                # Combine uncertainty from mean and variance
                combined_uncertainty = (1 - mean_uncertainty) * torch.exp(-mean_var_uncertainty) * (1 - entro_norm_uncertainty)
                reliability_map = combined_uncertainty / np.prod(np.array(combined_uncertainty.shape[-3:]))

                # mixed_img, mixed_lab = mixup_data(labeled_img, labeled_lab, unlabeled_img, pseudo_labels, alpha)

                certain_mask = torch.zeros([reliability_map.shape[0], reliability_map.shape[1], reliability_map.shape[2], reliability_map.shape[3]]).cuda()
                certain_mask = reliability_map[reliability_map > 0.9] = 1.0

                unlab_ema_out_soft = unlab_ema_out_soft * certain_mask
                unlab_ema_out_soft1 = unlab_ema_out_soft1 * certain_mask
                unlab_ema_mask = torch.argmax(unlab_ema_out_soft, dim=1)
                unlab_ema_mask1 = torch.argmax(unlab_ema_out_soft1, dim=1)  
                unlab_ema_mask_onehot = to_one_hot(unlab_ema_mask.unsqueeze(1), args.num_classes)

                loss_pl = sup_loss(unlab_ema_out, unlab_ema_mask, dice_loss, pixel_level_ce_loss, focal_loss, iou_loss, reliability_map)

                unlab_ema_fts = F.interpolate(unlab_ema_out[4], size=unlab_ema_mask.shape[-3:], mode='trilinear')
                unlab_prototypes = getPrototype(unlab_ema_fts, unlab_ema_mask_onehot, region=reliability_map)
                # unlab_prototypes = [(unlab_prototypes[c] + consistency_weight * mix_prototypes[c]) / (1 + consistency_weight) for c in range(args.num_classes)]
                # unlab_prototypes = [(unlab_prototypes[c] + mix_prototypes[c]) / 2 for c in range(args.num_classes)]

            mix_img, mix_lab_a, mix_lab_b, lam = mixup_data(lab_img, lab_lab, unlab_img, unlab_ema_mask1)
            mix_out = net_au(mix_img)
            mix_fts = F.interpolate(mix_out[4], size=lab_lab.shape[-3:], mode='trilinear')
            mix_lab_onehot = to_one_hot(mix_lab_a.unsqueeze(1), args.num_classes)
            mix_prototypes = getPrototype(mix_fts, mix_lab_onehot)  

            loss_mix_supervised = lam * sup_loss(mix_out, mix_lab_a, dice_loss, ce_loss, focal_loss, iou_loss) + (1 - lam) * sup_loss(mix_out, mix_lab_b, dice_loss, ce_loss, focal_loss, iou_loss) 
            loss_supervised = (loss_mix_supervised + loss_supervised) / 2
            
            lab_prototypes = [(lab_prototypes[c] + mix_prototypes[c]) / 2 for c in range(args.num_classes)]
            unlab_prototypes = [(unlab_prototypes[c] + mix_prototypes[c]) / 2 for c in range(args.num_classes)]
            
            '''Prototype fusion'''
            prototypes_sim_lu = calculate_similarity_matrix(lab_prototypes, unlab_prototypes)
            prototypes = [(lab_prototypes[c] + consistency_weight * unlab_prototypes[c]) / (1 + consistency_weight) for
                          c in range(args.num_classes)]

            lab_dist = torch.stack([calDist(lab_fts, prototype, scaler=args.scaler) for prototype in prototypes], dim=1)
            unlab_dist = torch.stack(
                [calDist(unlab_ema_fts, prototype, scaler=args.scaler) for prototype in prototypes], dim=1)

            '''triple loss'''
            triple_loss = soft_triple_loss(prototypes, lab_fts, lab_lab)
            triple_loss += consistency_weight * torch.sum(soft_triple_loss(prototypes, unlab_ema_fts, unlab_ema_mask) * reliability_map)

            '''Prototype consistency learning'''
            loss_pc_lab = ce_loss(lab_dist, lab_lab)
            loss_pc_unlab = torch.sum(pixel_level_ce_loss(unlab_dist, unlab_ema_mask) * reliability_map)
            loss_pc = loss_pc_lab + consistency_weight * loss_pc_unlab
            '''Total'''
            loss = loss_supervised + loss_pc + triple_loss + loss_pl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(net, ema_net, args.alpha, iter_num)
            iter_num += 1

            # lr_ = args.lr * (1.0 - iter_num / (args.num_epochs * 40)) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            lab_masks = torch.softmax(lab_out_dice, dim=1)
            lab_masks = torch.argmax(lab_masks, dim=1)
            train_dice = statistic.dice_ratio(lab_masks, lab_lab)

            logging.info('epoch : %d, step : %d, loss_all: %.4f,'
                         #'loss_ce: %.4f, loss_dice: %.4f, loss_focal: %.4f, loss_iou: %.4f, '
                         'loss_supervised: %.4f, loss_pc_lab: %.4f, '
                         'loss_pc_unlab: %.4f, train_dice: %.4f' % (
                             epoch + 1, step, loss.item(),
                             # loss_ce.item(), loss_dice.item(), loss_focal.item(), loss_iou.item(),
                             loss_supervised.item(), loss_pc_lab.item(), loss_pc_unlab.item(), train_dice))

            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            # writer.add_scalar('info/loss_iou', loss_iou, iter_num)
            writer.add_scalar('info/loss_supervised', loss_supervised, iter_num)

            writer.add_scalar('info/loss_pc_lab', loss_pc_lab, iter_num)
            writer.add_scalar('info/loss_pc_unlab', loss_pc_unlab, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('info/loss_all', loss, iter_num)
            writer.add_scalar('val/train_dice', train_dice, iter_num)

        '''Test'''
        if (epoch + 1) % args.save_step == 0:
            if 'TBAD' in args.data_dir:
                val_dice, maxdice, max_flag = test_AD(net, test_loader, args, maxdice)
            else:
                val_dice, maxdice, max_flag = test_LA_Pancreas(net, test_loader, args, maxdice)

            writer.add_scalar('val/test_dice', val_dice, epoch + 1)

            save_mode_path = os.path.join(save_path,
                                          'epoch_{}_iter_{}_dice_{}.pth'.format(
                                              epoch + 1, iter_num, round(val_dice, 4)))
            save_net_opt(net, optimizer, save_mode_path, epoch + 1)

            # save_ema_path = os.path.join(save_path,
            #                 'ema_epoch_{}_iter_{}_dice_{}.pth'.format(
            #                     epoch+1, iter_num, round(val_dice, 4)))
            # save_net_opt(ema_net, optimizer, save_ema_path, epoch+1)

            # save_au_path = os.path.join(save_path,
            #                              'au_epoch_{}_iter_{}_dice_{}.pth'.format(
            #                                  epoch + 1, iter_num, round(val_dice, 4)))
            # save_net_opt(net_au, optimizer, save_au_path, epoch + 1)

            if max_flag:
                save_net_opt(net, optimizer, save_path / 'best.pth', epoch + 1)
                save_net_opt(ema_net, optimizer, save_path / 'best_ema.pth', epoch + 1)
                save_net_opt(net_au, optimizer, save_path / 'best_au.pth', epoch + 1)

        writer.flush()


if __name__ == '__main__':
    if os.path.exists(resultdir + '/codes'):
        shutil.rmtree(resultdir + '/codes')
    shutil.copytree('.', resultdir + '/codes',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    main()
