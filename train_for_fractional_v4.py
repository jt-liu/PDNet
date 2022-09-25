import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
from torchvision import transforms
from pytorch_msssim import MS_SSIM

import math
from loss import SILogLoss
from models.fractional_prediction import Eff_estimation

# from dataloader import getTrainingTestingData, getTestData
from datasets_KITTI import MyDataset
from utils import AverageMeter, DepthNorm, colorize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        # print("Conv2d Layer Inited")
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        # print("Linear Layer Inited")


def cal_section(min, max, idx):
    length_section = (max - min) / 10
    right = min + length_section * (idx + 1)
    left = right - length_section
    return left, right


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bs', '--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', default=True, help='if set, crops according to Garg  ECCV16',
                        action='store_true')
    # parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    # parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    # parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    #########FOR KITTI########
    parser.add_argument("--dataset", default='kitti', type=str, help="Dataset to train on")
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--data_path', type=str, help='dataset path',
                        default='/home/ywq/project/LapDepth-release-master/datasets/KITTI')
    parser.add_argument('--trainfile_kitti', type=str, help='train dataset list',
                        default='filenames/eigen_train_files_with_gt_dense.txt')
    parser.add_argument('--testfile_kitti', type=str, help='test dataset list',
                        default="filenames/eigen_test_files_with_gt_dense.txt")
    parser.add_argument('--height', type=int, default=352)
    parser.add_argument('--width', type=int, default=704)

    args = parser.parse_args()
    device = torch.device('cuda')
    cudnn.enabled = True
    cudnn.benchmark = True

    model = Eff_estimation().to(device)
    print('Model created.')
    model.decoder.apply(weights_init)
    print("#####weights inited#####")
    state_dict = torch.load('ckpts/Jun20/epoch20_best_rmse_params.pt')
    model.load_state_dict(state_dict)
    start_epoch = -1

    # Logging
    prefix = 'VisionTransformer_' + str(args.bs)
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Training parameters
    batch_size = args.bs

    # loss
    criterion_ueff = SILogLoss()

    # load data
    # train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)
    train_set = MyDataset(args, train=True)
    test_set = MyDataset(args, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        num_workers=16, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=16, pin_memory=True)
    best_rms = 999

    # optim scheduler
    optimizer = optim.AdamW(params=model.parameters(), weight_decay=0.01, lr=args.lr)
    # optimizer = optim.Adam(params=model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader), cycle_momentum=True,
                                                    base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                                    div_factor=30, final_div_factor=2)

    # start training...
    for epoch in range(start_epoch + 1, args.epochs):
        # Switch to train mode
        # scheduler.step()

        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            if sample_batched[1].ndim != 4 and sample_batched[1][0] == False:
                continue
            model.train()
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched[0].cuda())  # 'image'
            depth = torch.autograd.Variable(sample_batched[1].cuda(non_blocking=True))  # 'depth'
            # Predict
            output, x_block1, x_block2, x_block3, x_block4 = model(image)
            # print(output[:, :, 100:200, 200:300])
            output = nn.functional.interpolate(output, depth.shape[-2:], mode='bilinear', align_corners=True)
            x_block1 = nn.functional.interpolate(x_block1, depth.shape[-2:], mode='bilinear', align_corners=True)
            x_block2 = nn.functional.interpolate(x_block2, depth.shape[-2:], mode='bilinear', align_corners=True)
            x_block3 = nn.functional.interpolate(x_block3, depth.shape[-2:], mode='bilinear', align_corners=True)
            x_block4 = nn.functional.interpolate(x_block4, depth.shape[-2:], mode='bilinear', align_corners=True)
            # Compute the loss
            mask = depth > args.min_depth
            l_dense = criterion_ueff(output, depth, mask=mask.to(torch.bool), interpolate=False)
            output[depth <= args.min_depth] = 0.001
            depth[depth <= args.min_depth] = 0.001
            ms_ssim_module = MS_SSIM(data_range=80, size_average=True, channel=1)
            ssim_loss = 1 - ms_ssim_module(output, depth)
            ssim_loss = torch.sqrt(ssim_loss)

            # criterion_l2 = torch.nn.MSELoss()
            criterion_l2 = torch.nn.SmoothL1Loss()
            gt_min_depth = depth[mask].min()
            gt_max_depth = depth[mask].max()
            # print(gt_min_depth, gt_max_depth)
            distributed_depth = torch.histc(depth[mask], 10, 0, 0, out=None)
            # print(distributed_depth)
            _, idx_distributed_depth = distributed_depth.sort(descending=True)
            # print(idx_distributed_depth)

            #################xb1 xb1 xb1 xb1########################
            left_5_1, right_5_1 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[4])
            left_5_2, right_5_2 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[5])
            left_5_3, right_5_3 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[6])
            left_5_4, right_5_4 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[7])
            left_5_5, right_5_5 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[8])
            left_5_6, right_5_6 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[9])
            mask_5 = ((depth > left_5_1) & (depth < right_5_1)) | ((depth > left_5_2) & (depth < right_5_2)) | (
                    (depth > left_5_3) & (depth < right_5_3)) | ((depth > left_5_4) & (depth < right_5_4)) | (
                             (depth > left_5_5) & (depth < right_5_5)) | ((depth > left_5_6) & (depth < right_5_6))
            pixel_5 = criterion_l2(x_block1[mask_5], depth[mask_5])
            #################xb2 xb2 xb2 xb2########################
            left_6_1, right_6_1 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[3])
            left_6_2, right_6_2 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[4])
            left_6_3, right_6_3 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[5])
            left_6_4, right_6_4 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[6])
            left_6_5, right_6_5 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[7])
            mask_6 = ((depth > left_6_1) & (depth < right_6_1)) | ((depth > left_6_2) & (depth < right_6_2)) | (
                    (depth > left_6_3) & (depth < right_6_3)) | ((depth > left_6_4) & (depth < right_6_4)) | (
                             (depth > left_6_5) & (depth < right_6_5))
            pixel_6 = criterion_l2(x_block2[mask_6], depth[mask_6])
            #################xb3 xb3 xb3 xb3########################
            left_8_1, right_8_1 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[1])
            left_8_2, right_8_2 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[2])
            left_8_3, right_8_3 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[3])
            mask_8 = ((depth > left_8_1) & (depth < right_8_1)) | ((depth > left_8_2) & (depth < right_8_2)) | (
                    (depth > left_8_3) & (depth < right_8_3))
            pixel_8 = criterion_l2(x_block3[mask_8], depth[mask_8])
            #################xb4 xb4 xb4 xb4########################
            left_12_1, right_12_1 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[0])
            left_12_2, right_12_2 = cal_section(gt_min_depth, gt_max_depth, idx_distributed_depth[1])
            mask_12 = ((depth > left_12_1) & (depth < right_12_1)) | ((depth > left_12_2) & (depth < right_12_2))
            pixel_12 = criterion_l2(x_block4[mask_12], depth[mask_12])

            if math.isnan(pixel_5): pixel_5.data.item = torch.Tensor(0)
            if math.isnan(pixel_6): pixel_6.data.item = torch.Tensor(0)
            if math.isnan(pixel_8): pixel_8.data.item = torch.Tensor(0)
            if math.isnan(pixel_12): pixel_12.data.item = torch.Tensor(0)

            loss = 8 * l_dense + 2 * ssim_loss + (0.1 * (0.2 * pixel_5 + 0.5 * pixel_6 + 0.6 * pixel_8 + pixel_12)) ** 2
            # Update step
            # print(loss)
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            niter = epoch * N + i
            if i % 10 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
                # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'])
                # Log to tensorboard
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], niter)
                writer.add_scalar('Train/Loss', losses.val, niter)
                writer.add_scalar('Train/SI', 8 * l_dense.data.item(), niter)
                writer.add_scalar('Train/SSIM', 2 * ssim_loss.data.item(), niter)
                writer.add_scalar('Train/pixel_5', 0.1 * 0.2 * pixel_5.data.item(), niter)
                writer.add_scalar('Train/pixel_6', 0.1 * 0.5 * pixel_6.data.item(), niter)
                writer.add_scalar('Train/pixel_8', 0.1 * 0.6 * pixel_8.data.item(), niter)
                writer.add_scalar('Train/pixel_12', 0.1 * pixel_12.data.item(), niter)
            # if i == 20:
            #     break
            if i < 2000:
                if i % 200 == 0:
                    LogProgress(model, writer, test_loader, niter)
            else:
                if i % 1000 == 0:
                    LogProgress(model, writer, test_loader, niter)
                '''if i == 6336:
                    err = evaluate(model, writer, epoch, args)
                    f = open('f.txt', 'a')
                    print(err, file=f)
                    f.close()
                    torch.save(model.state_dict(), './epoch{0}_params.pt'.format(epoch+0.5))'''

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,}
                 # 'scheduler': scheduler.state_dict()}
        torch.save(state, './effiwithseman.pth')
        err = evaluate(model, writer, epoch, args)
        rmse = err[4]
        if rmse < best_rms:
            best_rms = rmse
            best_rmse_state_dict = model.state_dict()
            torch.save(best_rmse_state_dict, './epoch{0}_best_rmse_params.pt'.format(epoch))
        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), './final_params.pt')


def evaluate(model, writer, epoch, args):
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    test_set = MyDataset(args, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=16, pin_memory=True)
    # depth_scores = np.zeros((6, len(test_loader)))  # six metrics
    depth_scores = np.zeros((6, 652))  # six metrics
    with torch.no_grad():
        j = -1
        for i, sample_batched in enumerate(test_loader):
            if sample_batched[1].ndim != 4 and sample_batched[1][0] == False:
                continue
            model.eval()
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched[0].cuda())
            depth = torch.autograd.Variable(sample_batched[1]).cpu().numpy()
            # print(depth.shape)

            pred, _, _, _, _ = model(image)
            pred = pred.cpu().numpy()
            flip_trans = transforms.RandomHorizontalFlip(p=1)
            image_flip = flip_trans(image)
            pred_flip, _, _, _, _ = model(image_flip)
            pred_flip = pred_flip.cpu().numpy()

            valid_mask = np.logical_and(depth > 1e-3, depth < 80)
            if args.garg_crop or args.eigen_crop:
                _, _, gt_height, gt_width = depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[:, :, int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[:, :, int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[:, :, 45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

            final = (0.5 * pred) + (0.5 * flip_trans(torch.Tensor(pred_flip)).numpy())
            final = torch.nn.functional.interpolate(torch.Tensor(final), size=[352, 1216], mode='bilinear',
                                                    align_corners=True).numpy()
            # print(depth.shape, valid_mask.shape)
            errors = compute_errors(depth[valid_mask], final[valid_mask])
            print(i)
            print(errors[4])
            j += 1
            for k in range(len(errors)):
                depth_scores[k][j] = errors[k]

        e = depth_scores.mean(axis=1)
        writer.add_scalar('epoch/a1', e[0], epoch)
        writer.add_scalar('epoch/a2', e[1], epoch)
        writer.add_scalar('epoch/a3', e[2], epoch)
        writer.add_scalar('epoch/rel', e[3], epoch)
        writer.add_scalar('epoch/rms', e[4], epoch)
        writer.add_scalar('epoch/log_10', e[5], epoch)
        return e


def LogProgress(model, writer, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        for i, sample_batched in enumerate(test_loader):
            image = torch.autograd.Variable(sample_batched[0].cuda())
            if sample_batched[1].ndim != 4 and sample_batched[1][0] == False:
                continue
            depth = torch.autograd.Variable(sample_batched[1].cuda(non_blocking=True))
            # depth_n = DepthNorm(depth)
            # print("\n\n\n",depth_n.min(), depth_n.max())
            if epoch == 0:
                writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
                # writer.add_image('Train.2.Image_mask', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
                writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
            output, x_block1, x_block2, x_block3, x_block4 = model(image)
            print("\n\n\n", output.min(), output.max())
            # print(output, "\n\n\n",depth_n[:, :, 405:471, 401:601])
            del image
            del depth
            del output
            del x_block1, x_block2, x_block3, x_block4
            break


if __name__ == '__main__':
    main()
