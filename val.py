import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from dataloader_nyu import getTestData
from models.fractional_prediction import Eff_estimation


def up_sample(pred):
    return F.interpolate(pred, size=[480, 640], mode='bilinear', align_corners=True)


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
    parser.add_argument('--bs', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg  ECCV16',
                        action='store_true')
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    args = parser.parse_args()
    torch.manual_seed(2022)
    device = torch.device('cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = Eff_estimation().to(device)
    state_dict = torch.load('PDNet_nyu.pt')
    model.load_state_dict(state_dict)
    print(evaluate(model, device, args))


def evaluate(model, device, args):
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

    test_loader = getTestData(batch_size=1)
    depth_scores = np.zeros((6, len(test_loader)))  # six metrics
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            model.eval()
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].to(device))
            depth = torch.autograd.Variable(sample_batched['depth']).cpu().numpy()
            # print(depth.shape)

            pred, _, _, _, _ = model(image)
            pred = pred.cpu().numpy()
            flip_trans = transforms.RandomHorizontalFlip(p=1)
            image_flip = flip_trans(image)
            pred_flip, _, _, _, _ = model(image_flip)
            pred_flip = pred_flip.cpu().numpy()

            valid_mask = np.logical_and(depth > 1e-3, depth < 10)
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
            final = torch.nn.functional.interpolate(torch.Tensor(final), size=depth.shape[-2:], mode='bilinear',
                                                    align_corners=True).numpy()
            # print(depth.shape, valid_mask.shape)
            errors = compute_errors(depth[valid_mask], final[valid_mask])
            print(i)
            print(errors[4])
            for k in range(len(errors)):
                depth_scores[k][i] = errors[k]

        e = depth_scores.mean(axis=1)
        return e


if __name__ == '__main__':
    main()
