import torch.utils.data as data
from PIL import Image
import numpy as np
from PIL import ImageFile
from transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy, \
    Normalize
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _is_pil_image(img):
    return isinstance(img, Image.Image)


class MyDataset(data.Dataset):
    def __init__(self, args, train=True, return_filename=False):
        if train is True:
            self.datafile = args.trainfile_kitti
            self.angle_range = (-1, 1)
            self.depth_scale = 256.0
        else:
            self.datafile = args.testfile_kitti
            self.depth_scale = 256.0
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.return_filename = return_filename
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)

    def __getitem__(self, index):
        divided_file = self.fileset[index].split()
        date = divided_file[0].split('/')[0] + '/'

        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.args.data_path + '/' + divided_file[0]
        rgb = Image.open(rgb_file)
        gt = False
        gt_dense = False
        if self.train is False:
            divided_file_ = divided_file[0].split('/')
            filename = divided_file_[1] + '_' + divided_file_[4]

            # Considering missing gt in Eigen split
            if divided_file[1] != 'None':
                gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                gt = Image.open(gt_file)
            else:
                pass
        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]

            gt = Image.open(gt_file)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)

        # cropping in size that can be divided by 16
        h = rgb.height
        w = rgb.width
        bound_left = (w - 1216) // 2
        bound_right = bound_left + 1216
        bound_top = h - 352
        bound_bottom = bound_top + 352

        rgb = rgb.crop((bound_left, bound_top, bound_right, bound_bottom))

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32)) / self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)

        rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)

        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        else:
            return rgb, gt, gt_dense

    def __len__(self):
        return len(self.fileset)


class Transformer(object):
    def __init__(self, args):
        self.train_transform = EnhancedCompose([
            RandomCropNumpy((args.height, args.width)),
            RandomHorizontalFlip(),
            [RandomColor(multiplier_range=(0.8, 1.2)), None, None],
            ArrayToTensorNumpy(),
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
        self.test_transform = EnhancedCompose([
            ArrayToTensorNumpy(),
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
