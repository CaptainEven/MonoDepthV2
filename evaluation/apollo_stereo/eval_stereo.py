"""
    Brief: Evaluate disparity
    Author: wangpeng54@baidu.com
    Date: 2019/6/8
"""

import argparse
import glob
import os

import cv2
import numpy as np
import utils.utils as uts


class StereoEval(object):
    """
    Evaluation of pose
    """

    def __init__(self, args):
        """
        Initializer.
        :param args:
        :param is_resize:
        :param net_size:
        """
        self.args = args
        self._names = ['D1_bg', 'D1_fg', 'D1_all']
        self._offset = []
        self.bg_mask_path = args.gt_dir + '/bg_mask/'
        self.fg_mask_path = args.gt_dir + '/fg_mask/'

        self.disparity_path = args.gt_dir + '/disparity/'
        self.res_disparity_path = args.test_dir + '/disparity/'

        self.depth_path = args.gt_dir + '/depth/'
        self.res_depth_path = args.test_dir + '/depth/'

        self.is_resize = args.is_resize
        self.net_size = args.net_size  # width, height

        ## camera intrinsics
        self.baseline = args.baseline
        self.focal = args.focal

    def _check_disp(self):
        """
        check whether results folder and gt folder has the same dir tree
        :return:
        """
        res_list = glob.glob(self.res_disparity_path + '*.png')
        self.res_disp_names = [os.path.basename(line) for line in res_list]
        res_list.sort()
        self.res_disp_names.sort()

        gt_list = glob.glob(self.disparity_path + '*.png')
        self.gt_disp_names = [os.path.basename(line) for line in gt_list]
        gt_list.sort()
        self.gt_disp_names.sort()

        for image_name in self.gt_disp_names:
            assert image_name in self.res_disp_names, \
                'image %s is not in presented' % image_name

        return res_list, gt_list

    def _check_depth(self):
        """
        check whether results folder and gt folder has the same dir tree
        :return:
        """
        res_list = glob.glob(self.res_depth_path + '*.npy')
        self.res_depth_names = [os.path.basename(line) for line in res_list]
        res_list.sort()
        self.res_depth_names.sort()

        gt_list = glob.glob(self.depth_path + '*.npy')
        self.gt_depth_names = [os.path.basename(line) for line in gt_list]
        gt_list.sort()
        self.gt_depth_names.sort()

        for image_name in self.gt_depth_names:
            assert image_name in self.res_depth_names, \
                'image %s is not in presented' % image_name

        return res_list, gt_list

    def reset(self):
        """
        reset the metric
        """
        self._err_all = []
        self._err_fg = []
        self._err_bg = []

    def load_disparity(self, file_path):
        """
        :param file_path:
        :return:
        """
        disparity = np.float32(cv2.imread(file_path, cv2.IMREAD_UNCHANGED))
        return disparity

    def load_depth(self, file_path):
        """
        :param file_path:
        :return:
        """
        if os.path.isfile(file_path):
            return np.load(file_path)
        else:
            return None

    def load_mask(self, file_path):
        """
        :param file_path:
        :return:
        """
        if not os.path.exists(file_path):
            raise ValueError('%s not exist' % file_path)

        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        return mask

    def eval_disparity(self):
        """
        evaluate the results folder
        :return:
        """
        res_files, gt_files = self._check_disp()

        self.reset()
        for i, (res_file, gt_file) in enumerate(zip(res_files, gt_files)):
            if not os.path.split(res_file)[-1] == os.path.split(gt_file)[-1]:
                print('[Warning]: res file and gt file do not match!')
                continue

            f_name = os.path.basename(gt_file)[:-4]

            gt_disparity = self.load_disparity(gt_file)
            res_disparity = self.load_disparity(res_file)

            ## compute disparity metrics
            self.update_disp_metrics(gt_disparity, res_disparity, f_name)
            print('File {:s} evaluated | [{:d}/{:d}]'.format(f_name, i + 1, len(gt_files)))

        names, values = self.get()

        f = open(self.args.res_file, 'w')
        print('%5s %5s %5s' % tuple(names))
        f.write('%5s %5s %5s' % tuple(names))
        print('%5.4f %5.4f %5.4f' % tuple(values))
        f.write('%5.4f %5.4f %5.4f' % tuple(values))
        f.close()

    def eval_depth(self):
        """
        :return:
        """
        res_files, gt_files = self._check_depth()

        self.reset()
        for i, (res_file, gt_file) in enumerate(zip(res_files, gt_files)):
            if not os.path.split(res_file)[-1] == os.path.split(gt_file)[-1]:
                print('[Warning]: res file and gt file do not match!')
                continue

            f_name = os.path.basename(gt_file)[:-4]

            gt_depth = self.load_depth(gt_file)
            # gt_depth = gt_depth * 1.02
            res_depth = self.load_depth(res_file)

            ## compute disparity metrics
            self.update_depth_metrics(gt_depth, res_depth, f_name)
            print('File {:s} evaluated | [{:d}/{:d}]'.format(f_name, i + 1, len(gt_files)))

        names, values = self.get()

        f = open(self.args.res_file, 'w')
        print('%5s %5s %5s' % tuple(names))
        f.write('%5s %5s %5s' % tuple(names))
        print('%5.4f %5.4f %5.4f' % tuple(values))
        f.write('%5.4f %5.4f %5.4f' % tuple(values))
        f.close()

    def update_disp_metrics(self, gt_disp, pred_disp, img_name):
        """
        Update metrics.
        :param gt_disp:
        :param pred_disp:
        :param img_name:
        :return:
        """
        fg_mask = self.load_mask('%s/%s.png' % (self.fg_mask_path, img_name))
        bg_mask = self.load_mask('%s/%s.png' % (self.bg_mask_path, img_name))
        if self.is_resize and self.net_size is not None:
            fg_mask = cv2.resize(fg_mask, self.net_size)
            bg_mask = cv2.resize(bg_mask, self.net_size)

        valid = fg_mask + bg_mask > 0

        abs_err = np.abs(gt_disp - pred_disp) * valid
        err_count = np.logical_and(abs_err > 3.0, (abs_err / (gt_disp + 1e-6) > 0.05))
        err_all = np.sum(err_count) / np.float32(np.sum(valid))
        err_fg = np.sum(err_count * fg_mask) / np.float32(np.sum(fg_mask))
        err_bg = np.sum(err_count * bg_mask) / np.float32(np.sum(bg_mask))

        self._err_all.append(err_all)
        self._err_fg.append(err_fg)
        self._err_bg.append(err_bg)

    def get_delta_disp(self, gt_depth, pred_depth, valid_mask):
        """
        :param gt_depth:
        :param pred_depth:
        :param valid_mask:
        :return:
        """
        delta_depth = (gt_depth - pred_depth) * valid_mask
        positive_mask = delta_depth > 0.0
        mask = valid_mask * positive_mask
        delta_disp = (delta_depth * mask) / (np.square(gt_depth * mask) / (self.baseline * self.focal) + 1e-6)
        return delta_disp

    def update_depth_metrics(self, gt_depth, pred_depth, img_name):
        """
        :param gt_depth:
        :param pred_depth:
        :param img_name:
        :return:
        """
        fg_mask = self.load_mask('%s/%s.png' % (self.fg_mask_path, img_name))
        bg_mask = self.load_mask('%s/%s.png' % (self.bg_mask_path, img_name))
        if self.is_resize and self.net_size is not None:
            fg_mask = cv2.resize(fg_mask, self.net_size)
            bg_mask = cv2.resize(bg_mask, self.net_size)

        valid_mask = fg_mask + bg_mask > 0

        abs_err = np.abs(gt_depth - pred_depth) * valid_mask

        delta_disp = self.get_delta_disp(gt_depth, pred_depth, valid_mask)
        positive_mask = delta_disp > 0.0
        mean_delta_disp = np.mean(delta_disp * positive_mask)
        self.mean_delta_disp = mean_delta_disp

        ## @even: err analysis
        self.depth_err_analysis(gt_depth, pred_depth, valid_mask, n_bins=20)

        err_count = np.logical_and(abs_err > 1.0, (abs_err / (gt_depth + 1e-6) > 0.05))
        err_all = np.sum(err_count) / np.float32(np.sum(valid_mask))
        err_fg = np.sum(err_count * fg_mask) / np.float32(np.sum(fg_mask))
        err_bg = np.sum(err_count * bg_mask) / np.float32(np.sum(bg_mask))

        self._err_all.append(err_all)
        self._err_fg.append(err_fg)
        self._err_bg.append(err_bg)

    def depth_err_analysis(self, gt_depth, pred_depth, valid_mask, n_bins=20):
        """
        :param gt_depth:
        :param pred_depth:
        :param valid_mask:
        :param n_bins:
        :return:
        """
        assert gt_depth.shape == pred_depth.shape == valid_mask.shape

        # ## error compensation
        # delta_depth = np.square(pred_depth) / (self.baseline * self.focal) * self.mean_delta_disp
        # positive_mask = pred_depth > 0.0
        # delta_depth = delta_depth * positive_mask
        # pred_depth = pred_depth + delta_depth

        abs_err = np.abs(gt_depth - pred_depth) * valid_mask
        abs_err_rate = (abs_err / (gt_depth + 1e-6)) * valid_mask

        ## bin statistcs
        max_gt_depth = np.max(gt_depth)
        bins = np.zeros(n_bins, dtype=np.float32)
        bin_size = max_gt_depth / n_bins
        for i in range(n_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size

            bin_mask = (gt_depth >= bin_start) & (gt_depth < bin_end)
            # print(bin_mask)
            bin_valid_mask = bin_mask * valid_mask
            # print(bin_valid_mask)

            ## bin percentage
            percent = np.sum(bin_valid_mask) / np.sum(valid_mask) * 100.0

            ##
            bin_valid_err_rate = abs_err_rate[bin_valid_mask]
            bin_valid_err = abs_err[bin_valid_mask]
            if len(bin_valid_err_rate) == 0:
                continue

            # mean_err_rate = np.mean(bin_valid_err_rate) * 100.0
            median_err_rate = np.median(bin_valid_err_rate) * 100.0
            median_err = np.median(bin_valid_err)
            # min_err_rate = np.min(bin_valid_err_rate) * 100.0
            # max_err_rate = np.max(bin_valid_err_rate) * 100.0
            print('Bin[{:7.3f}m, {:7.3f}m] pixel percentage: {:7.3f}%'
                  ' | median abs_err: {:7.3f}m'
                  ' | median abs_err_rate: {:7.3f}%'
                  .format(bin_start, bin_end, percent, median_err, median_err_rate))

    def get(self):
        """
        Get current state of metrics.
        :return:
        """
        err_all = np.array(self._err_all)
        err_fg = np.array(self._err_fg)
        err_bg = np.array(self._err_bg)
        values = [np.mean(err_fg), np.mean(err_bg), np.mean(err_all)]

        return (self._names, values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation stereo output.')
    parser.add_argument('--test_dir',
                        type=str,
                        default='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_001/stereo_res',
                        # './test_eval_data/stereo_res/'
                        help='the dir of results')
    parser.add_argument('--gt_dir',
                        type=str,
                        default='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_001',
                        # './test_eval_data/stereo_gt/'
                        help='the dir of ground truth')
    parser.add_argument('--res_file',
                        type=str,
                        default='./stereo_train_01_res.txt',  # './test_eval_data/res.txt'
                        help='the dir of ground truth')
    parser.add_argument('--allow_missing',
                        type=uts.str2bool,
                        default='true',
                        help='the dir of ground truth')
    parser.add_argument('--is_resize',
                        type=bool,
                        default=True,  # False
                        help='')
    parser.add_argument('--net_size',
                        type=int,
                        nargs="+",
                        default=None)  # None  (1024, 320)
    parser.add_argument('--baseline',
                        type=float,
                        default=0.36,
                        help='Baseline(m).')
    parser.add_argument('--focal',
                        type=float,
                        default=2301.3147,
                        help='Focal length(pixel).')

    args = parser.parse_args()
    evaluator = StereoEval(args)
    # evaluator.eval_disparity()
    evaluator.eval_depth()
