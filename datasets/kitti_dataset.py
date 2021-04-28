# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import PIL.Image as pil
import numpy as np
import skimage.transform

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """
    Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        """
        :return:
        """
        line = self.file_names[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_idx, side, do_flip):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :param do_flip:
        :return:
        """
        color = self.loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class MyKittiDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        """
        super(MyKittiDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        K = np.array([[718.335, 0.0, 609.5593, 0.0],
                      [0.0, 718.335, 172.8540, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.K = np.eye(4, dtype=np.float32)
        self.K[0] = K[0] / self.img_width
        self.K[1] = K[1] / self.img_height

        # self.K[0, 2] = 0.5
        # self.K[1, 2] = 0.5

        self.full_res_shape = (self.img_width, self.img_height)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.img_ext = '.jpg'  # use jpg image format

    def check_depth(self):
        """
        :return:
        """
        return False

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path,
                                  folder,
                                  "image_0{}/data/".format(self.side_map[side]),
                                  f_str)

        return image_path

    def get_color(self, folder, frame_idx, side, do_flip):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :param do_flip:
        :return:
        """
        color = self.loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class ApolloDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        """
        super(ApolloDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        K = np.array([[2301.3147, 0.0, 1489.8536, 0.0],
                      [0.0, 2301.3147, 479.1750, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.K = np.eye(4, dtype=np.float32)
        self.K[0] = K[0] / self.img_width
        self.K[1] = K[1] / self.img_height

        # self.K[0, 2] = 0.5
        # self.K[1, 2] = 0.5

        self.full_res_shape = (self.img_width, self.img_height)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.img_ext = '.jpg'  # Apollo use jpg image format

    def check_depth(self):
        """
        :return:
        """
        return False

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:05d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path,
                                  folder,
                                  "image_0{}/".format(self.side_map[side]),
                                  f_str)

        return image_path

    def get_color(self, folder, frame_idx, side, do_flip):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :param do_flip:
        :return:
        """
        color = self.loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class MyExperimentDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(MyExperimentDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        K = np.array(
            [[998.72290039062500, 0.0, 671.15643310546875, 0.0],
             [0.0, 1000.0239868164063, 384.32458496093750, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32
        )
        self.K = np.eye(4, dtype=np.float32)
        self.K[0] = K[0] / self.img_width  # the first row
        self.K[1] = K[1] / self.img_height  # the second row

        self.full_res_shape = (self.img_width, self.img_height)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.img_ext = '.png'  #

    def check_depth(self):
        """
        :return:
        """
        return False

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:05d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path,
                                  folder,
                                  "image_0{}/".format(self.side_map[side]),
                                  f_str)

        return image_path

    def get_color(self, folder, frame_idx, side, do_flip):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :param do_flip:
        :return:
        """
        color = self.loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """
    KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path,
                                  folder,
                                  "image_0{}/data".format(self.side_map[side]),
                                  f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        :param folder:
        :param frame_index:
        :param side:
        :param do_flip:
        :return:
        """
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(depth_gt,
                                            self.full_res_shape[::-1],
                                            order=0,
                                            preserve_range=True,
                                            mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:06d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_idx, side):
        """
        :param folder:
        :param frame_idx:
        :param side:
        :return:
        """
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str
        )

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        :param folder:
        :param frame_index:
        :param side:
        :param do_flip:
        :return:
        """
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str
        )

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
