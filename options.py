# encoding=utf-8

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import argparse
import os

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS: eg: # '/mnt/diskd/public/kitti',
        self.parser.add_argument("--data_path",
                                 type=str,  # '/mnt/diskc/even/monodepthv2_dataset'
                                 default='/mnt/diskd/public/kitti_data/',
                                 help="path to the training data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory to save log file and checkpoins.",
                                 default='./log_kitti')

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 default="stereo_model",
                                 help="the name of the folder to save the model in")
        self.parser.add_argument("--split",
                                 type=str,
                                 default="eigen_full",  # "eigen_full", "my_split"
                                 help="which training split to use",
                                 choices=[
                                     "eigen_zhou",
                                     "eigen_full",
                                     "odom",
                                     "benchmark",
                                     "my_split",
                                     "apollo_split"
                                 ])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])  # resnet backbone selection
        self.parser.add_argument("--dataset",
                                 type=str,
                                 default="my_kitti",  # "kitti", "my_experiment"
                                 help="dataset to train on",
                                 choices=[
                                     "kitti",
                                     "kitti_odom",
                                     "kitti_depth",
                                     "kitti_test",
                                     "my_experiment",
                                     "apollo_stereo",
                                     "my_kitti"
                                 ])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")

        ## ----- 640×192  1024x320  1280×720  768×448
        self.parser.add_argument("--height",
                                 type=int,
                                 default=320,  # 192, 448
                                 help="input image height")
        self.parser.add_argument("--width",
                                 type=int,
                                 default=1024,  # 640, 768
                                 help="input image width")
        self.parser.add_argument("--image_height",
                                 type=int,
                                 default=375,  # 375, 720, 960,
                                 help="Input image height")
        self.parser.add_argument("--image_width",
                                 type=int,
                                 default=1242,  # 1242, 1280, 3130
                                 help="input image width")
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 default=0.1,
                                 help="minimum depth(m)")
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 default=100.0,  # 100.0, 10.0, 200.0
                                 help="maximum depth(m)")
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--use_stereo",
                                 type=bool,
                                 default=True,
                                 help="if set, uses stereo pair for training")
        self.parser.add_argument("--frame_ids",
                                 type=int,
                                 default=[0],  # [0, -1, 1]
                                 nargs="+",
                                 help="frames to load")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 default=10,  # 16, 12, 10
                                 help="batch size")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 default=1e-4,  # 1e-4
                                 help="learning rate")
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 default=50,
                                 help="number of epochs")
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 default="pretrained",
                                 type=str,
                                 help="pretrained or scratch",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 default="separate_resnet",
                                 help="normal or shared",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 default=0,  # 8, 10, 12
                                 help="number of dataloader workers")

        # LOADING options
        # './log_apollo/stereo_model/models/weights_1/'  './weights/stereo_1024x320'
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 default=None,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 default=["net"],  # ["encoder", "depth", "pose_encoder", "pose"]
                                 help="models to load")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 default=10,
                                 help="number of batches between each tensorboard log")
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 default=1,
                                 help="number of epochs between each save")

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="my_split",  # "eigen"
                                 choices=["eigen",
                                          "eigen_benchmark",
                                          "benchmark",
                                          "odom_9",
                                          "odom_10",
                                          "my_split"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
