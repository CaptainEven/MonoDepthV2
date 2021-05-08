# encoding=utf-8

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import PIL.Image as pil
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist, find_free_gpu, select_device

# calib_f_path = '/mnt/diskd/public/kitti/training/calib/000010.txt'

## kitti stereo rig parameters:
f = 718.335
cx = 609.5593
cy = 172.8540
b = 0.54  # m

# ## xiaomi stereo rig parameters:
# f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
# cx = 671.15643310546875
# cy = 384.32458496093750
# b = 0.12  # m

# ## apollo_stereo stereo rig parameters:
# f = 2301.3147
# cx = 1489.8536
# cy = 479.1750
# b = 0.36  # m

METRIC_SCALE = b * 10.0  # xiaomi binocular: 5.4, 1.2, 3.6


def disp2depth(b, f, disp):
    """
    :param b:
    :param f:
    :param disp:
    :return:
    """
    disp = disp.astype(np.float32)
    non_zero_inds = np.where(disp)

    depth = np.zeros_like(disp, dtype=np.float32)
    depth[non_zero_inds] = b * f / disp[non_zero_inds]

    return depth


def depth2disp(b, f, depth):
    """
    :param b:
    :param f:
    :param depth:
    :return:
    """
    depth = depth.astype(np.float32)
    positive_inds = np.where(depth > 0.0)

    disp = np.zeros_like(depth, dtype=np.float32)
    disp[positive_inds] = b * f / depth[positive_inds]

    return disp


def parse_args():
    parser = argparse.ArgumentParser(description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--device',
                        type=int,
                        default=-1,
                        help='GPU ids.')
    parser.add_argument('--weights_dir',
                        type=str,
                        default='./log_kitti/stereo_model/models/weights_5/',  # 'weights'
                        help='The directory to store weights file')
    parser.add_argument('--image_path',
                        type=str,
                        default='./assets/0000000200.jpg',  # apollo_train_0.jpg
                        help='path to a test image or folder of images')
    parser.add_argument('--video_path',
                        type=str,
                        default='./assets/indoor.mp4',
                        help='')
    parser.add_argument('--input_mode',
                        type=str,
                        default='img',  # video or img
                        help='Input data type: image or video.')
    parser.add_argument('--model_name',
                        type=str,
                        default='stereo_1024x320',  # 'stereo_768x448'
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320",
                            "stereo_768x448"
                        ])
    parser.add_argument("--weights_init",
                        default="pretrained",
                        type=str,
                        help="pretrained or scratch",
                        choices=["pretrained", "scratch"])
    parser.add_argument("--scales",
                        nargs="+",
                        type=int,
                        help="scales used in the loss",
                        default=[0, 1, 2, 3])
    parser.add_argument("--num_layers",
                        type=int,
                        help="number of resnet layers",
                        default=18,
                        choices=[18, 34, 50, 101, 152])  # resnet backbone selection
    parser.add_argument('--ext',
                        type=str,
                        default='jpg',  # 'jpg'
                        help='image extension to search for in folder')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--min_depth",
                        type=float,
                        default=0.1,
                        help="minimum depth")
    parser.add_argument("--max_depth",
                        type=float,
                        default=100.0,  # 100.0 10.0
                        help="maximum depth")

    return parser.parse_args()


def project_depth_to_pointcloud(calib, depth, max_height):
    """
    :param calib:
    :param depth:
    :param max_height:
    :return:
    """
    rows, cols = depth.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    uv_depth = np.stack([u, v, depth])
    uv_depth = uv_depth.reshape((3, -1))
    uv_depth = uv_depth.T
    cloud = calib.project_image_to_velo(uv_depth)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_height)  # depth > 0 and height < max_high
    return cloud[valid]


def test_stereo_net(args):
    """
    :param args:
    :return:
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if not os.path.isdir(args.weights_dir):
        print('[Err]: invalid weights directory.')
        return

    ## ---------- set device
    device = str(find_free_gpu())
    print('Using gpu: {:s}'.format(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = select_device(device='cpu' if not torch.cuda.is_available() else device)

    ## Define model
    # net = networks.StereoNet(num_layers=18,
    #                          pre_trained=False,
    #                          num_input_images=1,
    #                          num_ch_enc=[64, 64, 128, 256, 512],
    #                          scales=range(4),
    #                          num_output_channels=1,
    #                          use_skips=True)
    net = networks.MonoDepthV2(args)
    print(net)

    ## check whether weights_dir is leaf dir
    sub_dirs = [x for x in os.listdir(args.weights_dir) if os.path.isdir(args.weights_dir + '/' + x)]
    if len(sub_dirs) > 0:
        model_path = os.path.join(args.weights_dir, args.model_name)
    else:
        model_path = args.weights_dir

    print("-> Loading model from ", model_path)
    net_path = os.path.join(model_path, "net.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pre-trained encoder and decoder")

    # print('Net: \n', net)
    loaded_dict_net = torch.load(net_path, map_location=device)

    # Extract the height and width of image that this model was trained with
    net_height = loaded_dict_net['height']
    net_width = loaded_dict_net['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_net.items() if k in net.state_dict()}
    net.load_state_dict(filtered_dict_enc)
    print('{:s} loaded.'.format(net_path))

    ## Set network device and work mode
    net.to(device)
    net.eval()

    # FINDING INPUT IMAGES OR VIDEO
    if args.input_mode == 'img':
        if os.path.isfile(args.image_path):
            # Only testing on a single image
            paths = [args.image_path]
            output_directory = os.path.dirname(args.image_path)
        elif os.path.isdir(args.image_path):
            # Searching folder for images
            paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
            output_directory = args.image_path
        else:
            raise Exception("Can not find args.image_path: {}".format(args.image_path))

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, image_path in enumerate(paths):
                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue

                # Load image and preprocess
                img = pil.open(image_path).convert('RGB')
                img_width, img_height = img.size

                # Pre-processing
                img = img.resize((net_width, net_height), pil.LANCZOS)
                img = transforms.ToTensor()(img)  # [0, 1]
                img = transforms.Normalize(mean=0.45, std=1.0)(img).unsqueeze(0)  # normalize

                # ---------- PREDICTION
                img = img.to(device)
                # features = net.encoder.forward(img)
                # outputs = net.decoder.forward(features)
                outputs = net.forward(img)
                # ----------

                ## ----- Get output
                disp = outputs[("disp", 0)]  # [0, 1]
                disp_resized = torch.nn.functional.interpolate(disp,
                                                               (img_height, img_width),
                                                               mode="bilinear",
                                                               align_corners=False)  # [0, 1]

                ## ----- Resizing
                scaled_depth_inv, scaled_depth = disp_to_depth(disp_resized, args.min_depth, args.max_depth)

                ## Get file name
                output_name = os.path.splitext(os.path.basename(image_path))[0]

                ## ----- @even: Save metric depth image(gray scale) file
                metric_depth = METRIC_SCALE * scaled_depth  # turn scaled depth to metric depth
                metric_depth = metric_depth.squeeze().cpu().numpy()
                max_depth = np.max(metric_depth)  # get max depth value
                print('Max depth: {:.3f}m'.format(max_depth))

                metric_depth_img = np.uint16(metric_depth)  # transform to uint16 format, * 256.0
                metric_depth_img_f_path = output_directory + '/' + output_name + '_depthimg' + '.png'
                img = pil.fromarray(metric_depth_img)
                img.save(metric_depth_img_f_path)  # save depth image
                print('{:s} saved.'.format(metric_depth_img_f_path))
                metric_depth_npy_f_path = output_directory + '/' + output_name + '.npy'
                np.save(metric_depth_npy_f_path, metric_depth)  # save depth npy file
                print('{:s} saved.'.format(metric_depth_npy_f_path))


def test_simple(args):
    """
    Function to predict for a single image or folder of images
    :param args:
    :return:
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    ## ---------- set device
    device = str(find_free_gpu())
    print('Using gpu: {:s}'.format(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = select_device(device='cpu' if not torch.cuda.is_available() else device)

    download_model_if_doesnt_exist(args.model_name, args.weights_dir)

    ## check whether weights_dir is leaf dir
    sub_dirs = [x for x in os.listdir(args.weights_dir) if os.path.isdir(args.weights_dir + '/' + x)]
    if len(sub_dirs) > 0:
        model_path = os.path.join(args.weights_dir, args.model_name)
    else:
        model_path = args.weights_dir

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pre-trained encoder")
    encoder = networks.ResnetEncoder(18, False)
    print('Encoder: \n', encoder)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    print('{:s} loaded.'.format(encoder_path))

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pre-trained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    print('Depth decoder:\n', depth_decoder)

    loaded_dict_dec = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict_dec)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES OR VIDEO
    if args.input_mode == 'img':
        if os.path.isfile(args.image_path):
            # Only testing on a single image
            paths = [args.image_path]
            output_directory = os.path.dirname(args.image_path)
        elif os.path.isdir(args.image_path):
            # Searching folder for images
            paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
            output_directory = args.image_path
        else:
            raise Exception("Can not find args.image_path: {}".format(args.image_path))

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, image_path in enumerate(paths):
                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue

                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                img_width, img_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # ---------- PREDICTION
                input_image = input_image.to(device)
                features = encoder.forward(input_image)
                outputs = depth_decoder.forward(features)
                # ----------

                disp = outputs[("disp", 0)]  # [0, 1]
                disp_resized = torch.nn.functional.interpolate(disp,
                                                               (img_height, img_width),
                                                               mode="bilinear",
                                                               align_corners=False)  # [0, 1]

                # Saving numpy file
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                scaled_disp, _ = disp_to_depth(disp, args.min_depth, args.max_depth)
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

                ## @even: Save metric depth image(gray scale) file
                scaled_depth_inv, scaled_depth = disp_to_depth(disp_resized, args.min_depth, args.max_depth)
                metric_depth = METRIC_SCALE * scaled_depth  # turn scaled depth to metric depth
                metric_depth = metric_depth.squeeze().cpu().numpy()
                metric_depth_img = np.uint16(metric_depth * 256)  # *256 for better visualization
                metric_depth_img_f_path = output_directory + '/' + output_name + '_depth_metric.png'
                img = pil.fromarray(metric_depth_img)
                img.save(metric_depth_img_f_path)
                print('{:s} saved.'.format(metric_depth_img_f_path))

                ## @even: Save metric depth numpy file
                depth_metric_np_f_path = output_directory + '/' + output_name + '_depth_metric.npy'
                np.save(depth_metric_np_f_path, metric_depth)
                print('{:s} saved.'.format(depth_metric_np_f_path))
                max_depth = np.max(metric_depth)  # get max depth value
                print('Max depth: {:.3f}m'.format(max_depth))

                # ## @even: convert to KITTI pseudo lidar data
                # if not os.path.isfile(calib_f_path):
                #     print('[Err]: invalid calib file path.')
                # calib = kitti_util.Calibration(calib_f_path)
                # lidar = project_depth_to_pointcloud(calib, metric_depth, 1.0)
                # lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
                # lidar = lidar.astype(np.float32)
                # lidar_bin_f_path = output_directory + '/' + output_name + '_pointcloud.bin'
                # lidar.tofile(lidar_bin_f_path)
                # print('{:s} saved.'.format(lidar_bin_f_path))

                ## @even: transform metric depth to disparity, save disparity image to show
                dispimg_f_path = output_directory + '/' + output_name + '_dispimg.png'
                disp_f_path = output_directory + '/' + output_name + '_disparity.png'
                disp = depth2disp(b, f, metric_depth)  # transform depth to disparity
                cv2.imwrite(disp_f_path, disp)  # save metric disparity
                print('{:s} saved.'.format(disp_f_path))
                disp = (disp * 256.0).astype('uint16')  # * 256 for better visualization
                disp_img = pil.fromarray(disp)
                disp_img.save(dispimg_f_path)  # save disparity image for visualization
                print('{:s} saved.'.format(dispimg_f_path))

                # Saving color-mapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                v_max = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=v_max)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                color_mapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(color_mapped_im)
                name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                print("   Processed {:d} of {:d} images - saved prediction to {}"
                      .format(idx + 1, len(paths), name_dest_im))

    elif args.input_mode == 'video':
        if not os.path.isfile(args.video_path):
            print('[Err]: invalid video file path.')
            return

        output_directory = os.path.dirname(args.video_path)
        video_name = os.path.split(args.video_path)[-1][:-4]

        cap = cv2.VideoCapture(args.video_path)
        FRAME_NUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("-> Predicting on {:d} test images".format(FRAME_NUM))
        for i in tqdm(range(FRAME_NUM)):
            success, frame = cap.read()
            if not success:
                break

            output_name = '{:05d}'.format(i)
            with torch.no_grad():
                # Load image and preprocess
                input_image = pil.fromarray(frame).convert('RGB')
                img_width, img_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # ---------- PREDICTION
                input_image = input_image.to(device)
                features = encoder.forward(input_image)
                outputs = depth_decoder.forward(features)
                # ----------

                disp = outputs[("disp", 0)]  # [0, 1]
                disp_resized = torch.nn.functional.interpolate(disp,
                                                               (img_height, img_width),
                                                               mode="bilinear",
                                                               align_corners=False)  # [0, 1]

                disp_resized_np = disp_resized.squeeze().cpu().numpy()

            # Saving color-mapped depth image
            v_max = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=v_max)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            color_mapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(color_mapped_im)
            name_dest_im = os.path.join(output_directory, "{}_depth_show.jpeg".format(output_name))
            im.save(name_dest_im)

        ## transform depth_show into a video
        print('Zipping to mp4...')
        depth_video_path = output_directory + '/{:s}'.format(video_name) + '_depth.mp4'
        cmd_str = 'ffmpeg -f image2 -r 6 -i {}/%05d_depth_show.jpeg -b 5000k -c:v mpeg4 {}' \
            .format(output_directory, depth_video_path)
        print(cmd_str)
        os.system(cmd_str)

    print('-> Done!')


def build_img_depth_show(rgb_video_path, show_video_path):
    """
    :param rgb_video_path:
    :param show_video_path:
    :return:
    """
    if not (os.path.isfile(rgb_video_path) and os.path.isfile(show_video_path)):
        print('[Err]: invalid file path.')
        return

    output_directory = os.path.dirname(rgb_video_path)
    video_name = os.path.split(rgb_video_path)[-1][:-4]

    cap_rgb = cv2.VideoCapture(rgb_video_path)
    cap_depth = cv2.VideoCapture(show_video_path)
    frame_num_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num_depth = int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (frame_num_rgb == frame_num_depth)
    FRAME_NUM = frame_num_rgb

    print("-> Predicting on {:d} test frames".format(FRAME_NUM))
    for i in tqdm(range(FRAME_NUM)):
        success_rgb, frame_rgb = cap_rgb.read()
        success_depth, frame_depth = cap_depth.read()

        rgb_shape = frame_rgb.shape
        depth_shape = frame_depth.shape
        success = success_rgb & success_depth & (rgb_shape == depth_shape)

        if not success:
            break

        ## build a new frame
        h, w, c = rgb_shape
        new_frame = np.zeros((h * 2, w, c), dtype=np.uint8)
        new_frame[:h, :, :] = frame_rgb
        new_frame[h:h * 2, :, :] = frame_depth
        # print(new_frame.shape)

        ## save new frame
        frame_save_path = output_directory + '/{:05d}_show.jpeg'.format(i)
        im = pil.fromarray(new_frame)
        im.save(frame_save_path)

    ## transform depth_show into a video
    print('Zipping to mp4...')
    show_video_path = output_directory + '/{:s}'.format(video_name) + '_show.mp4'
    cmd_str = 'ffmpeg -f image2 -r 6 -i {}/%05d_show.jpeg -b 5000k -c:v mpeg4 {}' \
        .format(output_directory, show_video_path)
    print(cmd_str)
    os.system(cmd_str)
    print('-> Done!')


def get_disps_depths_for_apollo_stereo(args,
                                       data_root,
                                       disp_dir='',
                                       depth_dir='',
                                       is_resize=True):
    """
    :param args:
    :param data_root:
    :param disp_dir:
    :param depth_dir:
    :param is_resize:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    if not os.path.isdir(disp_dir):
        if disp_dir != '':
            os.makedirs(disp_dir)
        else:
            dir_name = os.path.split(data_root)[-1]
            disp_dir = data_root + '/stereo_res/disparity'.format(dir_name)
            depth_dir = data_root + '/stereo_res/depth'.format(dir_name)
            if not os.path.isdir(disp_dir):
                os.makedirs(disp_dir)
            if not os.path.isdir(depth_dir):
                os.makedirs(depth_dir)

    ## ---------- set device
    if args.device == -1:
        device = str(find_free_gpu())
    else:
        device = str(args.device)
    print('Using gpu: {:s}'.format(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = select_device(device='cpu' if not torch.cuda.is_available() else device)

    download_model_if_doesnt_exist(args.model_name, args.weights_dir)

    ## check whether weights_dir is leaf dir
    sub_dirs = [x for x in os.listdir(args.weights_dir) if os.path.isdir(args.weights_dir + '/' + x)]
    if len(sub_dirs) > 0:
        model_path = os.path.join(args.weights_dir, args.model_name)
    else:
        model_path = args.weights_dir

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pre-trained encoder")
    encoder = networks.ResnetEncoder(num_layers=18,
                                     pre_trained=False,
                                     num_input_images=1,
                                     num_ch_enc=[64, 64, 128, 256, 512])
    print('Encoder:\n', encoder)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pre-trained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc,
                                          scales=range(4),
                                          num_output_channels=1,
                                          use_skips=True)
    print('Depth decoder:\n', depth_decoder)

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    ## Process each frame
    image_02_dir = data_root + '/image_02'
    img_paths = [image_02_dir + '/' + x for x in os.listdir(image_02_dir)]
    img_paths.sort()

    with torch.no_grad():
        for idx, image_path in enumerate(img_paths):
            img_name = os.path.split(image_path)[-1]

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            img_width, img_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # ---------- PREDICTION
            input_image = input_image.to(device)
            features = encoder.forward(input_image)
            outputs = depth_decoder.forward(features)
            # ----------

            disp = outputs[("disp", 0)]  # [0, 1]
            if is_resize:  # if need to be resized to original image size
                disp_resized = torch.nn.functional.interpolate(disp,
                                                               (img_height, img_width),
                                                               mode="bilinear",
                                                               align_corners=False)  # [0, 1]

            ## @even: Save metric depth image(gray scale) file
            if is_resize:
                scaled_depth_inv, scaled_depth = disp_to_depth(disp_resized, args.min_depth, args.max_depth)
            else:
                scaled_depth_inv, scaled_depth = disp_to_depth(disp, args.min_depth, args.max_depth)

            metric_depth = METRIC_SCALE * scaled_depth  # turn scaled depth to metric depth
            metric_depth = metric_depth.squeeze().cpu().numpy()
            metric_depth_img = np.uint16(metric_depth)  # transform to uint16 format
            metric_depth_img_f_path = depth_dir + '/' + img_name.replace('.jpg', '.png')
            img = pil.fromarray(metric_depth_img)
            img.save(metric_depth_img_f_path)  # save depth image
            print('{:s} saved.'.format(metric_depth_img_f_path))
            metric_depth_npy_f_path = depth_dir + '/' + img_name.replace('.jpg', '.npy')
            np.save(metric_depth_npy_f_path, metric_depth)  # save depth npy file
            print('{:s} saved.'.format(metric_depth_npy_f_path))
            max_depth = np.max(metric_depth)  # get max depth value
            print('Max depth: {:.3f}m'.format(max_depth))

            ## @even: transform metric depth to disparity, save disparity image to show
            # dispimg_f_path = disp_dir + '/' + output_name + '_dispimg.png'
            disp_f_path = disp_dir + '/' + img_name.replace('.jpg', '.png')  # use png to save disparity image file
            disp = depth2disp(b, f, metric_depth)  # transform depth to disparity
            disp = disp * 200.0
            disp = disp.astype(np.uint16)
            cv2.imwrite(disp_f_path, disp)  # save metric disparity * 200
            print('[{:d}/{:d}]: {:s} saved.\n'.format(idx + 1, len(img_paths), disp_f_path))
            # disp = (disp * 256.0).astype('uint16')  # * 256 for better visualization
            # disp_img = pil.fromarray(disp)
            # disp_img.save(dispimg_f_path)  # save disparity image for visualization
            # print('{:s} saved.'.format(dispimg_f_path))


# from torchsummary import summary


def test_each_layer_of_net(args):
    """
    :param args:
    :return:
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if not os.path.isdir(args.weights_dir):
        print('[Err]: invalid weights directory.')
        return

    ## ---------- set device
    device = str(find_free_gpu())
    print('Using gpu: {:s}'.format(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    # device = select_device(device='cpu' if not torch.cuda.is_available() else device)
    device = 'cpu'

    ## Define model
    net = networks.MonoDepthV2(args)
    print(net)

    ## check whether weights_dir is leaf dir
    sub_dirs = [x for x in os.listdir(args.weights_dir) if os.path.isdir(args.weights_dir + '/' + x)]
    if len(sub_dirs) > 0:
        model_path = os.path.join(args.weights_dir, args.model_name)
    else:
        model_path = args.weights_dir

    print("-> Loading model from ", model_path)
    net_path = os.path.join(model_path, "net.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pre-trained encoder and decoder")

    # print('Net: \n', net)
    loaded_dict_net = torch.load(net_path, map_location=device)

    # extract the height and width of image that this model was trained with
    net_height = loaded_dict_net['height']
    net_width = loaded_dict_net['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_net.items() if k in net.state_dict()}
    net.load_state_dict(filtered_dict_enc)
    print('{:s} loaded.'.format(net_path))

    ## set network device and mode
    net.to(device)
    net.eval()

    X = torch.ones([1, 3, 320, 1024]).to(device)

    layers_dict = dict(net.named_children())
    for layer_i, (layer_name, layer) in enumerate(layers_dict.items()):
        ## traverse each child parameter of the layer
        for param_i, (param_name, param) in enumerate(layer.named_parameters()):
            print(param_name)
            X = param_name.forward(X)
            print(X.shape)

    # summary(net, (3, 1024, 320), batch_size=1, device=device)


if __name__ == '__main__':
    args = parse_args()
    # test_simple(args)
    test_stereo_net(args)
    # test_each_layer_of_net(args)

    # build_img_depth_show(rgb_video_path='./assets/indoor.mp4',
    #                      show_video_path='./assets/indoor_depth.mp4')

    # get_disps_depths_for_apollo_stereo(args=args,
    #                                    data_root='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_002',
    #                                    disp_dir='',
    #                                    depth_dir='',
    #                                    is_resize=True)
