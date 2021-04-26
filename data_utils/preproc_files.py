# encoding=utf-8

import os
import cv2
import numpy as np
from tqdm import tqdm
from test_simple import disp2depth


def find_not_exist():
    f_path = '../splits/kitti_archives_to_download.txt'
    if not os.path.isfile(f_path):
        print('[Err]: invalid file path.')
        return

    ## Find files endwith .zip
    dir_path = '/mnt/diskd/public/kitti_data'
    if not os.path.isdir(dir_path):
        print('[Err]: invalid directory.')
        return

    f_name_list = [x for x in os.listdir(dir_path)]
    print('Total {:d} files already exists.'.format(len(f_name_list)))

    cnt = 0
    with open(f_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split('/')
            f_name = items[-1]
            if not f_name.endswith('.zip'):
                print('[Warning]: did not find zip file.')
                continue

            if f_name not in f_name_list:
                print('{:s} not downloaded yet...'.format(f_name))
                cnt += 1
    print('Total {:d} file not downloaded yet.'.format(cnt))


def gen_my_split(data_root, split_root):
    """
    :param data_root:
    :param split_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    image_02_dir = data_root + '/image_02'
    image_03_dir = data_root + '/image_03'
    if not (os.path.isdir(image_02_dir) and os.path.isdir(image_03_dir)):
        print('[Err]: image_02 or image_03 dir not exist.')
        retrun

    if not os.path.isdir(split_root):
        os.makedirs(split_root)
        print('{:s} made.'.format(split_root))

    image_02_f_names = [x for x in os.listdir(image_02_dir) if x.endswith('.png')]
    image_03_f_names = [x for x in os.listdir(image_03_dir) if x.endswith('.png')]
    image_02_f_names.sort()
    image_03_f_names.sort()
    assert image_02_f_names == image_03_f_names

    train_f_path = split_root + '/train_files.txt'
    valid_f_path = split_root + '/val_files.txt'

    with open(train_f_path, 'w', encoding='utf-8') as f_train, \
            open(valid_f_path, 'w', encoding='utf-8') as f_valid:
        for img_name in tqdm(image_02_f_names):
            frame_id = int(img_name.split('.')[0][:-1])
            # print(frame_id)

            if np.random.random() < 0.05:
                f_valid.write('img_pairs {:d} l\n'.format(frame_id))
                f_valid.write('img_pairs {:d} r\n'.format(frame_id))
            else:
                f_train.write('img_pairs {:d} l\n'.format(frame_id))
                f_train.write('img_pairs {:d} r\n'.format(frame_id))


def gen_apollo_split(data_root, split_root,
                     train_dirs=[
                         'stereo_train_001',
                         'stereo_train_002',
                         'stereo_train_003',
                         'stereo_test'
                     ],
                     test_dirs=['stereo_test'],
                     ext='.jpg'):
    """
    :param data_root:
    :param split_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    if not os.path.isdir(split_root):
        os.makedirs(split_root)
        print('{:s} made.'.format(split_root))

    ## Get sub-directories
    train_dir_names = [x for x in os.listdir(data_root) if x in train_dirs and
                       os.path.isdir(data_root + '/' + x)]
    test_dir_names = [x for x in os.listdir(data_root) if x in test_dirs and
                      os.path.isdir(data_root + '/' + x)]
    train_dir_names.sort()
    test_dir_names.sort()

    train_f_path = split_root + '/train_files.txt'
    valid_f_path = split_root + '/val_files.txt'

    with open(train_f_path, 'w', encoding='utf-8') as f_train:
        for dir_name in train_dir_names:
            dir_path = data_root + '/' + dir_name

            image_02_dir = dir_path + '/image_02'
            image_03_dir = dir_path + '/image_03'
            if not (os.path.isdir(image_02_dir) and os.path.isdir(image_03_dir)):
                print('[Err]: image_02 or image_03 dir not exist.')
                continue

            image_02_f_names = [x for x in os.listdir(image_02_dir) if x.endswith(ext)]
            image_03_f_names = [x for x in os.listdir(image_03_dir) if x.endswith(ext)]
            image_02_f_names.sort()
            image_03_f_names.sort()
            assert image_02_f_names == image_03_f_names

            for img_name in tqdm(image_02_f_names):
                frame_id = int(img_name.split('.')[0])
                # print(frame_id)

                f_train.write('{:s} {:d} l\n'.format(dir_name, frame_id))
                f_train.write('{:s} {:d} r\n'.format(dir_name, frame_id))

    with open(valid_f_path, 'w', encoding='utf-8') as f_test:
        for dir_name in test_dir_names:
            dir_path = data_root + '/' + dir_name

            image_02_dir = dir_path + '/image_02'
            image_03_dir = dir_path + '/image_03'
            if not (os.path.isdir(image_02_dir) and os.path.isdir(image_03_dir)):
                print('[Err]: image_02 or image_03 dir not exist.')
                continue

            image_02_f_names = [x for x in os.listdir(image_02_dir) if x.endswith(ext)]
            image_03_f_names = [x for x in os.listdir(image_03_dir) if x.endswith(ext)]
            image_02_f_names.sort()
            image_03_f_names.sort()
            assert image_02_f_names == image_03_f_names

            for img_name in tqdm(image_02_f_names):
                frame_id = int(img_name.split('.')[0])
                # print(frame_id)

                f_test.write('{:s} {:d} l\n'.format(dir_name, frame_id))
                f_test.write('{:s} {:d} r\n'.format(dir_name, frame_id))


def rename_files(data_root, ext='.jpg'):
    """
    :param data_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    image_02_dir = data_root + '/image_02'
    image_03_dir = data_root + '/image_03'
    if not (os.path.isdir(image_02_dir) and os.path.isdir(image_03_dir)):
        print('[Err]: image_02 or image_03 dir not exist.')
        retrun

    image_02_f_names = [x for x in os.listdir(image_02_dir) if x.endswith(ext)]
    image_03_f_names = [x for x in os.listdir(image_03_dir) if x.endswith(ext)]
    image_02_f_names.sort()
    image_03_f_names.sort()
    assert len(image_02_f_names) == len(image_03_f_names)

    for fr_i, img_name in tqdm(enumerate(image_02_f_names)):
        left_img_path = image_02_dir + '/' + img_name
        right_img_path = image_03_dir + '/' + img_name.replace('Camera_5', 'Camera_6')

        if not (os.path.isfile(left_img_path) and os.path.isfile(right_img_path)):
            print('Stereo pair {:s} not exists.')
            continue

        new_left_img_path = image_02_dir + '/{:05d}{:s}'.format(fr_i, ext)
        new_right_img_path = image_03_dir + '/{:05d}{:s}'.format(fr_i, ext)
        os.rename(left_img_path, new_left_img_path)
        os.rename(right_img_path, new_right_img_path)


def rename_a_dir(dir_path, ext='.png'):
    """
    :param dir_path:
    :return:
    """
    if not os.path.isdir(dir_path):
        print('[Err]: invalid directoty.')
        return

    img_name_list = [x for x in os.listdir(dir_path) if x.endswith(ext)]
    img_name_list.sort()
    print('Total {:d} files to be renmaed.'.format(len(img_name_list)))

    for i, img_name in tqdm(enumerate(img_name_list)):
        old_img_path = dir_path + '/' + img_name
        new_img_path = dir_path + '/' + '/{:05d}{:s}'.format(i, ext)

        if not os.path.isfile(old_img_path):
            print('[Warning]: invalid file path:{:s}.'.format(old_img_path))
            continue

        if not os.path.isfile(new_img_path):
            os.rename(old_img_path, new_img_path)
        else:
            print('{:s} already exists.'.format(new_img_path))


def call_rename_files(data_root):
    """
    :param data_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    ## Get sub-directories
    dir_names = [x for x in os.listdir(data_root) if os.path.isdir(data_root + '/' + x)]
    for dir_name in tqdm(dir_names):
        dir_path = data_root + '/' + dir_name

        rename_files(dir_path)


def gen_GT_metric_depth_files(disp_dir, depth_dir, net_size=None):
    """
    :param disp_dir:
    :param depth_dir:
    :param net_size:
    :return:
    """
    # apollo stereo rig parameters
    f = 2301.3147
    cx = 1489.8536
    cy = 479.1750
    b = 0.36  # m

    if not os.path.isdir(disp_dir):
        print('[Err]: invalid disparity dir.')
        return

    if not os.path.isdir(depth_dir):
        os.makedirs(depth_dir)
        print('{:s} made.'.format(depth_dir))

    disp_f_names = [x for x in os.listdir(disp_dir) if x.endswith('.png')]
    disp_f_names.sort()
    print('Total {:d} disparity file need to be converted to depth file.'.format(len(disp_f_names)))

    for i, disp_name in enumerate(disp_f_names):
        # print(disp_name)

        ## ----- load disparity file
        disp_f_path = disp_dir + '/' + disp_name
        if not os.path.isfile(disp_f_path):
            print('[Warning]: disparity image {:s} not exists.'.format(disp_f_path))
            continue
        disp = np.float32(cv2.imread(disp_f_path, cv2.IMREAD_UNCHANGED))

        ## ----- transform disparity to depth file(.npy file)
        disp /= 200.0  # To get metric disparity: in apollo stereo readme
        depth = disp2depth(b, f, disp)
        if net_size is not None:
            depth = cv2.resize(depth, net_size)

        depth_img_f_path= depth_dir + '/' + disp_name
        depth_img = depth.astype(np.uint16)
        cv2.imwrite(depth_img_f_path, depth_img)
        print('Max depth: {:.3f}m'.format(np.max(depth)))
        print('{:s} saved.'.format(depth_img_f_path))

        ## save metric depth file(.npy file)
        depth_npy_f_path = depth_dir + '/' + disp_name.replace('.png', '.npy')
        np.save(depth_npy_f_path, depth)
        print('{:s} saved.'.format(depth_npy_f_path))
        print('{:d}/{:d}\n'.format(i + 1, len(disp_f_names)))


def cp_and_rename_files(data_root):
    """
    :param data_root:
    :return:
    """
    if not os.path.isdir(data_root):
        print('[Err]: invalid')

def modify_split_file(in_f_path, out_f_path):
    if not os.path.isfile(in_f_path):
        print('[Err]: invalid input file path.')
        return

    with open(in_f_path, 'r', encoding='utf-8') as f_in, \
            open(out_f_path, 'w', encoding=utf-8) as f_out:
        for line in f_in.readlines():
            print(line)
            items = line.split(' ')
            print(items)



if __name__ == '__main__':
    # find_not_exist()
    #
    # gen_my_split(data_root='/mnt/diskc/even/monodepthv2_dataset/img_pairs',
    #              split_root='../splits/my_split')

    # rename_files(data_root='/mnt/diskc/even/monodepthv2_dataset/img_pairs')
    # call_rename_files(data_root='/mnt/diskc/even/monodepthv2_dataset/ApolloScape')
    # rename_a_dir(dir_path='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_003/disparity')

    # gen_apollo_split(data_root='/mnt/diskc/even/monodepthv2_dataset/ApolloScape',
    #                  split_root='../splits/apollo_split')

    gen_GT_metric_depth_files(disp_dir='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_002/disparity',
                              depth_dir='/mnt/diskc/even/monodepthv2_dataset/ApolloScape/stereo_train_002/depth',
                              net_size=None)  # (1024, 320)  None
