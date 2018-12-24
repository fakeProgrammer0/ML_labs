#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-8 下午2:54
# @Author  : xiezheng
# @Site    : 
# @File    : test_on_FDDB.py

import logging
import os
import sys

import cv2

from tools.train_detect import MtcnnDetector

data_dir = '/home/datasets/FDDB'
out_dir = 'results/FDDB-results'

# log part
logger = logging.getLogger("Test-FDDB")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
# %(asctime)s: 打印日志的时间    %(levelname)s: 打印日志级别名称    %(message)s: 打印日志信息

# StreamHandler: print log
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.INFO)
stream_handler.formatter = formatter  # 也可以直接给formatter赋值
logger.addHandler(stream_handler)


# FileHandler: save log-file
# filename = os.path.join(out_dir, 'output_%s.log' % (time.strftime("%Y%m%d%H%M%S", time.localtime())))
# file_handler = logging.FileHandler(filename)
# file_handler.setLevel(level=logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-folds-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))
        imdb.append(image_names)
    return imdb


if __name__ == "__main__":
    mtcnn_detector = MtcnnDetector(p_model_path='./results/pnet/log_bs512_lr0.010_072402/check_point/model_050.pth',
                                   r_model_path='./results/rnet/log_bs512_lr0.001_072502/check_point/model_050.pth',
                                   o_model_path='./results/pnet/log_bs512_lr0.001_0726402/check_point/model_050.pth',
                                   min_face_size=12,
                                   use_cuda=False)
    # logger.info("Init the MtcnnDetector.")
    imdb = get_imdb_fddb(data_dir)
    nfold = len(imdb)

    for i in range(nfold):
        image_names = imdb[i]
        dets_file_name = os.path.join(out_dir, 'FDDB-det-fold-%02d.txt' % (i + 1))
        fid = open(dets_file_name, 'w')
        # image_names_abs = [os.path.join(data_dir, 'originalPics', image_name + '.jpg') for image_name in image_names]

        for idx, im_name in enumerate(image_names):
            img_path = os.path.join(data_dir, 'originalPics', im_name + '.jpg')
            img = cv2.imread(img_path)
            boxes, _ = MtcnnDetector.detect_face(img)

            if boxes is None:
                fid.write(im_name + '\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue

            fid.write(im_name + '\n')
            fid.write(str(len(boxes)) + '\n')

            for box in boxes:
                fid.write('%f %f %f %f %f\n' % (
                    float(box[0]), float(box[1]), float(box[2] - box[0] + 1), float(box[3] - box[1] + 1), box[4]))

        fid.close()
