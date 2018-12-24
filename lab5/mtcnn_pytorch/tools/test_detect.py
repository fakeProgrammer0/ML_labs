import time

import torch.utils.model_zoo as model_zoo
import cv2

from models import PNet,ONet,RNet

import logging
from tools.image_tools import *
import tools.utils as utils

model_urls = {
    'pnet': 'https://github.com/xiezheng-cs/mtcnn_pytorch/releases/download/mtcnn/pnet-3da9e965.pt',
    'rnet': 'https://github.com/xiezheng-cs/mtcnn_pytorch/releases/download/mtcnn/rnet-ea379816.pt',
    'onet': 'https://github.com/xiezheng-cs/mtcnn_pytorch/releases/download/mtcnn/onet-4b09b161.pt',
}

logger = logging.getLogger("app")
class MtcnnDetector(object):
    ''' P, R, O net for face detection and landmark alignment'''

    def __init__(self,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 use_cuda=True):
        self.pnet_detector, self.rnet_detector, self.onet_detector = self.create_mtcnn_net(use_cuda)
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def create_mtcnn_net(self, use_cuda=True):
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        pnet = PNet()
        pnet.load_state_dict(model_zoo.load_url(model_urls['pnet']))
        pnet.to(self.device).eval()

        onet = ONet()
        onet.load_state_dict(model_zoo.load_url(model_urls['onet']))
        onet.to(self.device).eval()

        rnet = RNet()
        rnet.load_state_dict(model_zoo.load_url(model_urls['rnet']))
        rnet.to(self.device).eval()

        return pnet, rnet, onet

    def generate_bounding_box(self, map, reg, scale, threshold):
        '''
        generate bbox from feature map
        for PNet, there exists no fc layer, only convolution layer ,so feature map n x m x 1/4
        Parameters:
            map: numpy array , n x m x 1, detect score for each position
            reg: numpy array , n x m x 4, bbox
            scale: float number, scale of this detection
            threshold: float number, detect threshold
        Returns:
            bbox array
        '''
        stride = 2
        cellsize = 12

        t_index = np.where(map > threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i]
                              for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])

        score = map[t_index[0], t_index[1], 0]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round(
                                     (stride * t_index[1] + cellsize) / scale),
                                 np.round(
                                     (stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T

    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel,input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(
            img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5, input bboxes
            w: float number, width of the input image
            h: float number, height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1, start point of the bbox in target image
            edy, edx : numpy array, n x 1, end point of the bbox in target image
            y, x : numpy array, n x 1, start point of the bbox in original image
            ey, ex : numpy array, n x 1, end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1, height and width of the bbox
        """

        tmpw = (bboxes[:, 2] - bboxes[:, 0]).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1]).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox,))
        dy = np.zeros((numbox,))
        edx, edy = tmpw.copy(), tmph.copy()

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w)
        edx[tmp_index] = tmpw[tmp_index] + w - ex[tmp_index]
        ex[tmp_index] = w

        tmp_index = np.where(ey > h)
        edy[tmp_index] = tmph[tmp_index] + h - ey[tmp_index]
        ey[tmp_index] = h

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array, input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12
        current_scale = float(net_size) / \
                        self.min_face_size  # find initial scale
        im_resized = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        # fcn for pnet
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            image_tensor = convert_image_to_tensor(im_resized)
            feed_imgs = image_tensor.unsqueeze(0)

            feed_imgs = feed_imgs.to(self.device)

            cls_map, reg = self.pnet_detector(feed_imgs)
            cls_map_np = convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = convert_chwTensor_to_hwcNumpy(reg.cpu())

            boxes = self.generate_bounding_box(
                cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0]
        bh = all_boxes[:, 3] - all_boxes[:, 1]

        boxes = np.vstack([all_boxes[:, 0],
                           all_boxes[:, 1],
                           all_boxes[:, 2],
                           all_boxes[:, 3],
                           all_boxes[:, 4]
                           ])

        boxes = boxes.T

        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 all_boxes[:, 4]
                                 ])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        if dets is None:
            return None, None

        dets = utils.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                if tmph[i] > 0 and tmpw[i] > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                    tmp[dy[i]:edy[i], dx[i]:edx[i], :] = im[y[i]:ey[i], x[i]:ex[i], :]
                    crop_im = cv2.resize(tmp, (24, 24))
                    crop_im_tensor = convert_image_to_tensor(crop_im)
                    # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
                    cropped_ims_tensors.append(crop_im_tensor)
            except ValueError as e:
                print('dy: {}, edy: {}, dx: {}, edx: {}'.format(dy[i], edy[i], dx[i], edx[i]))
                print('y: {}, ey: {}, x: {}, ex: {}'.format(y[i], ey[i], x[i], ex[i]))
                print(e)

        feed_imgs = torch.stack(cropped_ims_tensors)

        feed_imgs = feed_imgs.to(self.device)

        cls_map, reg = self.rnet_detector(feed_imgs)
        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7)
        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        bw = keep_boxes[:, 2] - keep_boxes[:, 0]
        bh = keep_boxes[:, 3] - keep_boxes[:, 1]
        boxes = np.vstack([keep_boxes[:, 0],
                           keep_boxes[:, 1],
                           keep_boxes[:, 2],
                           keep_boxes[:, 3],
                           keep_cls[:, 0]
                           ])
        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])
        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        h, w, c = im.shape
        if dets is None:
            return None, None

        dets = utils.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                if tmph[i] > 0 and tmpw[i] > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                    tmp[dy[i]:edy[i], dx[i]:edx[i], :] = im[y[i]:ey[i], x[i]:ex[i], :]
                    crop_im = cv2.resize(tmp, (48, 48))
                    crop_im_tensor = convert_image_to_tensor(crop_im)
                    cropped_ims_tensors.append(crop_im_tensor)
            except ValueError as e:
                print(e)

        feed_imgs = torch.stack(cropped_ims_tensors)

        feed_imgs = feed_imgs.to(self.device)

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0]
        bh = keep_boxes[:, 3] - keep_boxes[:, 1]

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])

        boxes_align = boxes_align.T

        landmark = np.vstack([
            align_landmark_topx + keep_landmark[:, 0] * bw,
            align_landmark_topy + keep_landmark[:, 1] * bh,
            align_landmark_topx + keep_landmark[:, 2] * bw,
            align_landmark_topy + keep_landmark[:, 3] * bh,
            align_landmark_topx + keep_landmark[:, 4] * bw,
            align_landmark_topy + keep_landmark[:, 5] * bh,
            align_landmark_topx + keep_landmark[:, 6] * bw,
            align_landmark_topy + keep_landmark[:, 7] * bh,
            align_landmark_topx + keep_landmark[:, 8] * bw,
            align_landmark_topy + keep_landmark[:, 9] * bh,
        ])

        landmark_align = landmark.T

        return boxes_align, landmark_align

    def detect_face(self, img):
        ''' Detect face over image '''
        boxes_align = np.array([])
        landmark_align = np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()

            logger.info(f"Total time cost: {t1+t2+t3:.4f}s, "
                        f"PNet time cost: {t1:.4f}s, "
                        f"RNet time cost: {t2:.4f}s, "
                        f"ONet time cost: {t2:.4f}s. ")


        return boxes_align, landmark_align
