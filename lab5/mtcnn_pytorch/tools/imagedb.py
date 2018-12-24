import os
import numpy as np
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageDB(object):
    def __init__(self, image_annotation_file, prefix_path='', mode='train'):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.mode = mode

    def load_image_set_index(self):
        ''' Get image index 
        
        Returns:
            image_set_index: str, relative path of image
        '''
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index

    def load_imdb(self):
        ''' Get and save ground truth image database 
        
        Returns:
            gt_imdb: dict, image database with annotations
        '''

        gt_imdb = self.load_annotations()

        return gt_imdb

    def real_image_path(self, index):
        ''' Given image's relative index, return full path of image '''

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file = index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(
            image_file), 'Path does not exist: {}'.format(image_file)

        return image_file

    def load_annotations(self, annotation_type=1):
        ''' Load annotations
        
        what's the meaning of annotation_type ? I don't know !
        Returns:
            imdb: dict, image database with annotations
        '''

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            im_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((4,))
                imdb_['landmark_target'] = np.zeros((10,))
                if len(annotation[2:]) == 4:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                if len(annotation[2:]) == 14:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                    landmark = annotation[6:]
                    imdb_['landmark_target'] = np.array(landmark).astype(float)
            imdb.append(imdb_)
        return imdb

    def append_flipped_images(self, imdb):
        ''' append flipped images to imdb 
        
        Returns:
            imdb: dict, image database with flipped image annotations
        '''
        print('append flipped images to imdb ', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            landmark_ = imdb_['landmark_target'].copy()
            landmark_ = landmark_.reshape((5, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]

            item = {'image': imdb_['image'],
                    'label': imdb_['label'],
                    'bbox_target': m_bbox,
                    'landmark_target': landmark_.reshape((10)),
                    'flipped': True}

            imdb.append(item)
        self.image_set_index *= 2
        print('after flipped images appended to imdb ', len(imdb))

        return imdb


class FaceDataset(data.Dataset):
    def __init__(self, image_annotation_file, prefix_path='', transform=None, is_train=False):
        self.image_annotation_file = image_annotation_file
        self.prefix_path = prefix_path
        self.is_train = is_train
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.gt_imdb = self.load_annotations()
        if self.is_train:
            self.gt_imdb = self.append_flipped_images(self.gt_imdb)
        self.transform = transform
        self.loader = pil_loader

    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index

    def real_image_path(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file = index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(
            image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def load_annotations(self):
        """Load annotations

        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            im_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_['image'] = im_path

            if not self.is_train:
                # gt_boxes = map(float, annotation[1:])
                # boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
                # imdb_['gt_boxes'] = boxes
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((4,))
                imdb_['landmark_target'] = np.zeros((10,))
                if len(annotation[2:]) == 4:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                if len(annotation[2:]) == 14:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                    landmark = annotation[6:]
                    imdb_['landmark_target'] = np.array(landmark).astype(float)
            imdb.append(imdb_)
        return imdb

    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('append flipped images to imdb', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            landmark_ = imdb_['landmark_target'].copy()
            landmark_ = landmark_.reshape((5, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]

            item = {'image': imdb_['image'],
                    'label': imdb_['label'],
                    'bbox_target': m_bbox,
                    'landmark_target': landmark_.reshape((10)),
                    'flipped': True}

            imdb.append(item)
        self.image_set_index *= 2
        return imdb

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        imdb_ = self.gt_imdb[idx]
        image = self.loader(imdb_['image'])
        labels = {}
        labels['label'] = imdb_['label']
        labels['bbox_target'] = imdb_['bbox_target']
        labels['landmark_target'] = imdb_['landmark_target']

        if self.transform:
            image = self.transform(image)

        return image, labels
