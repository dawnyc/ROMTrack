import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Lasot(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dir if root is None else root
        super().__init__('LaSOT', root, image_loader)

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
            elif split == "test":
                sequence_list = self._get_sequence_list()
                return sequence_list
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def _get_sequence_list(self):
        sequence_list = ['airplane-1',
                         'airplane-9',
                         'airplane-13',
                         'airplane-15',
                         'basketball-1',
                         'basketball-6',
                         'basketball-7',
                         'basketball-11',
                         'bear-2',
                         'bear-4',
                         'bear-6',
                         'bear-17',
                         'bicycle-2',
                         'bicycle-7',
                         'bicycle-9',
                         'bicycle-18',
                         'bird-2',
                         'bird-3',
                         'bird-15',
                         'bird-17',
                         'boat-3',
                         'boat-4',
                         'boat-12',
                         'boat-17',
                         'book-3',
                         'book-10',
                         'book-11',
                         'book-19',
                         'bottle-1',
                         'bottle-12',
                         'bottle-14',
                         'bottle-18',
                         'bus-2',
                         'bus-5',
                         'bus-17',
                         'bus-19',
                         'car-2',
                         'car-6',
                         'car-9',
                         'car-17',
                         'cat-1',
                         'cat-3',
                         'cat-18',
                         'cat-20',
                         'cattle-2',
                         'cattle-7',
                         'cattle-12',
                         'cattle-13',
                         'spider-14',
                         'spider-16',
                         'spider-18',
                         'spider-20',
                         'coin-3',
                         'coin-6',
                         'coin-7',
                         'coin-18',
                         'crab-3',
                         'crab-6',
                         'crab-12',
                         'crab-18',
                         'surfboard-12',
                         'surfboard-4',
                         'surfboard-5',
                         'surfboard-8',
                         'cup-1',
                         'cup-4',
                         'cup-7',
                         'cup-17',
                         'deer-4',
                         'deer-8',
                         'deer-10',
                         'deer-14',
                         'dog-1',
                         'dog-7',
                         'dog-15',
                         'dog-19',
                         'guitar-3',
                         'guitar-8',
                         'guitar-10',
                         'guitar-16',
                         'person-1',
                         'person-5',
                         'person-10',
                         'person-12',
                         'pig-2',
                         'pig-10',
                         'pig-13',
                         'pig-18',
                         'rubicCube-1',
                         'rubicCube-6',
                         'rubicCube-14',
                         'rubicCube-19',
                         'swing-10',
                         'swing-14',
                         'swing-17',
                         'swing-20',
                         'drone-13',
                         'drone-15',
                         'drone-2',
                         'drone-7',
                         'pool-12',
                         'pool-15',
                         'pool-3',
                         'pool-7',
                         'rabbit-10',
                         'rabbit-13',
                         'rabbit-17',
                         'rabbit-19',
                         'racing-10',
                         'racing-15',
                         'racing-16',
                         'racing-20',
                         'robot-1',
                         'robot-19',
                         'robot-5',
                         'robot-8',
                         'sepia-13',
                         'sepia-16',
                         'sepia-6',
                         'sepia-8',
                         'sheep-3',
                         'sheep-5',
                         'sheep-7',
                         'sheep-9',
                         'skateboard-16',
                         'skateboard-19',
                         'skateboard-3',
                         'skateboard-8',
                         'tank-14',
                         'tank-16',
                         'tank-6',
                         'tank-9',
                         'tiger-12',
                         'tiger-18',
                         'tiger-4',
                         'tiger-6',
                         'train-1',
                         'train-11',
                         'train-20',
                         'train-7',
                         'truck-16',
                         'truck-3',
                         'truck-6',
                         'truck-7',
                         'turtle-16',
                         'turtle-5',
                         'turtle-8',
                         'turtle-9',
                         'umbrella-17',
                         'umbrella-19',
                         'umbrella-2',
                         'umbrella-9',
                         'yoyo-15',
                         'yoyo-17',
                         'yoyo-19',
                         'yoyo-7',
                         'zebra-10',
                         'zebra-14',
                         'zebra-16',
                         'zebra-17',
                         'elephant-1',
                         'elephant-12',
                         'elephant-16',
                         'elephant-18',
                         'goldfish-3',
                         'goldfish-7',
                         'goldfish-8',
                         'goldfish-10',
                         'hat-1',
                         'hat-2',
                         'hat-5',
                         'hat-18',
                         'kite-4',
                         'kite-6',
                         'kite-10',
                         'kite-15',
                         'motorcycle-1',
                         'motorcycle-3',
                         'motorcycle-9',
                         'motorcycle-18',
                         'mouse-1',
                         'mouse-8',
                         'mouse-9',
                         'mouse-17',
                         'flag-3',
                         'flag-9',
                         'flag-5',
                         'flag-2',
                         'frog-3',
                         'frog-4',
                         'frog-20',
                         'frog-9',
                         'gametarget-1',
                         'gametarget-2',
                         'gametarget-7',
                         'gametarget-13',
                         'hand-2',
                         'hand-3',
                         'hand-9',
                         'hand-16',
                         'helmet-5',
                         'helmet-11',
                         'helmet-19',
                         'helmet-13',
                         'licenseplate-6',
                         'licenseplate-12',
                         'licenseplate-13',
                         'licenseplate-15',
                         'electricfan-1',
                         'electricfan-10',
                         'electricfan-18',
                         'electricfan-20',
                         'chameleon-3',
                         'chameleon-6',
                         'chameleon-11',
                         'chameleon-20',
                         'crocodile-3',
                         'crocodile-4',
                         'crocodile-10',
                         'crocodile-14',
                         'gecko-1',
                         'gecko-5',
                         'gecko-16',
                         'gecko-19',
                         'fox-2',
                         'fox-3',
                         'fox-5',
                         'fox-20',
                         'giraffe-2',
                         'giraffe-10',
                         'giraffe-13',
                         'giraffe-15',
                         'gorilla-4',
                         'gorilla-6',
                         'gorilla-9',
                         'gorilla-13',
                         'hippo-1',
                         'hippo-7',
                         'hippo-9',
                         'hippo-20',
                         'horse-1',
                         'horse-4',
                         'horse-12',
                         'horse-15',
                         'kangaroo-2',
                         'kangaroo-5',
                         'kangaroo-11',
                         'kangaroo-14',
                         'leopard-1',
                         'leopard-7',
                         'leopard-16',
                         'leopard-20',
                         'lion-1',
                         'lion-5',
                         'lion-12',
                         'lion-20',
                         'lizard-1',
                         'lizard-3',
                         'lizard-6',
                         'lizard-13',
                         'microphone-2',
                         'microphone-6',
                         'microphone-14',
                         'microphone-16',
                         'monkey-3',
                         'monkey-4',
                         'monkey-9',
                         'monkey-17',
                         'shark-2',
                         'shark-3',
                         'shark-5',
                         'shark-6',
                         'squirrel-8',
                         'squirrel-11',
                         'squirrel-13',
                         'squirrel-19',
                         'volleyball-1',
                         'volleyball-13',
                         'volleyball-18',
                         'volleyball-19']
        return sequence_list
