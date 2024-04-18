import os

import numpy as np
import re

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class TNL2KDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        attributes_path = '{}/{}/attributes.txt'.format(self.base_path, sequence_name)
        attributes_rect = load_text(str(attributes_path), delimiter=',', dtype=np.float64)

        nlp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        nlp_rect = load_text(str(nlp_path), delimiter=',', dtype=str)
        nlp_rect = str(nlp_rect)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), 
                        object_class=None, target_visible=None)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        for seq in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                sequence_list.append(seq)

        return sequence_list
