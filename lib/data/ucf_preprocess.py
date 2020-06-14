from utils.config import cfg

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.temporal_sampling import TemporalSampler
import numpy as np
from copy import copy


class UCF101(Dataset):
    def __init__(self, mode, data_entities, spatial_trans, subset=1):
        self.mode = mode
        self.annotations_path, self.images_path, self.flows_path = data_entities
        self.spatial_trans = spatial_trans

        self.valid_f25 = True if self.mode == 'valid' and cfg.RST.VALID_F25 else False

        self.temporal_sampler = TemporalSampler('f25' if self.valid_f25 else cfg.RST.FRAME_SAMPLING_METHOD)

        with open(os.path.join(self.annotations_path, 'annot0{}.json'.format(cfg.SPLIT_NO))) as fp:
            self.annotations = json.load(fp)
        self.class_labels = self.annotations['labels']
        self.annotations = self.annotations['training' if self.mode == 'train' else 'validation']

        if subset > 1:
            full_length = len(list(self.annotations.keys()))
            self.indices = list(np.array(list(self.annotations.keys()))[np.random.permutation(full_length)[:int(full_length/4)]])  # [:100]
        else:
            self.indices = list(self.annotations.keys())
        if self.mode == 'valid':  # these have inconsistent video size so avoids mini-batching at validation
            for i in ['v_PommelHorse_g05_c01', 'v_PommelHorse_g05_c02',
                      'v_PommelHorse_g05_c03', 'v_PommelHorse_g05_c04']:
                try:
                    self.indices.remove(i)
                except ValueError:
                    continue
        if 'v_LongJump_g18_c03' in self.indices:    # a bug in the provided data set
            self.annotations['v_LongJump_g18_c03']['nframes'] -= 1

        self.images_only, self.flows_only = True, True

    def __getitem__(self, index):
        images = np.load('/home/srip19-pointcloud/linjun/st-net/spatiotemporal-multiplier-networks-pytorch/dataset/preprocess/{}/{}_images.npy'.format(self.mode, index), allow_pickle=True)
        flows = np.load('/home/srip19-pointcloud/linjun/st-net/spatiotemporal-multiplier-networks-pytorch/dataset/preprocess/{}/{}_flows.npy'.format(self.mode, index), allow_pickle=True)
#         i_annotation = np.load('/home/srip19-pointcloud/linjun/st-net/spatiotemporal-multiplier-networks-pytorch/dataset/preprocess/{}/{}_annotation.npy'.format(self.mode, index), allow_pickle=True)
#         print(images, flows, i_annotation)
        key = self.indices[index]
        i_annotation = copy(self.annotations[key])
        nframes = i_annotation['nframes']
        i_annotation['label'] -= 1  # Fix MATLAB indexing for labels
        return images, flows, i_annotation

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def load_images_list(images_list, image_path):
        images = [Image.open(os.path.join(image_path, i)) for i in images_list]
        return images

    @staticmethod
    def load_flows_list(flows_list, flow_path):
        flows = [[Image.open(os.path.join(flow_path, j)) for j in i] for i in flows_list]

        return flows

    @staticmethod
    def pack_frames(images, flows):
        images_o, flows_o = [], []
        if not len(images) == 0:
            images_o = torch.stack(images).transpose(1, 0)
        if not len(flows) == 0:
            flows_o = torch.stack([torch.cat(i) for i in flows]).transpose(1, 0)
        return images_o, flows_o
