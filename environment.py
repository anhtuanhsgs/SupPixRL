import os, sys, glob, time, copy
from os import sys, path
import cv2, random
import numpy as np
from collections import deque

from skimage.measure import label
from skimage.morphology import disk
from skimage.morphology import binary_dilation
import skimage.io as io

from sklearn.metrics import adjusted_rand_score
from skimage.transform import resize as resize3D
from Utils.utils import *
from Utils.img_aug_func import *
import albumentations as A

import matplotlib.pyplot as plt
from random import shuffle
from PIL import Image, ImageFilter
from utils import guassian_weight_map, density_map, malis_rand_index, malis_f1_score, adjusted_rand_index
from skimage.draw import line_aa
from misc.Voronoi import *
import time
from rewards import split_reward_fn, merge_reward_fn
from super_pixel import RAG

BG = 1 # Background label
DEBUG_T = 2 # Initialize coloring for the first DEBUG_T steps

class General_env ():
    def init (self, config):
        print (config)

        self.T = config ['T'] # Number of steps
        self.size = config ["size"]
        self.base = config ['base']
        self.rng = np.random.RandomState(time_seed ())
        self.max_lbl = config ['base'] ** (self .T) - 1
        self.pred_lbl2rgb = color_generator (self.max_lbl + 1)
        self.gt_lbl2rgb = color_generator (111)
        self.is3D = self.config ["3D"]

    def seed (self, seed):
        self.rng = np.random.RandomState(seed)

    def aug (self, image, mask):

        if self.is3D:
            if not (self.size[1] == self.size[2] == self.size[0]):
                [image, mask] = FlipRev3D ([image, mask], self.rng)
                rotn = self.rng.randint (4)
                [image, mask] = [rotate3D (img, rotn) for img in [image, mask]]
            else:
                [image, mask] = RotFlipRev3D ([image, mask], self.rng)
            ret = {"image": image, "mask": mask}
            return ret ['image'], ret ['mask']
        
        if self.config ["data"] == "zebrafish":
            randomBrightness =  A.RandomBrightness (p=0.3, limit=0.1)
            RandomContrast = A.RandomContrast (p=0.1, limit=0.1)
        else:
            randomBrightness = A.RandomBrightness (p=0.7, limit=0.1)
            RandomContrast = A.RandomContrast (p=0.5, limit=0.1)

        if image.shape [-1] == 3:
            if self.config ["data"] in ["Cityscape", "kitti"]:
                aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                     
                            A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.7),
                        A.ShiftScaleRotate (p=0.7, shift_limit=0.2, rotate_limit=10, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.4, 0.4), border_mode=cv2.BORDER_CONSTANT),
                        A.RandomBrightness (p=0.7, limit=0.5),
                        A.RandomContrast (p=0.5),
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.5, blur_limit=4),
                        ]
                    )
            else:
                aug = A.Compose([
                            A.HorizontalFlip (p=0.5),
                            A.VerticalFlip(p=0.5),              
                            A.RandomRotate90(p=0.5),
                            A.Transpose (p=0.5),
                            A.OneOf([
                                A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                                A.GridDistortion(p=0.9, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                                A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                                ], p=0.7),
                            A.ShiftScaleRotate (p=0.7, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.3, 0.5), border_mode=cv2.BORDER_CONSTANT),
                            A.CLAHE(p=0.3),
                            A.RandomBrightness (p=0.7, limit=0.5),
                            A.RandomContrast (p=0.5),
                            A.GaussNoise (p=0.5),
                            A.Blur (p=0.5, blur_limit=4),
                            ]
                        )
        else:
            aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.VerticalFlip(p=0.5),              
                        A.RandomRotate90(p=0.5),
                        A.Transpose (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.5, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                            A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                            A.OpticalDistortion(p=0.5, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.6),
                        A.ShiftScaleRotate (p=0.5, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT),
                        # A.CLAHE(p=0.3),
                        randomBrightness,
                        RandomContrast,
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.3, blur_limit=4),
                        ]
                    )
        if self.config ["DEBUG"] or self.config ["no_aug"]:
            aug = A.Compose ([])

        ret = aug (image=image, mask=mask)        

        return ret ['image'], ret ['mask']

    def step_inference (self, action):
        action = np.concatenate ([0], action)
        self.action = action
        nd_action = self.rag.vec2img (action)
        
        self.new_lbl = self.lbl + action * (self.base ** self.step_cnt)
        self.lbl = self.new_lbl
        done = False
        info = {}
        reward = np.zeros (self.size, dtype=np.float32)

        
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * nd_action / (self.base - 1) - 1) * 255
        self.step_cnt += 1

        if self.step_cnt >= self.T:
            done = True

        ret = (self.observation (), reward, done, info)
        return ret

    def step (self, action):
        # action should be a 1D list of size n_segments [0-Base]
        # make action list to have n_segments + 1 (label start from 1)
        action = np.concatenate (([0], action), axis=0)
        self.action = action
        nd_action = self.rag.vec2img (action)

        self.new_lbl = self.lbl + action * (self.base ** self.step_cnt)
        done = False

        # Normalize to [-255, 255]: (2 * x - max_x) - 1 * 255
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * nd_action / (self.base - 1) - 1) * 255
        info = {}

        # Initialize the reward map
        reward = np.zeros (self.n_segments + 1, dtype=np.float32)

        #TODO: add reward for background
        # reward += self.background_reward (False)
        
        split_reward = np.zeros (self.n_segments + 1, dtype=np.float32)
        merge_reward = np.zeros (self.n_segments + 1, dtype=np.float32)

        merge_ratio = np.zeros (self.n_segments + 1, dtype=np.float32)
        split_ratio = np.zeros (self.n_segments + 1, dtype=np.float32)

        # Max value, only used to normlize visualization iamge of split/merge ratios
        range_split = 1.0 * 2 * self.config ["spl_w"] 
        range_merge = 1.0 * 2 * self.config ["mer_w"] 


        # Update split rewards
        split_reward += split_reward_fn (self.lbl, self.new_lbl, self.rag, self.step_cnt, self.T)

        # Update merge rewards
        merge_reward += merge_reward_fn (self.lbl, self.new_lbl, self.rag, self.step_cnt, self.T)

        reward += self.config ["spl_w"] * split_reward + self.config ["mer_w"] * merge_reward #+ split_reward * merge_reward`
        merge_ratio += merge_reward / range_merge 
        split_ratio += split_reward / range_split

        self.split_ratio_sum = self.split_ratio_sum + split_ratio
        self.merge_ratio_sum = self.merge_ratio_sum + merge_ratio

        self.lbl = self.new_lbl
        self.step_cnt += 1
        
        #Reward
        self.rewards.append (reward)    
        self.sum_reward += reward
        if self.step_cnt >= self.T:
            done = True
        ret = (self.observation (), reward, done, info)
        return ret

    def random_init_lbl (self):
        # Randomly initialize DEBUG_T steps
        if (DEBUG_T == 0):
            return
        for t in range (DEBUG_T):
            action = np.zeros (self.n_segments, dtype=np.int32)
            for ins_segments in self.rag.instances:
                act = self.rng.randint (self.base)
                for seg_idx in ins_segments:
                    action [seg_idx-1] = act

            self.step (action)


    def reset_end (self):
        """
            Call after custom reset func for initialization of edges and graph
        """
        self.rag = RAG (self.config, self.raw, self.gt_lbl, self.config['n_segments'], compactness=0.1, split_r=self.config['split_r'])
        # Segment index count from 1   
        self.n_segments = self.rag.n_segments 
        # Prediction base-10 label (for visisualize and final result) [n_segments]
        self.lbl = np.zeros (self.n_segments + 1, dtype=np.int32)

        # Predict base-k label mask T x H x W
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.rewards = []

        # For visualization of rewards matrix for merge and split, normlized to be mid value 128
        self.split_ratio_sum = (np.zeros (self.n_segments + 1, dtype=np.float32) + 0.5)
        self.merge_ratio_sum = (np.zeros (self.n_segments + 1, dtype=np.float32) + 0.5)

        # Accumulate reward map
        self.sum_reward = np.zeros (self.n_segments + 1, dtype=np.float32)
        self.random_init_lbl ()

    def observation (self):
        # Observation matrix of shape CxHxW or CxHxWxD
        # C = T + len ([raw, supix_mask])

        # Map to nD image then normalize the base-10 predicted label to [0-1]
        lbl = self.rag.vec2img (self.lbl) / self.max_lbl * 255.0

        #TODO: use a generic data definition
        # Add [raw image, over-segmentation boundary map, coloring mask] to the observation
        if self.config ["data_chan"] == 1:
            obs = [self.raw [None].astype (np.float32)]
        elif self.config ["data_chan"] == 3:
            obs = [np.transpose (self.raw.astype (np.float32), [2, 0, 1])]

        obs.append (np.transpose (self.rag.boundary_map, [2, 0, 1]) * 255.0)
        obs.append (self.mask)

        obs = np.concatenate (obs, 0)

        ''' Range of obs elemnents:
            raw image: [0,1]
            oversegmentation boundary map: [0, 1]
            coloring mask: [-1, 1]
        '''
        return obs / 255.0

    def visualize (self):
        # Return a visualization of the states and rewards of size H'xW'x3 [0-255]

        # Get the data to a temporary variable
        if self.is3D:
            # If using 3D volume data -> visualize the middle slice
            index = len (self.raw) // 2
            tmp_raw = self.raw [index]
            tmp_lbl = self.lbl [index]
            tmp_gt_lbl = self.gt_lbl [index]
        else:
            tmp_raw = self.raw
            tmp_lbl = self.lbl
            tmp_gt_lbl = self.gt_lbl

        #TODO: Define a generic/abstract datatype (3d/4d)
        #Convert to RGB image HxWx3
        if self.config ["data_chan"] == 1:
            raw = np.repeat (np.expand_dims (tmp_raw, -1), 3, -1).astype (np.uint8)
        elif self.config ["data_chan"] == 3:
            raw = tmp_raw

        # Map to nD image then convert the base-10 predicted label to RGB
        lbl = self.rag.vec2img (tmp_lbl.astype (np.int32))
        lbl = self.pred_lbl2rgb (lbl)

        # Convert base-10 groundtruth label to RGB
        gt_lbl = tmp_gt_lbl % 111
        gt_lbl += ((gt_lbl == 0) & (tmp_gt_lbl != 0))
        gt_lbl = self.gt_lbl2rgb (gt_lbl)
        
        # Convert base-k label mask to RGB, stacking horizontal
        masks = []
        for i in range (self.T):
            if self.is3D:
                mask_i = self.mask [i][index]
            else:
                mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        #TODO: Define concrete max reward function
        max_reward = 7

        # Normalize reward stack to [0-1] and convert reward map to RGB [HxW]     
        rewards = []

        # Stacking horizontal, add a sum of all reward in the front (T+1 HxW maps)
        for reward_i in [self.sum_reward] + self.rewards:
            # Map to nD image
            reward_i = self.rag.vec2img (reward_i)
            if self.is3D:
                reward_i = reward_i [index]
            reward_i = ((reward_i + max_reward) / (2 * max_reward) * 255).astype (np.uint8) 

            reward_i = np.repeat (np.expand_dims (reward_i, -1), 3, -1)
            rewards.append (reward_i)

        # Append the undecided reward information (0s valued) Hx(T+1)Wx3
        while (len (rewards) < self.T + 1):
            rewards.append (np.zeros_like (rewards [0]))

        # Convert the split/merge ratio effectiveness of decisions to RGB
        split_ratio_sum = self.rag.vec2img (self.split_ratio_sum)
        merge_ratio_sum = self.rag.vec2img (self.merge_ratio_sum)
        if self.is3D:
            split_ratio_sum = np.repeat (np.expand_dims ((self.split_ratio_sum [index] * 255).astype (np.uint8), -1), 3, -1)
            merge_ratio_sum = np.repeat (np.expand_dims ((self.merge_ratio_sum [index] * 255).astype (np.uint8), -1), 3, -1)
        else:
            split_ratio_sum = np.repeat (np.expand_dims ((self.split_ratio_sum * 255).astype (np.uint8), -1), 3, -1)
            merge_ratio_sum = np.repeat (np.expand_dims ((self.merge_ratio_sum * 255).astype (np.uint8), -1), 3, -1)

        line1 = [raw, lbl, gt_lbl,] + masks # Raw image, label, groundtruth, decision maps [Hx(T+3)Wx3]

        # Append merge/split ratio to begining of the rewards map list [Hx(T+3)Wx3]
        while (len (rewards) < len (line1)):
            rewards = [np.zeros_like (rewards [-1])] + rewards

        rewards[0] = self.rag.vec2img (split_ratio_sum)
        rewards[1] = self.rag.vec2img (merge_ratio_sum)

        line1 = np.concatenate (line1, 1)
        line2 = np.concatenate (rewards, 1)

        # Final RGB image  [2Hx(T+3)Wx3]
        ret = np.concatenate ([line1, line2], 0) 
        return ret

class EM_env (General_env):
    def __init__ (self, config, raw_list, gt_lbl_list=None, isTrain=True, seed=0):
        self.isTrain = isTrain
        self.raw_list = raw_list
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(seed)
        self.config = config
        self.init (config)

    def random_crop (self, size, imgs):
        y0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
        x0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
        ret = []
        if self.is3D:
            z0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
            y0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
            x0 = self.rng.randint (imgs[0].shape[2] - size[2] + 1)

            for img in imgs:
                ret += [img[z0:z0+size[0], y0:y0+size[1], x0:x0+size[2]]]
        else:
            for img in imgs:
                ret += [img[y0:y0+size[0], x0:x0+size[1]]]
        return ret

    def reset (self, model=None, gpu_id=0):
        self.step_cnt = 0

        # Randomly choose a train sample
        image_idx = self.rng.randint (0, len (self.raw_list))
        self.raw = np.array (self.raw_list [image_idx], copy=True)

        if (self.isTrain):
            self.gt_lbl = np.copy(self.gt_lbl_list [image_idx])
        else:
            self.gt_lbl = np.zeros_like (self.raw)

        self.reset_end ()
        return self.observation ()

    def set_sample (self, idx, resize=False):
        # Preparation for inference of a single image (no augmentation)
        self.step_cnt = 0
        idx = idx
        if not self.is3D:
            while (self.raw_list [idx].shape [0] < self.size [0] \
                or self.raw_list [idx].shape [1] < self.size [1]):
                idx = self.rng.randint (len (self.raw_list))
        else:
            while (self.raw_list [idx].shape [0] < self.size [0] \
                or self.raw_list [idx].shape [1] < self.size [1] \
                or self.raw_list [idx].shape [2] < self.size [2]):
                idx = self.rng.randint (len (self.raw_list))

        self.raw = np.array (self.raw_list [idx], copy=True)
        if self.gt_lbl_list is not None:
            self.gt_lbl = np.array (self.gt_lbl_list [idx], copy=True)
        else:
            self.gt_lbl = np.zeros (self.size, dtype=np.int32)

        if (not resize):
            if self.gt_lbl_list is not None:
                self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
            else:
                self.raw = self.random_crop (self.size, [self.raw]) [0]
        else:
            self.raw = cv2.resize (self.raw, (self.size [1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            self.gt_lbl = cv2.resize (self.gt_lbl.astype (np.int32), (self.size [1], self.size [0]), interpolation=cv2.INTER_NEAREST)

        self.split_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)
        self.merge_ratio_sum = (np.zeros (self.size, dtype=np.float32) + 0.5) * (self.gt_lbl > 0)

        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []

        self.reset_end ()
        return self.observation ()
