# ray import 
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.data as ray_data 

# import MLP related libraries 
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os


# loading preprocessed data from s3 
s3_bucket = "s3://your-bucket-name/"
dataset_key = "data.zip"
dataset_path = ray_data.from_s3(s3_bucket, dataset_key)
   
dataset = ray_data.load(dataset_path)  


def train_func():
    pass

scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
trainer = TorchTrainer(train_func, scaling_config=scaling_config) # launch a distributed training job 
result = trainer.fit()
