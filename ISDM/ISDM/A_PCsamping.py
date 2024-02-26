from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import cv2
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
#import sampling_pc
import A_sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling_3noise_fujian_mask import (ReverseDiffusionPredictor,
                                         LangevinCorrector,
                                         EulerMaruyamaPredictor,
                                         AncestralSamplingPredictor,
                                         NoneCorrector,
                                         NonePredictor,
                                         AnnealedLangevinDynamics)
import datasets
import scipy.io as io
from operator_fza import forward, backward, forward_torch, backward_torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# @title Load the score-based model
sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
    from configs.ve import church_ncsnpp_continuous as configs
    ckpt_filename = "./exp_train_MNIST_max1_N1000/checkpoint_25.pth"
    config = configs.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
batch_size = 1  # 64#@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0  # @param {"type": "integer"}
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)
optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# @title Visualization code

def write_images(x, image_save_path):
  x = np.clip(x * 255, 0, 255).astype(np.uint8)
  cv2.imwrite(image_save_path, x)

# @title PC inpainting
predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16  # @param {"type": "number"}
n_steps = 1  # @param {"type": "integer"}
probability_flow = False  # @param {"type": "boolean"}
pc_inpainter = controllable_generation_fza.get_pc_inpainter(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)
psnr_result = []
ssim_result = []
for j in range(0, 1, 1):

    img = cv2.imread('./Input/sim/mnist.Img.png')


    rec_data = cv2.imread('./Input/sim/mnist.re.png')
    rec_data = rec_data / np.max(rec_data)
    #rec_data = rec_data[383:2431, 183:2231]
    rec_data = torch.tensor(rec_data).cuda()
    #mask_data = cv2.imread('./PSF.png')

    for i in range(1):
        print('##################' + str(i) + '#######################')
        img_size = config.data.image_size
        channels = config.data.num_channels
        shape = (batch_size, channels, img_size, img_size)
        sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                                  inverse_scaler, snr, n_steps=n_steps,
                                                  probability_flow=probability_flow,
                                                  continuous=config.training.continuous,
                                                  eps=sampling_eps, device=config.device)
        x,psnr_n,ssim_n = sampling_fn(score_model,img,mask_data,rec_data, j)
        min_1 = x.min()
        max_1 = x.max()
        x = (x - min_1) / (max_1 - min_1)
        psnr_result = sum(psnr_n) / (len(psnr_n))
        ssim_result = sum(ssim_n) / (len(ssim_n))
        print(psnr_result, ssim_result)
