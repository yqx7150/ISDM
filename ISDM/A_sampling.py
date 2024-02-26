import functools
import torch
import numpy as np
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
import os.path as osp
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from TV_norm import TVnorm
from tvdenoise import tvdenoise
import scipy.io as io

_CORRECTORS = {}
_PREDICTORS = {}

def write_Datax(filedir, dataname, model_num, radio, psnr, ssim, mse):
  if not os.path.exists(filedir):
    os.mkdir(filedir)
  with open(osp.join(filedir, dataname), "a+") as f:  # a+
    f.writelines(str(model_num) + ': ' + str(round(radio, 2)) + ',  '  + str(round(psnr, 4)) + ',  ' + str(
      round(ssim, 4)) + ',  ' + str(round(mse, 4)) )
    f.write('\n')


def write_Data_mx(filedir, dataname, index, model_num, radio, psnr, ssim, mse):
  if not os.path.exists(filedir):
    os.mkdir(filedir)
  with open(osp.join(filedir, dataname), "w+") as f:  # a+
    f.writelines(str(model_num) + '  (' + str(index) + ' ) :  ' + str(round(radio, 2)) + ',  ' + '[' + str(
      round(psnr, 4)) + ',  ' + str(round(ssim, 4)) + ',  ' + str(round(mse, 4)) + ']')
    f.write('\n')


def write_Data_mid(filedir, dataname, model_num, radio, num, psnr, ssim, mse):
  if not os.path.exists(filedir):
    os.mkdir(filedir)
  with open(osp.join(filedir, dataname), "a+") as f:  # a+
    f.writelines(str(model_num) + ': ' + 'radio=' + str(round(radio, 2)) + ',  ' + str(num) + ' :' + '[' + str(
      round(psnr, 4)) + ',  ' + str(round(ssim, 4)) + ',  ' + str(round(mse, 4)) + ']')
    f.write('\n')


def write_images(x, image_save_path):
  x = np.clip(x * 255, 0, 255).astype(np.uint8)
  cv2.imwrite(image_save_path, x)


def save_data(save_filepath, dataname, data_key, save_data):
  if not os.path.exists(save_filepath):
    os.mkdir(save_filepath)
  savemat(osp.join(save_filepath, dataname), {data_key: save_data})


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  sampler_name = config.sampling.method
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)

  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde

    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad1 = score_fn(x1, t) # 5 
      grad2 = score_fn(x2, t) # 5 
      grad3 = score_fn(x3, t) # 5 
      
      noise1 = torch.randn_like(x1) # 4 
      noise2 = torch.randn_like(x2) # 4
      noise3 = torch.randn_like(x3) # 4
      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 ) # 6 
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 # 7
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 # 7
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 # 7
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 # 7
    return x1,x2,x3, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean, t)


def cal_pccs(x, y):
  """
  warning: data format must be narray
  :param x: Variable 1
  :param y: The variable 2
  :return: pccs
  """
  x_mean, y_mean = np.mean(x), np.mean(y)
  return np.sum((x - x_mean) * (y - y_mean)) / np.sqrt(
    np.sum((x - x_mean) * (x - x_mean)) * np.sum((y - y_mean) * (y - y_mean)))


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model,data,mask,data_ob, j):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():

      x_mean = sde.prior_sampling(shape).to(device)
      x1 = x_mean
      x2 = x_mean
      x3 = x_mean
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      psnr_max=0
      ssim_max=0
      pcc_max=0
      psnr_n=[]
      ssim_n=[]
      pcc_n=[]
      for i in range(1000):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        masked = data_ob.permute(2, 0, 1).unsqueeze(0)
        masked_data, std = sde.marginal_prob(masked, vec_t)
        ########################predictor######################
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
        a = 6
        x_red =x_mean[0, 0, :, :] + a*backward(masked[0, 0, :, :], mask[:, :, 0])
        x_green =x_mean[0, 1, :, :] + a*backward(masked[0, 1, :, :], mask[:, :, 1])
        x_blue =x_mean[0, 2, :, :] +  a*backward(masked[0, 2, :, :], mask[:, :, 2])
        x_mean = torch.stack((x_red, x_green, x_blue), dim=0).unsqueeze(0)
        x_red = x_mean[0, 0, :, :] + a * backward(masked[0, 0, :, :], mask[:, :, 0])
        x_green = x_mean[0, 1, :, :] + a * backward(masked[0, 1, :, :], mask[:, :, 1])
        x_blue = x_mean[0, 2, :, :] + a * backward(masked[0, 2, :, :], mask[:, :, 2])
        x_mean = torch.stack((x_red, x_green, x_blue), dim=0).unsqueeze(0)

        ########################corrector#######################
        x1, x2, x3, x_mean = corrector_update_fn(x1, x2, x3, x_mean, vec_t, model=model)
        x_red = x_mean[0, 0, :, :] + a * backward(masked[0, 0, :, :], mask[:, :, 0])
        x_green = x_mean[0, 1, :, :] + a * backward(masked[0, 1, :, :], mask[:, :, 1])
        x_blue = x_mean[0, 2, :, :] + a * backward(masked[0, 2, :, :], mask[:, :, 2])
        x_mean = torch.stack((x_red, x_green, x_blue), dim=0).unsqueeze(0)
        x_red = x_mean[0, 0, :, :] + a * backward(masked_data[0, 0, :, :], mask[:, :, 0])
        x_green = x_mean[0, 1, :, :] + a * backward(masked_data[0, 1, :, :], mask[:, :, 1])
        x_blue = x_mean[0, 2, :, :] + a * backward(masked_data[0, 2, :, :], mask[:, :, 2])
        x = torch.stack((x_red, x_green, x_blue), dim=0).unsqueeze(0)
        x_0 = tvdenoise(x[0, 0, :, :], 10, 2)
        x_1 = tvdenoise(x[0, 1, :, :], 10, 2)
        x_2 = tvdenoise(x[0, 2, :, :], 10, 2)
        x = torch.stack((x_0, x_1, x_2), dim=0).unsqueeze(0)
        x_mean_save_cv=x[0, :, :, :].cpu().numpy().transpose(1,2,0)
        data_ori=data[:,:,:]
        x_mean_save_cv=np.clip(x_mean_save_cv,0,1)
        data_ori=np.clip(data_ori,0,1)

        psnr=compare_psnr(np.abs(x_mean_save_cv)*255,np.abs(data_ori)*255,data_range=255)
        ssim=compare_ssim(np.abs(x_mean_save_cv),np.abs(data_ori),multichannel=True,data_range=1)
        pcc=cal_pccs(np.abs(x_mean_save_cv[:,:,0]),np.abs(data_ori[:,:,0]))
        print(i,'PSNR:',psnr,'SSIM:',ssim,'PCC:',pcc)
      
        if psnr > psnr_max:
          psnr_max = psnr
          ssim_max = ssim
          pcc_max = pcc

          psnr_n.append(psnr_max)
          ssim_n.append(ssim_max)
          pcc_n.append(pcc_max)
          get_image_out(x_mean_save_cv[:, :, 1],j)
        x_mean_save_img = np.stack((x_mean_save_cv[:, :, 2], x_mean_save_cv[:, :, 1], x_mean_save_cv[:, :, 0]), axis=2)
      #print('psnr_n',psnr_n)
      #print('ssim_n',ssim_n)
      return x_mean_save_cv,psnr_n,ssim_n

  return pc_sampler

