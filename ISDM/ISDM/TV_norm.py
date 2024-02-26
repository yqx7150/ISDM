import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt

def diffv(x):
    h = np.array([[0], [1], [-1]])
    sol = conv2c(x, h)

    return sol
def diffh(x):
    h = np.array([0, 1, -1])
    sol = conv2c(x, h)
    return sol

def conv2c(x, h):
    # Circular 2D convolution
    if h.ndim == 1:
        h = np.expand_dims(h, axis=1)
    x = wraparound(x, h)
    y = signal.convolve2d(x, h, mode='valid')
    return y
def wraparound(x, m):

    # % Extend x so as to wrap around on both axes, sufficient to allow a
    # % "valid" convolution with m to return the cyclical convolution.
    # % We assume mask origin near centre of mask for compatibility with
    # % "same" option.
    mx, nx = x.shape
    mm = m.shape[0]
    if m.ndim > 1:
        nm = m.shape[1]
    else:
        nm = 1
    assert mm <= mx and nm <= nx
    mo = int(np.floor((1+mm)/2))
    no = int(np.floor((1+nm)/2))
    # % reflected mask origin
    ml = mo-1
    nl = no-1
    # % mask left/above origin
    mr = mm-mo
    nr = nm-no
    # % mask right/below origin
    me = mx-ml+1
    ne = nx-nl+1
    # % reflected margin in input
    mt = mx+ml
    nt = nx+nl
    # % top of image in output
    my = mx+mm-1
    ny = nx+nm-1
    # % output size
    y = np.zeros((my, ny))
    y[mo-1: mt, no-1:nt] = x
    # % central region
    # 可能所有右界都要+1
    if ml > 0:
        y[0: ml, no-1: nt] = x[me-1: mx, :]
        # % top side
        if nl > 0:
            y[0: ml, 0: nl] = x[me-1: mx, ne-1: nx]
        # % top left corner
        if nr > 0:
            y[0: ml, nt: ny] = x[me-1: mx, 0: nr]
        # % top right corner
    if mr > 0:
        y[mt: my, no-1: nt] = x[0: mr, :]
        # % bottom side
        if nl > 0:
            y[mt: my, 0: nl] = x[0: mr, ne-1: nx]
        # % bottom left corner
        if nr > 0:
            y[mt: my, nt: ny] = x[0: mr, 0: nr]
        # % bottom right corner
    if nl > 0:
        y[mo-1: mt, 0: nl] = x[:, ne-1: nx]
        # % left side
    if nr > 0:
        y[mo-1: mt, nt: ny] = x[:, 0: nr]
        # % right side
    return y

def TVnorm(x):
    x=x.cpu().numpy()
    # Nx, Ny, Nz = x.shape #三维
    Nx, Ny = x.shape
    if x.ndim > 2:
        Nz = x.shape[2]
    else:
        Nz = 1
    x = x.reshape(Nx, Ny*Nz)
    y = np.sum(np.sqrt(diffh(x)**2+diffv(x)**2))
    y=torch.tensor(y,dtype=torch.float32).cuda()
    return y
