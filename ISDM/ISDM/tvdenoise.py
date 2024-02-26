import numpy as np
import torch

def tvdenoise(f, lam, iters):
    if lam < 0:
        print('Parameter lambda must be nonnegative.')
        exit(0)

    dt = 0.25
    f=f.cpu().numpy()
    N = f.shape
    id1 = list(range(2, N[0]+1))
    id1.append(N[0])
    iu = list(range(1, N[0]))
    iu.insert(0, 1)
    ir = list(range(2, N[1]+1))
    ir.append(N[1])
    il = list(range(1, N[1]))
    il.insert(0, 1)
    id1 = np.array(id1)
    iu = np.array(iu)
    ir = np.array(ir)
    il = np.array(il)
    p1 = np.zeros(f.shape)
    p2 = np.zeros(f.shape)
    divp = np.zeros(f.shape)
    lastdivp = np.zeros(f.shape)

    if len(N) == 2:  # TV denoising
        # %while norm(divp(: ) - lastdivp(: ), inf) > Tol
        for i in range(iters+1):
            lastdivp = divp
            z = divp - f*lam
            z1 = z[:, ir-1] - z
            z2 = z[id1-1, :] - z
            denom = 1 + dt*np.sqrt(z1**2 + z2**2)
            p1 = (p1 + dt*z1)/denom
            p2 = (p2 + dt*z2)/denom
            divp = p1 - p1[:, il-1] + p2 - p2[iu-1, :]
    elif len(N) == 3:  # % Vectorial TV denoising
        repchannel = np.ones(N[2], 1)

        # %while norm(divp(: ) - lastdivp(: ), inf) > Tol
        for i in range(iters+1):
            lastdivp = divp
            z = divp - f*lam
            z1 = z[:, ir-1, :] - z
            z2 = z[id1-1, :, :] - z
            denom = 1 + dt*np.sqrt(np.sum(z1**2 + z2**2, 3))
            denom = denom[:, :, repchannel-1]
            p1 = (p1 + dt*z1)/denom
            p2 = (p2 + dt*z2)/denom
            divp = p1 - p1[:, il-1, :] + p2 - p2[iu-1, :, :]

    u = f - divp/lam
    u=torch.tensor(u,dtype=torch.float32).cuda()
    return u
