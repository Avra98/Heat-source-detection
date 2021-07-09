import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from scipy.ndimage import convolve
import torch 
import scipy as sp
import scipy.sparse.linalg
from scipy.linalg import circulant
from numpy.linalg import matrix_power
from skimage.transform import radon, rescale
from scipy import ndimage, misc
import imageio
import ipdb
import warnings
import time
warnings.filterwarnings("ignore")

def ADMM_2DTV(Sigma,y,constraint=True,maxit=3000,tol=1e-6,epsilon=1e2,rho=1):
    N = len(y)
    n = int(np.sqrt(N))
    y = y.reshape(N,)
    err = []
    loss = []
    D1 = sp.sparse.diags(np.ones((N,)),0) -sp.sparse.diags(np.ones((N-1,)),-1)
    D1[0,-1] = -1
    D2 = sp.sparse.diags(np.ones((N,)),0) - sp.sparse.diags(np.ones((N-n,)),n) - sp.sparse.diags(np.ones((n,)),n-N) 
    Lambda1 = sp.sparse.diags(scipy.fft.fft(D1[0,:].toarray().reshape(N,)),0)
    Lambda2 = sp.sparse.diags(scipy.fft.fft(D2[0,:].toarray().reshape(N,)),0)
    x = y
    dx = np.zeros(N,)
    dy = np.zeros(N,)
    bx = np.zeros(N,)
    by = np.zeros(N,)
    Fy = sp.fft.fft(y).conjugate()
    d = (2*epsilon*Sigma**2+rho*Lambda1.conjugate()@Lambda1+rho*Lambda2.conjugate()@Lambda2).diagonal()
    invD = sp.sparse.diags(1/d,0)
    for i in range(maxit):
        x = np.real(sp.fft.fft(invD@(2*epsilon*Sigma@Fy+rho*Lambda1.conjugate()@ \
                        sp.fft.fft(dx-bx).conjugate()+rho*Lambda2.conjugate()@sp.fft.fft(dy-by).conjugate())))/N
        if constraint == True:
            x = np.maximum(x,0)
        loss.append(np.linalg.norm(D1@x,1)+epsilon*np.linalg.norm(y-sp.fft.fft(Sigma@sp.fft.fft(x).conjugate())/N)**2)
        dx = (abs(D1@x+bx)>(1./rho))*(abs(D1@x+bx)-1./rho)*np.sign(D1@x+bx)
        dy = (abs(D2@x+by)>(1./rho))*(abs(D2@x+by)-1./rho)*np.sign(D2@x+by)
        bx = bx + D1@x - dx
        by = by + D2@x - dy
        err.append(np.linalg.norm(D1@x-dx)+np.linalg.norm(D2@x-dy))
        if i>50 and np.linalg.norm(D1@x-dx)+np.linalg.norm(D2@x-dy) < tol:
            print('iter',i)
            break
    return x,err,loss

def myADMM_1DTV(Sigma,y,constraint=False,maxit=3000,tol=1e-6,epsilon=1e2,rho=1):
    N = len(y)
    y = y.reshape(N,)
    err = []
    loss = []
    D1 = sp.sparse.diags(np.ones((N,)),0) -sp.sparse.diags(np.ones((N-1,)),-1)
    D1[0,-1] = -1
    Lambda = ssp.sparse.diags(scipy.fft.fft(D1[0,:].toarray().reshape(N,)),0)
    x = np.random.randn(N,)
    z = np.random.randn(N,)
    u = np.random.randn(N,)
    F = sp.fft.fft(np.eye(N))/np.sqrt(N)
    Fy = sp.fft.fft(y).conjugate()
    d = (2*epsilon*Sigma**2+rho*Lambda.conjugate()@Lambda).diagonal()
    invD = sp.sparse.diags(1/d,0)
    for i in range(maxit):
        x = np.real(sp.fft.fft(invD@(2*epsilon*Sigma@Fy+rho*Lambda.conjugate()@sp.fft.fft(z-u).conjugate())))/N
        if constraint == True:
            x = np.maximum(x,0)
        loss.append(np.linalg.norm(D1@x,1)+epsilon*np.linalg.norm(y-sp.fft.fft(Sigma@sp.fft.fft(x).conjugate())/N)**2)
        z = (abs(D1@x+u)>(1./rho))*(abs(D1@x+u)-1./rho)*np.sign(D1@x+u)
        u = u + D1@x - z
        err.append(np.linalg.norm(D1@x-z))
        if i>50 and np.linalg.norm(D1@x-z) < tol:
            print('iter',i)
            break 
    return x,err,loss

def psnr(A,B):
    MSE = np.mean((A-B)**2)**0.5
    return 20*np.log(np.amax(abs(A))/MSE)/np.log(10)

def makeA2d(n=256,alpha=20):
    s = 1/(2*alpha)
    N = n**2
    A = (1-4*s)*sp.sparse.diags(np.ones((N,)),0) + s*sp.sparse.diags(np.ones((N-1,)),-1) + s*sp.sparse.diags(np.ones((N-1,)),1) \
        + s*sp.sparse.diags(np.ones((N-n,)),n) + s*sp.sparse.diags(np.ones((N-n,)),-n) \
        + s*sp.sparse.diags(np.ones((n,)),N-n) + s*sp.sparse.diags(np.ones((n,)),n-N)
    A[0,-1] = s
    A[-1,0] = s
    return A

def inside_ellipse(N, theta, ax1, ax2):
    """
    returns a boolean image marking the pixels inside the given ellipse
    """
    x, y = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N))
    x_rot = torch.cos(theta)*x + torch.sin(theta)*y
    y_rot = -torch.sin(theta)*x + torch.cos(theta)*y
    return x_rot**2 / ax1**2 + y_rot**2 / ax2**2 < 1

def make_im(N, thetas, ax1s, ax2s):
  
    im = torch.zeros(N,N)  

    for ax1, ax2, theta in zip(ax1s, ax2s, thetas):
        im[inside_ellipse(N, theta, ax1, ax2)] += 1
    im = im/im.max()
    return im

def get_ellipse(n=256,image='ellipse',alpha=20,Ncount=1000,filter='Gaussian_local',noise=True):
    N = n**2
    if image == 'ellipse':
        ax1s = [0.2, 0.5]  # solution (anything else is fine)
        ax2s = [0.9, 0.7]  # solution
        thetas = torch.tensor([0.4*math.pi, 0.4*math.pi])  # solution
        im = make_im(n, thetas, ax1s, ax2s).numpy().reshape(n,n).T  # ground truth image
        imres=im.reshape(N,)
    elif image == 'phanton':
        im = imageio.imread('phanton.png')[:,:,0]
        im = rescale(im, scale=n/512, mode='reflect')
    elif image == 'chessboard':
        im1 = imageio.imread('chessboard.jpg')[:,:,0]
        im1 = rescale(im1, scale=n/612, mode='reflect')
        im2=im1[33:256,33:256]
        im = rescale(im2, scale=n/223, mode='reflect')  
    elif image == 'eyechart':
        im1 = imageio.imread('eyechart.jpg')
        im = rescale(im1, scale=n/1280, mode='reflect')
 
    A2d=makeA2d(n,alpha)
    imres=im.reshape(N,)
    Sigma = sp.sparse.diags(np.real(scipy.fft.fft(A2d[0,:].toarray().reshape(N,))))
    mode1 = 'wrap'
    if filter == 'Gaussian':
        mres = np.real(sp.fft.fft((Sigma**Ncount@sp.fft.fft(imres).conjugate()).reshape(N,))/N)
        meas = mres.reshape(n,n)
    elif filter == 'average':
        meas = ndimage.uniform_filter(im, size=20)
    elif filter == 'maximum':
        meas = ndimage.maximum_filter(im,size=20)
    elif filter == 'pyramid':
        size = 10
        P = pyramid(size)
        P = P/np.sum(P)
        meas = convolve(im,P,mode=mode1)
        print('size',size)
    elif filter == 'miscellaneous':
        size = 20
        P = pyramid(size)**0.5
        P = P/np.sum(P)
        meas = convolve(im,P,mode=mode1)
    elif filter == 'Gaussian_local':
        sigma = 2
        size = 40
        x1, y1 = np.meshgrid(np.linspace(-sigma*3,sigma*3,size), np.linspace(-sigma*3,sigma*3,size))
        dst = np.sqrt((x1*x1+y1*y1))
        muu = 0.000
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        gauss = gauss/np.sum(gauss)
        meas = convolve(im,gauss,mode=mode1)
        print('kernel shape',gauss.shape,'sigma',sigma)
    elif filter == 'mixed':
        size = 10
        P = pyramid(size)
        P = P/np.sum(P)
        meas1 = convolve(im,P,mode=mode1)
        
        x1, y1 = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
        dst = np.sqrt((x1*x1+y1*y1))
        sigma = 0.5
        muu = 0.000
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        gauss = gauss/np.sum(gauss)
        meas2 = convolve(im,gauss,mode=mode1)
        
        meas = meas1.copy()
        meas[int(n/2):,:] = meas2[int(n/2):,:] 
    if noise == True:
        meas += 0.05*np.random.randn(meas.shape[0],meas.shape[1])
    return im,meas,Sigma

def pyramid(n):
    a = np.minimum(np.arange(n),np.arange(n)[::-1])
    return np.minimum.outer(a,a)

def obj(lf,meas,Sigma,epsilon1=1e2):
    N = len(meas.reshape(-1))
    n = int(np.sqrt(N))
    D1 = sp.sparse.diags(np.ones((N,)),0) - sp.sparse.diags(np.ones((N-1,)),-1)
    D1[0,-1] = -1
    D2 = sp.sparse.diags(np.ones((N,)),0) - sp.sparse.diags(np.ones((N-n,)),n) - sp.sparse.diags(np.ones((n,)),n-N) 
    u_rec,err,loss = ADMM_2DTV(Sigma**(int(lf)),meas.reshape(N,),constraint=True,rho=2,maxit=300,epsilon=epsilon1)

    mix_norm = np.linalg.norm(((D1@(u_rec)).reshape(N,)**2+(D2@(u_rec)).reshape(N,)**2)**0.5,0.5)**0.5 \
               +5e-4*np.linalg.norm(((D1@(u_rec)).reshape(N,)**2+(D2@(u_rec)).reshape(N,)**2)**0.5,2)**2
    return mix_norm

