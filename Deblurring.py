import utils 
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
from scipy import ndimage, misc
import ipdb
import warnings
import argparse
import time
#%matplotlib inline
warnings.filterwarnings("ignore")

def run_2DTV(args):
    n = args.image_size
    N = n**2
    Ncount = 1000
    if args.image == 'ellipse':
        epsilon_s = 1e2
    elif args.image == 'phanton':
        epsilon_s = 5e1
    elif args.image == 'chessboard':
        epsilon_s = 3e0
    elif args.image == 'eyechart':
        epsilon_s = 3e0
    if args.epsilon:
        epsilon_s = args.epsilon
    #ipdb.set_trace()
    im,meas,Sigma = utils.get_ellipse(n,image=args.image,Ncount=Ncount,noise=args.noise,filter=args.blur,alpha=20)
    print('----------------The true image and measurements----------------')
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(im,vmax=1),plt.title('True image')
    plt.subplot(122),plt.imshow(meas,vmax=1),plt.title('Measurement')
    plt.show()
    print('----------------Searching for N----------------')
    print('rho=2,maxit=300')
    lf=100
    loss1 = utils.obj(lf,meas,Sigma,epsilon1=epsilon_s)
    loss_min = loss1
    N_best = 100
    mydict = {}
    mydict[int(lf)] = 0
    
    print('iter:',0,'N:',lf)
    for j in range(args.maxit):   
        lf2 = max(int(lf+50*np.random.randn()),0)
        loss2 = utils.obj(lf2,meas,Sigma,epsilon1=epsilon_s)  
        if loss2 < loss_min:
            loss_min = loss2
            N_best = int(lf2)
            if mydict.__contains__(int(lf2)):
                mydict[int(lf2)] += 1
            else:
                mydict[int(lf2)] = 1
        else:
            mydict[N_best] += 1
        if mydict[N_best] > 40:
            print('converged at:',N_best)
            break
        if ((loss1/loss2)>np.random.uniform(0.96,1)):
            print('iter',j+1,'previous N:',lf,'Next N:',lf2, 
                  'loss ratio:',loss1/loss2,'Accept:',True,'N_best:',N_best,'count:',mydict[N_best])
            lf=lf2
            loss1 = loss2
        else:
            print('iter',j+1,'previous N:',lf,'Next N:',lf2, 
                  'loss ratio:',loss1/loss2,'Accept:','False','N_best:',N_best,'count:',mydict[N_best])
    
    N_recon = N_best
    if args.N_recon:
        N_recon = args.N_recon
    x,err,loss = utils.ADMM_2DTV(Sigma**(int(N_recon)),meas.reshape(N,),constraint=True,rho=2,epsilon=epsilon_s,maxit=300)
    print('psnr of reconstruction',utils.psnr(im.reshape(N,),x))
    print('psnr of measurement',utils.psnr(im.reshape(N,),meas.reshape(N,)))
    print('------------------reconstruction--------------------')
    plt.figure(figsize=(20,5))
    plt.subplot(141),plt.imshow(x.reshape(n,n),vmax=1),plt.title('x')
    plt.subplot(142),plt.imshow(im,vmax=1),plt.title('u')
    plt.subplot(143),plt.imshow(meas,vmax=1),plt.title('measurement')
    plt.subplot(144),plt.imshow(im-x.reshape(n,n),vmax=1),plt.title('error map')
    plt.show()
    return im,meas,x.reshape(n,n)

def plot_curve(args):
    n = args.image_size
    N = n**2
    Ncount = 1000
    if args.image == 'ellipse':
        epsilon_s = 1e2
    elif args.image == 'phanton':
        epsilon_s = 5e1
    elif args.image == 'chessboard':
        epsilon_s = 3e0
    elif args.image == 'eyechart':
        epsilon_s = 3e0
    if args.epsilon:
        epsilon_s = args.epsilon
    im,meas,Sigma = utils.get_ellipse(n,image=args.image,Ncount=Ncount,filter=args.blur,noise=args.noise)
    print('----------------The true image and measurements----------------')
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(im,vmax=1),plt.title('True image')
    plt.subplot(122),plt.imshow(meas,vmax=1),plt.title('Measurement')
    plt.show()
    print('----------------Searching for N----------------')    
    #t = np.linspace(10,300,20)
    t = np.linspace(600,1400,20)
    a =[]
    b =[]
    D1 = sp.sparse.diags(np.ones((N,)),0) - sp.sparse.diags(np.ones((N-1,)),-1)
    D1[0,-1] = -1
    D2 = sp.sparse.diags(np.ones((N,)),0) - sp.sparse.diags(np.ones((N-n,)),n) - sp.sparse.diags(np.ones((n,)),n-N) 
    print('True300')
    for ti in range(20):
        u_rec,err,loss = utils.ADMM_2DTV(Sigma**(int(t[ti])),meas.reshape(N,),constraint=True,rho=2,maxit=300,epsilon=epsilon_s)
        #mix_norm = (np.linalg.norm(D1@(u_rec).reshape(N,), 0.5)+np.linalg.norm(D2@(u_rec).reshape(N,), 0.5))**0.5 + 1e-4*(np.linalg.norm(D1@(u_rec).reshape(N,), 2))**2
        mix_norm = np.linalg.norm(((D1@(u_rec)).reshape(N,)**2+(D2@(u_rec)).reshape(N,)**2)**0.5,0.5)**0.5 \
            +5e-4*np.linalg.norm(((D1@(u_rec)).reshape(N,)**2+(D2@(u_rec)).reshape(N,)**2)**0.5,2)**2
        psnr_rec = utils.psnr(im.reshape(N,),u_rec)
        a.append(mix_norm)
        b.append(psnr_rec)
        print('N:',t[ti],'psnr',psnr_rec,'mixed norm:',mix_norm)
    return a,b,t

def parse_args():
    parser = argparse.ArgumentParser(description="Run Signal deblurring.")
    parser.add_argument('-task', nargs='?', default='2DTV',
                        help='task: 2DTV or 2DTV')
    parser.add_argument('-image', nargs='?', default='ellipse',
                        help='image: ellipse or phanton or chessboard')
    parser.add_argument('-blur', nargs='?', default='Gaussian_local',
                        help='blur: Gaussian_local or Gaussian or pyramid')
    parser.add_argument('-maxit', type=int, default=150,
                        help='maximum iteration, default is 150')
    parser.add_argument('-image_size', type=int, default=256,
                        help='image size, default is 256')
    parser.add_argument('-epsilon', type=float, default=0,
                        help='Parameter of deblurring. Default is 0.')
    parser.add_argument('-N_recon', type=float, default=0,
                        help='Parameter of deblurring. Default is 0.')
    parser.add_argument('-noise', type=int, default=1,
                        help='Parameter of deblurring. Default is True.')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.task.startswith("1DTV"):
        im,meas,x_rec = run_1DTV(args)
    elif args.task.startswith("2DTV"):
        im,meas,x_rec = run_2DTV(args)
        return im,meas,x_rec    
    if args.task.startswith("plot"):
        a,b,t = plot_curve(args)
        return a,b,t
if __name__ == '__main__':
    #im,meas,x_rec = main()
    a,b,t = main()