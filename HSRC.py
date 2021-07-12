import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt
import pickle
import scipy.io as sio
import numpy as np 
from numpy.linalg import matrix_power
import cvxpy as cp
import scipy as sp
import random 
import pywt
import scipy
from scipy.fft import ifft, fft, fftfreq, fftshift
from math import pi, sqrt, exp

def makeA(N=500,alpha=5):
    s=1/(2*alpha)
    A=torch.zeros((N,N))
    for i in range(0,N-1):
        A[i,i]=1-2*s
        A[i,i-1]=s
        A[i,i+1]=s
    A[0,-1]=0
    A[-1,-1]=1-2*s
    A[-1,-2] = s
    Sigma = sp.sparse.diags(np.real(scipy.fft.fft(A.numpy()[0,:].reshape(N,))))
    return A,Sigma

def makeA_periodic(N=500,alpha=5):
    s=1/(2*alpha)
    A = (1-2*s)*sp.sparse.diags(np.ones((N,)),0) +s*sp.sparse.diags(np.ones((N-1,)),-1)+s*sp.sparse.diags(np.ones((N-1,)),1)
    A[0,-1] = s
    A[-1,0] = s
    A = A.toarray()
    A = torch.from_numpy(A)
    Sigma = sp.sparse.diags(np.real(scipy.fft.fft(A.numpy()[0,:].reshape(N,))))
    return A,Sigma

def mygauss(n=30,sigma=1):
    r = np.linspace(-3,3,n)
    mu = 0.
    g = np.exp(-( (r-mu)**2 / ( 2.0 * sigma**2 ) ) )
    #ipdb.set_trace()
    return g
    
def gensource(N=500,spikes=5,lim=40,type='TV'):
    N = 500
    u=torch.zeros((N,1))
    k = np.random.randint(lim+10,N-lim-10,spikes)
    u[k] = 50*torch.randn(spikes,1)
    if type == 'TV':
        u = np.cumsum(u)
    u[:lim] = 0                      
    u[-lim:] = 0
    u=abs(u).reshape(N,)
    return u

def meas(u,blur):
    A,Sigma=makeA(500,5)
    c = np.stack([np.linspace(0,30,30),np.linspace(30,0,30)],axis=1).T.reshape(60,1)[:,0]
    mov_filt=np.power(c,0.5)
    if blur=='gaussian':
        Ncount=1000
        Fw=torch.matrix_power(A, Ncount)
        meas=Fw@u
    elif blur=='linear':
        meas = u.clone()
        for i in range(N-60):
            meas[i+30] = u[i:(i+60)].dot(torch.from_numpy(c).type(dtype=torch.float32))/sum(c)
    elif blur=='non-linear':
        meas = u.clone()
        for i in range(N-60):
            meas[i+30] = u[i:(i+60)].dot(torch.from_numpy(mov_filt).type(dtype=torch.float32))/sum(mov_filt)
    elif blur=='hybrid':
        for i in range(N-60):
            if (i<=int(N/2)):
                meas[i+30] = u[i:(i+60)].dot(torch.from_numpy(c).type(dtype=torch.float32))/sum(c)
            else:
                meas[i+30] = u[i:(i+60)].dot(torch.from_numpy(mov_filt).type(dtype=torch.float32))/sum(mov_filt)
    elif blur=='gaussian_local':
        gauss = mygauss(n=30)
        gauss = gauss/sum(gauss)
        meas = torch.from_numpy(np.convolve(u,gauss,'same'))
    return meas


def l1solv_mix(Fw1,meas,D,u,power=2,epsilon=1e0):
# Create variable.
   x_l1 = cp.Variable(shape=(500,1))
   constraints = [x_l1>=0] #, cp.norm(Fw1.numpy()@x_l1-meas, 2)<=torch.norm(noise)]
# Form objective.
   obj = cp.Minimize(cp.norm(D@x_l1, 1)+ cp.norm((Fw1.numpy()@x_l1)[:,0]-meas, 2)**power*epsilon)
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
   mx_norm = torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 0.5)**0.5+(torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 2))**2*5e-4
   psnr_recon = psnr1(u,x_l1.value.T)
   #print('psnr:{:f}, mixednorm:{:f}'.format(psnr_recon,mx_norm))
   return mx_norm,psnr_recon
   #return x_l1.value

    
def l1solv(Fw1,meas,D,u):
# Create variable.
   x_l1 = cp.Variable(shape=(500,1))
   constraints = [x_l1>=0] #, cp.norm(Fw1.numpy()@x_l1-meas, 2)<=torch.norm(noise)]
# Form objective.
   obj = cp.Minimize(cp.norm(D@x_l1, 1)+ cp.norm((Fw1.numpy()@x_l1)[:,0]-meas, 2)*1e1)
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
   mx_norm = torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 0.5)**0.5+(torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 2))**2*5e-4
   psnr_recon = psnr1(u,x_l1.value.T)
   #print('psnr:{:f}, mixednorm:{:f}'.format(psnr_recon,mx_norm))
   return x_l1.value    
   
def l1solv_sparse(Fw1,meas,u):
# Create variable.
   x_l1 = cp.Variable(shape=(500,1))
   constraints = [x_l1>=0] #, cp.norm(Fw1.numpy()@x_l1-meas, 2)<=torch.norm(noise)]
# Form objective.
   obj = cp.Minimize(cp.norm(x_l1, 1)+ cp.norm((Fw1.numpy()@x_l1)[:,0]-meas, 2)**2*1e1)
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
   mx_norm = torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 0.5)**0.5+(torch.norm(torch.from_numpy(D@x_l1.value[:,0]), 2))**2*5e-4
   psnr_recon = psnr1(u,x_l1.value.T)
   #print('psnr:{:f}, mixednorm:{:f}'.format(psnr_recon,mx_norm))
   return x_l1.value    

def myADMM(Sigma,y,constrain=True,maxit=3000,tol=1e-6,epsilon=1e0,rho=1):
  y = y.numpy()
  N = len(y)
  D1 = sp.sparse.diags(np.ones((N,)),0) -sp.sparse.diags(np.ones((N-1,)),-1)
  D1[0,-1] = -1
  y = y.reshape(N,)
  err = []
  loss = []
  a = np.zeros(N,)
  a[0] = 1
  a[-1] = -1
  Lambda = sp.sparse.diags(scipy.fft.fft(a),0)
  x = np.zeros(N,)
  z = np.zeros(N,)
  u = np.zeros(N,)
  F = sp.fft.fft(np.eye(N))/np.sqrt(N)
  Fy = sp.fft.fft(y).conjugate()
  d = (2*epsilon*Sigma**2+rho*Lambda.conjugate()@Lambda).diagonal()
  invD = sp.sparse.diags(1/d,0)
  for i in range(maxit):
      x = np.real(sp.fft.fft(invD@(2*epsilon*Sigma@Fy+rho*Lambda.conjugate()@sp.fft.fft(z-u).conjugate())))/N
      if constrain == True:
        x = np.maximum(x,0)
      loss.append(np.linalg.norm(D1@x,1)+epsilon*np.linalg.norm(y-sp.fft.fft(Sigma@sp.fft.fft(x).conjugate())/N)**2)
      z = (abs(D1@x+u)>(1./rho))*(abs(D1@x+u)-1./rho)*np.sign(D1@x+u)
      u = u + D1@x - z
      if i>50 and np.linalg.norm(D1@x-z) < tol:
        break
  return x

def myADMM_sparse(Sigma,y,constrain=True,maxit=1000,tol=1e-12,epsilon=1e5,rho=0.05):
  N = len(y)
  y = y.numpy().reshape(N,)
  err = []
  loss = []
  x = np.zeros(N,)
  z = np.zeros(N,)
  u = np.zeros(N,)
  Fy = sp.fft.fft(y).conjugate()
  d = (2*epsilon*Sigma**2+rho*sp.sparse.eye(N)).diagonal()
  invD = sp.sparse.diags(1/d,0)
  for i in range(maxit):
      x = np.real(sp.fft.fft(invD@(2*epsilon*Sigma@Fy+rho*sp.fft.fft(z-u).conjugate())))/N
      if constrain == True:
          z = ((x+u)>(1./rho))*(x+u-1./rho)
      else:
          z = (abs(x+u)>(1./rho))*(abs(x+u)-1./rho)*np.sign(x+u)
      u = u + x - z
      if i>100 and np.linalg.norm(x-z) < tol:
        break
  return x
  
def getx1(Sigma,meas,u,D):
    t = np.linspace(100,2400,20)
    a1 =[]
    b1 =[]
    for ti in range(20):
        z = myADMM(Sigma**(int(t[ti])),meas,constrain=True,maxit=300,tol=1e-6,epsilon=1e0,rho=1)
        mix_norm = torch.norm(torch.from_numpy(D@z), 0.5)**0.5+(torch.norm(torch.from_numpy(D@z), 2))**2*5e-4
        #psnr_rec = psnr1(u,torch.from_numpy(z))
        a1.append(mix_norm)
        #b1.append(psnr_rec)
        #print('N',t[ti],'psnr:',psnr_rec)
    N_recon = np.round(t[a1==np.min(a1)])
    x1 = torch.from_numpy(myADMM(Sigma**int(N_recon),meas,constrain=True,maxit=3000,tol=1e-6,epsilon=1e0,rho=1))
    return psnr1(u,x1.T)  

def getx_simpleTV(meas,u):
    b = []
    t = [5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e0,5,1e1]
    for i in range(len(t)):
        z = simple_TV(meas,constrain=True,maxit=3000,tol=1e-6,epsilon=t[i],rho=1)
        #mix_norm = torch.norm(torch.from_numpy(D@z), 0.5)**0.5+(torch.norm(torch.from_numpy(D@z), 2))**2*5e-4
        psnr_rec = psnr1(u,torch.from_numpy(z))
        b.append(psnr_rec)
        print('t',t[i],'psnr',psnr_rec)
    return np.max(b)
    
def getx_sparse(Sigma,meas,u):
    t = np.linspace(100,2400,20)
    a1 =[]
    b1 =[]
    for ti in range(20):
        z = myADMM_sparse(Sigma**(int(t[ti])),meas,constrain=True,maxit=300,tol=1e-6,epsilon=1e0,rho=1)
        mix_norm = torch.norm(torch.from_numpy(z), 0.5)**0.5+(torch.norm(torch.from_numpy(z), 2))**2*5e-4
        psnr_rec = psnr1(u,torch.from_numpy(z))
        a1.append(mix_norm)
        #b1.append(psnr_rec)
        print('N',t[ti],'psnr:',psnr_rec)
    N_recon = np.round(t[a1==np.min(a1)])
    print(N_recon)
    x1 = torch.from_numpy(myADMM_sparse(Sigma**int(N_recon),meas,constrain=True,maxit=3000,tol=1e-6,epsilon=1e0,rho=1))
    return psnr1(u,x1.T)  
    
def psnr1(A,B):
    MSE = torch.mean((A-B)**2)**0.5
    return 20*np.log(torch.max(torch.abs(A))/MSE)/np.log(10)    
        
def getx(A,meas,u,D,power1,epsilon1):
    t1 = np.linspace(100,2400,20)
    a1 =[]
    b1 =[]
    for ti in range(20):
        mix_norm, psnr_rec = l1solv_mix(torch.matrix_power(A, int(t1[ti])),meas,D,u,power=power1,episilon=epsilon1)
        a1.append(mix_norm)
        b1.append(psnr_rec)
        print('psnr:',psnr_rec)
    N_recon = np.round(t1[a1==np.min(a1)])
    x1 = l1solv(torch.matrix_power(A, int(N_recon)),meas,D,u)
    return psnr1(u,x1.T)

def simple_TV(y,constrain=True,maxit=3000,tol=1e-6,epsilon=1e-1,rho=1):
  N = len(y)
  y = y.numpy().reshape(N,)
  D1 = sp.sparse.diags(np.ones((N,)),0) -sp.sparse.diags(np.ones((N-1,)),-1)
  D1[0,-1] = -1
  err = []
  loss = []
  a = np.zeros(N,)
  a[0] = 1
  a[-1] = -1
  Lambda = sp.sparse.diags(scipy.fft.fft(a),0)
  Sigma = sp.sparse.diags(np.ones((N,)),0)
  x = np.zeros(N,)
  z = np.zeros(N,)
  u = np.zeros(N,)
  F = sp.fft.fft(np.eye(N))/np.sqrt(N)
  Fy = sp.fft.fft(y).conjugate()
  d = (2*epsilon*Sigma**2+rho*Lambda.conjugate()@Lambda).diagonal()
  invD = sp.sparse.diags(1/d,0)
  for i in range(maxit):
      x = np.real(sp.fft.fft(invD@(2*epsilon*Sigma@Fy+rho*Lambda.conjugate()@sp.fft.fft(z-u).conjugate())))/N
      if constrain == True:
        x = np.maximum(x,0)
      loss.append(np.linalg.norm(D1@x,1)+epsilon*np.linalg.norm(y-sp.fft.fft(Sigma@sp.fft.fft(x).conjugate())/N)**2)
      z = (abs(D1@x+u)>(1./rho))*(abs(D1@x+u)-1./rho)*np.sign(D1@x+u)
      u = u + D1@x - z
      if i>50 and np.linalg.norm(D1@x-z) < tol:
        break
  return x 

def exp(trails):
    N = 500
    spikes = np.arange(2,20,2)
    store = np.zeros((trails,spikes.size))
    D = np.diag(np.ones((N,)),0) - np.diag(np.ones((N-1,)),-1)
    for i in range(len(spikes)):
        for j in range (trails):
            A,Sigma = makeA_periodic(500,5)
            u = gensource(N=500,spikes=spikes[i],lim=40)
            m = meas(u,blur='gaussian_local')
            #store[j,i] = getx(A,m,u,D)
            store[j,i] = getx1(Sigma,m,u,D)
            print('spikes',spikes[i],'trial:',j,'psnr',store[j,i])
    return store
    
def exp_sparse(trails):
    N = 500
    spikes = np.arange(2,20,2)
    store = np.zeros((trails,spikes.size))
    for i in range(len(spikes)):
        for j in range (trails):
            A,Sigma = makeA_periodic(500,5)
            u = gensource(N=500,spikes=spikes[i],lim=40,type='sparse')
            m = meas(u,blur='gaussian_local')
            #store[j,i] = getx(A,m,u,D)
            store[j,i] = getx_sparse(Sigma,m,u)
            print('spikes',spikes[i],'trial:',j,'psnr',store[j,i])
    return store


            
    
            
        
            
        
        
    
        
        
        
    
    
    
    
    

