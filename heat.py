import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt
import pickle
import scipy.io as sio
import numpy as np 
from numpy.linalg import matrix_power
import cvxpy as cp
import random 
import pywt
import scipy
from scipy.fft import ifft, fft, fftfreq, fftshift



def l1solv(Fw1,meas,norm):
# Create variable.
   x_l1 = cp.Variable(shape=(meas.shape[0],1))
   constraints = [x_l1>=0]
# Form objective.
   obj = cp.Minimize(cp.norm(x_l1, norm)+cp.norm(Fw1.numpy()@x_l1-meas, 2)**2*1e3)
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   #y = pywt.threshold(x_l1.value[:,0], 4, 'hard')
   #x2 = torch.from_numpy(y)
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
   return torch.norm(x2,0.5)**0.5 + 5e-4*torch.norm(x2,2)**2
   #return x2


def l1solv_TV(net1,Fw1,meas):
    # Create variable.
    x_l1 = cp.Variable(shape=(meas.shape[0],1))
    constraints = [x_l1>=0] #, cp.norm(Fw1.numpy()@x_l1-meas, 2)<=torch.norm(noise)]
   # Form objective.
    obj = cp.Minimize(cp.norm(D@x_l1, 1) + cp.norm((Fw1.numpy()@x_l1)[:,0]-meas, 2)**2*5e4)
   # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
    plt.plot(x_l1.value)
    plt.plot(u)
    return net1(x2/np.linalg.norm(x2))#,torch.norm((Fw1@x2).reshape(500,1)-meas)*1e-10*2


def l1solvnaive_TV(net1,meas):
   # Create variable.
   N = meas.shape[0]
   x_l1 = cp.Variable(shape=(N,1))
   constraints = [x_l1>=0] #, cp.norm(Fw1.numpy()@x_l1-meas, 2)<=torch.norm(noise)]
   # Form objective.
   obj = cp.Minimize(cp.norm(D@x_l1, 1) + cp.norm(x_l1[:,0]-meas, 2)**2*3e-3)
   # Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
    
   plt.plot(x_l1.value)
   plt.plot(u)
   #  return torch.norm(x2,0.1)*1e-20#+torch.norm((Fw1@x2).reshape(500,1)-meas)*1e-2
   return net1(x2/np.linalg.norm(x2))


def l1rec(Fw1,meas,norm):
# Create variable.
   x_l1 = cp.Variable(shape=(meas.shape[0],1))
   constraints = [x_l1>=0]
# Form objective.
   obj = cp.Minimize(cp.norm(x_l1, norm)+cp.norm(Fw1.numpy()@x_l1-meas, 2)**2*1e3)
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value[:,0]).cpu().type(dtype=torch.float32)
   return x2


def slope(N,A,meas):
    F_Nr=l1solv_TV(net1,torch.matrix_power(A, int(N+5)),meas)
    F_Nl=l1solv_TV(net1,torch.matrix_power(A, int(N-5)),meas)
    return (F_Nr-F_Nl)/10        


def search2(lf=100,rb=2500,walk=100,iter=100):
    meas, measn, u, A,Fw = gensingledata(n=500,alpha=20,m=5,Ncount=1000,sigma=1)
    for j in range(iter):
        if j==10:
            walk = 50
        slf =slope(lf,A,measn)
        srb = slope(rb,A,measn)
        if slf<0 and srb>0:
            lf=lf+walk
            rb=rb-walk
        elif slf<0 and srb<0:
            rb=rb+walk
        elif slf>0 and srb>0:
            lf=lf-walk  
        elif slf>0 and srb<0:
            lf=lf-walk
            rb=rb+walk    
        print(lf,rb,slf,srb)
            
        
        
def search3(lf=100,walk=50,iter=200):
    meas, measn, u, A,Fw = gensingledata(n=500,alpha=20,m=5,Ncount=1000,sigma=1)
    for j in range(iter):
        if (lf>900):
            walk = 10
        slf =slope(lf,A,meas)
        if slf<0:
            print(lf,slf)
            lf=lf+walk
        elif slf>0:
            print('Solution has been reached')  
            break
        
        
        
        

def genData(n=500,alpha=20,k=5,N=1000,Ncount=1000,sigma=1):
    source = torch.zeros(n,N)
    wsource = torch.zeros(n,N)
    A=makeA1d(n,alpha)
    Fw = torch.matrix_power(A, Ncount)
    for i in range(0,N):
        Unit=torch.zeros(n,1)
        Unit[torch.randint(0,n-2,(k,))+1,0]=1000*torch.rand(k)
        source[:,i]=Unit.T
        source[:,i] = source[:,i]/torch.norm(source[:,i])
        meas = Fw @ Unit
        measn= meas+sigma*torch.rand(meas.shape)
        offset = torch.randint(-900,900,(1,))
        wsource[:,i] = l1rec(torch.matrix_power(A, Ncount+int(offset)),measn,2)
        wsource[:,i] = wsource[:,i]/torch.norm(wsource[:,i])
    return source, wsource    

def makeA1d(n=500,alpha=20):
    A=torch.zeros((n,n))
    s=torch.zeros((n,1))
    s[0:int(n/2)]=1/(2*alpha)
    s[int(n/2):n-1]=1/(2*alpha)
    for i in range(0,n-1):
        A[i,i]=1-2*s[i]
        A[i,i-1]=s[i]
        A[i,i+1]=s[i]
    A[n-1,n-1]=1-2*s[i]
    A[n-1,n-2]=s[i]
    A[0,n-1]=0
    return A
    

def makeA2d(n=10,alpha=20):
    s=torch.zeros((n,n))
    s[0:n-1,0:n-1]=1/(2*alpha)
    sx=torch.reshape(s,(n**2,1))
    A=torch.zeros((n**2,n**2))
    sy=sx
    ##interior points
    for i in range(1,(n**2)-2):
        A[i,i]=(1-2*sx[i]-2*sy[i])
        A[i,i-1]=sx[i]
        A[i,i+1]=sx[i]
        A[i,((i+n)%(n**2))]=sy[i]
        A[i,((i-n)%(n**2))]=sy[i]
    
    ##x-axis boundaries
    A[0:n-1,:]=0
    A[:,0:n-1]=0
    A[((n**2)-n):((n**2)-1),:]=0
    A[:,((n**2)-n):((n**2)-1)]=0
        
    ##y-axis boundaries
    for i in range(1,n):
        A[(i*n)-1,:]=0
        A[(i*n)-2,:]=0
    return A


 
    
def gensingledata(n=500,alpha=20,m=5,Ncount=1000,sigma=0.1):
    #torch.random.seed()
    A=makeA1d(n,alpha)
    u=torch.zeros((n,1))
    #np.random.seed(0)
    k = np.random.randint(0,n-75,m)
    #torch.manual_seed(0)
    u[k+50] = 1000*torch.rand(m,1)
    #u[253]=759
    Fw=torch.matrix_power(A, Ncount)
    meas=Fw@u
    measn= meas+sigma*torch.rand(meas.shape)
    return meas, measn,u,A,Fw


def gensingledata_TV(n=500,alpha=5,m=10,Ncount=1000,sigma=2):
    A=makeA1d(n,alpha)
    u=torch.zeros((n,1))
    k = np.random.randint(0,N-50,m)
    u[k+25] = 50*torch.randn(m,1)
    u = np.cumsum(u)
    if np.linalg.norm(u[u<0])<np.linalg.norm(u[u>0]):
        u[u<0]=0
    else:
        u[u>0] =0 
        u = -u
    Fw=torch.matrix_power(A, Ncount)
    meas=Fw@u    
    noise = sigma*torch.randn(meas.shape)*1
    measn= meas+noise
    return meas, measn,u,A,Fw



def gensingledata2d(n=50,alpha=20,m=5,Ncount=5000,sigma=0.1):
    A2d=makeA2d(n,alpha)
    u=torch.zeros(n,n)
    np.random.seed(0)
    l = [(random.randrange(10,n-10), random.randrange(10,n-10)) for i in range(m)]
    for j in l:
        u[j]=1000*torch.rand(1)
    ures=torch.reshape(u,(n**2,1))
    Fw=torch.matrix_power(A2d, Ncount)
    mres=Fw@ures
    meas=torch.reshape(mres,(n,n))
    measn=meas + sigma*torch.rand(meas.shape)
    #x=l1rec(Fw,mres,1)
    #rec = torch.reshape(x,(n,n))
    return u,ures,meas,mres,A2d
        
    
    ###random walk search algorithm when the ratio of old norm to the new norm is more than one it always takes the accepted step. 
    ##But when the ratio is less than one, it decides to take a step at the outcome of a uniform proboality distribution output 
    
def searchalgo(meas,A,lf=100,iter=150):
    #meas, measn, u, A,Fw = gensingledata(n=500,alpha=20,m=15,Ncount=1000,sigma=1)
    for j in range(iter):   
        Fl=l1solv(torch.matrix_power(A, lf),meas,2)
        lf2=int(lf+50*np.random.randn())
        Fl2=l1solv(torch.matrix_power(A, lf2),meas,2)
        print(lf,lf2,Fl/Fl2)
        if ((Fl/Fl2)>np.random.uniform(0.999,1)):
            lf=lf2
        else:
            lf=lf
    return lf  

            
            
            
def plotnormvsN(meas,A,lw=100,rb=1500,points=150):
    N1 = torch.linspace(lw,rb,points)
    F=torch.zeros_like(N1)
    #meas, measn, u, A,Fw= gensingledata(n=500,alpha=20,m=10,Ncount=1000,sigma=1)
    #plt.plot(u)
    #plt.plot(meas)
    #plt.show()
    #print(torch.norm(meas,1)/torch.norm(u,1))
    for i in range(0,points):
        F[i]= l1solv(torch.matrix_power(A, int(N1[i])),meas,2)
    plt.plot(N1.detach().numpy(),F.detach().numpy())
    plt.xlabel('Ncount')
    plt.ylabel('Defined norm')
    #plt.title(' network norm')
    #plt.savefig('figures/network_norm_N10000.svg', format='svg', dpi=1200)
    return F,N1


def perreco(noise=False,sigma=0.04):
   meas, measn, u, A,Fw = gensingledata(n=500,alpha=20,m=5,Ncount=500,sigma=0.04)
   x_l1 = cp.Variable(shape=(meas.shape[0],1))
   if (noise==False): 
       constraints = [Fw.numpy()@x_l1==meas]
   else:
       eps=5     
       constraints = [cp.norm(Fw.numpy()@x_l1-measn, 1) <=eps ]     
   obj = cp.Minimize(cp.norm(x_l1, 1))
# Form and solve problem.
   prob = cp.Problem(obj, constraints)
   prob.solve()
   x2 = torch.from_numpy(x_l1.value).cpu().type(dtype=torch.float32)
    
   plt.plot(x2,label='Reconl1')
   print(torch.norm(x2,1))
   plt.plot(u,label='Original')
   print(torch.norm(u,1))
   plt.legend() 
   return u,x2 