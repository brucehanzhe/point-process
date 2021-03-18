# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:37:36 2020

@author: hanzh
"""

import numpy as np
from scipy.stats import expon
import time
import pandas as pd
import csv
from scipy.optimize import minimize
from multiprocessing import Pool
import os



class simRhawkes(object): 
    
    def __init__(self,p_mu=[2,0.5],p_h=0.5,eta=0.5,p_g=0.1,xi=0.1,a=0.1,b=0.1,cens=50,a1=1,b1=1,c1=2.5):
        self.p_mu=p_mu
        self.p_h=p_h
        self.eta=eta
        self.p_g=p_g
        self.xi=xi
        self.a=a
        self.b=b
        self.cens=cens
        self.a1=a1
        self.b1=b1
        self.c1=c1

#    def Miusea(self,x,mri):
#		return ((x-mri)/self.p_mu[1])**self.p_mu[0]*self.seatrend(x)-((2*self.a1*x/(self.cens**2)-self.b1/self.cens)/((self.p_mu[0]+1)*(self.p_mu[1]**self.p_mu[0])))*((x-mri)**(self.p_mu[0]+1))+((2*self.a1/self.cens**2)/((self.p_mu[0]+1)*(self.p_mu[0]+2)*(self.p_mu[1]**self.p_mu[0]))*(x-mri)**(self.p_mu[0]+2))
    def Miusea(self,x,mri):
        return ((x-mri)/self.p_mu[1])**self.p_mu[0]*self.seatrend(x)-((2*self.a1*x/(self.cens**2)-self.b1/self.cens)/((self.p_mu[0]+1)*(self.p_mu[1]**self.p_mu[0])))*((x-mri)**(self.p_mu[0]+1))+((2*self.a1/self.cens**2)/((self.p_mu[0]+1)*(self.p_mu[0]+2)*(self.p_mu[1]**self.p_mu[0]))*(x-mri)**(self.p_mu[0]+2))
    
    def miu(self,t):
        return (self.p_mu[0]/(self.p_mu[1]**self.p_mu[0]))*(t**(self.p_mu[0]-1))
    
    def Miu(self,t):
        return (t/self.p_mu[1])**self.p_mu[0]
    
    def invMiu(self,x,k,beta):
        return x**(1/k)*beta
        
    def hazard(self,t):
        return expon.pdf(t,scale=self.p_h)
    
    def Hazard(self,t):
        return expon.cdf(t,scale=self.p_h)
       
    def simmrk(self,sc,sp):
        return sc/sp*(np.random.uniform()**-sp-1) 
               
    def gamma(self,x):
        return (1+self.p_g*x)/(1+self.p_g*self.a/(1-self.b-self.xi))
    
    def seatrend(self,x):
        return self.a1*(x/self.cens)**2-self.b1*(x/self.cens)+self.c1  
    
    def g(self,x):
        return np.exp(x*self.p_g)
        
    def phi(self,t,tms):
        return self.eta*sum(self.hazard(t-tms[tms<t]))
    
    def Phi(self,t,tms):
        return self.eta*sum(self.Hazard(t-tms[tms<t]))
        
    def simevents(self):
        ts=quan(np.random.uniform(),lambda x: (x/self.p_mu[1])**self.p_mu[0])
        tms=np.array([ts])
        I=1
        append=np.append

             
        while max(tms)<self.cens:
            ts=ts+quan(np.random.uniform(),lambda x: self.Miusea(ts+x,tms[I-1])-self.Miusea(ts,tms[I-1])+self.eta*sum(self.Hazard(ts+x-tms[tms<(ts+x)]))-self.eta*sum(self.Hazard(ts-tms[tms<ts])))
            tms=append(tms,ts)
            m=self.miu(ts-tms[I-1])*self.seatrend(ts)
            ph=self.eta*sum(self.hazard(ts-tms[tms<ts]))
            lab=np.random.binomial(1,m/(m+ph))
            if lab==1:
                I=len(tms)
        return tms[tms<self.cens]

    def simeventsrej(self):
        ts=0
        tms=np.array([])     
        I=0
        ma=max(self.seatrend(0),self.seatrend(self.cens))
        append=np.append
        sh=self.p_mu[1]
        sc=self.p_mu[0]
        
        while ts<self.cens:
            if ts==0:                
                x1=self.invMiu(self.Miu(ts)+1/ma*np.random.exponential(),sc,sh)
                while np.random.binomial(1,self.seatrend(x1)/ma)!=1:
                    x1=self.invMiu(self.Miu(x1)+1/ma*np.random.exponential(),sc,sh)
                    
            else:
                x1=self.invMiu(self.Miu(ts-tms[I-1])+1/ma*np.random.exponential(),sc,sh)-tms[-1]+tms[I-1]
                if self.seatrend(x1+tms[-1])/ma >1:
                    x1=np.inf
                    pass
                else:
                    while np.random.binomial(1,self.seatrend(x1+tms[-1])/ma)!=1:
                        x1=self.invMiu(self.Miu(x1+tms[-1]-tms[I-1])+1/ma*np.random.exponential(),sc,sh)-tms[-1]+tms[I-1]
                        if self.seatrend(x1+tms[-1])/ma >1:
                            x1=np.inf
                            break
                                                  
            x2=biquan(np.random.exponential(),lambda x:self.eta*sum(self.Hazard(ts+x-tms[tms<(ts+x)]))-self.eta*sum(self.Hazard(ts-tms[tms<ts])))
            if x1<=x2:
                ts+=x1
                tms=append(tms,ts)
                I=len(tms)
            else:
                ts+=x2
                tms=append(tms,ts)
        
        tms=tms[tms<self.cens]

        return tms
    


#%%          
                
def quan(p,func,ub=np.inf):
    lo=up=k=0    

    while 1-np.exp(-func(up))<p:
        up+=2**k
        k=k+1
        
    while abs(lo-up)>np.finfo(float).eps**0.5:
        mid=0.5*(up+lo)
        if 1-np.exp(-func(mid))>=p:
            up=mid
        else:
            lo=mid     
    return 0.5*(lo+up)

def biquan(p,func,ub=np.inf):
    lo=up=k=0     
    if func(ub)<=p:
        return np.inf 
    while func(up)<p:
        up+=2**k
        k=k+1
        
    while abs(lo-up)>np.finfo(float).eps**0.5:
        mid=0.5*(up+lo)
        if func(mid)>=p:
            up=mid
        else:
            lo=mid     
    return 0.5*(lo+up)


def mll(p1,tms,cens):  
 
    p=np.exp(p1)
    
    def Miusea(x,mri):
        return ((x-mri)/p[1])**p[0]*seatrend(x)-(2*p[4]*(x-p[5])/((p[0]+1)*(p[1]**p[0])))*((x-mri)**(p[0]+1))+((2*p[4])/((p[0]+1)*(p[0]+2)*(p[1]**p[0]))*(x-mri)**(p[0]+2))
    
    def miu(t):
        return (p[0]/(p[1]**p[0]))*(t**(p[0]-1))
        
    def seatrend(x):
        return p[4]*(x-p[5])**2+p[6]  
    
    def hazard(t):
        return expon.pdf(t,scale=p[2])
        
    def Hazard(t):
        return expon.cdf(t,scale=p[2]) 
    
    def phi(t,tms):
        return p[3]*sum(hazard((t-tms[tms<t])))
        
    def Phi(t,tms):
        return p[3]*sum(Hazard((t-tms[tms<t])))
    
        
    n=len(tms)
    lden=np.zeros(n+1)
    lcon_den=np.zeros(n)
    lpi=np.zeros(n)
    lpimo=np.zeros(n)
    lden[0]=-Miusea(tms[0],0)+np.log(miu(tms[0]))+np.log(seatrend(tms[0]))
    res=-lden[0]
    i=1
    while i<=n:
        if i==1:
            if i<=n:                    
                lcon_den[0]=-Miusea(tms[1],tms[0])-Phi(tms[1],tms)+np.log(miu(tms[1]-tms[0])*seatrend(tms[1])+phi(tms[1],tms))
            else:
                lcon_den[0]=-Miusea(cens,tms[0])-Phi(cens,tms)
                
        else:
            ph = phi(tms[i-1],tms)   
            m = miu(tms[i-1]-tms[:(i-1)])*seatrend(tms[i-1])
            lpi[:(i-1)] = np.log(ph)-np.log(ph+m)+lcon_den[:(i-1)]+lpimo[:(i-1)]-lden[i-1]
            mx = max(lpimo[:(i-1)]+lcon_den[:(i-1)])
             
            lpi[i-1] = np.log(np.average(m/(ph+m),weights=np.exp(lcon_den[:(i-1)]+lpimo[:(i-1)]-mx)))
            
            if i<=n-1:
                lcon_den[:i]= -Miusea(tms[i],tms[:i])+Miusea(tms[i-1],tms[:i])-Phi(tms[i],tms)+Phi(tms[i-1],tms)+np.log(miu(tms[i]-tms[:i])*seatrend(tms[i])+phi(tms[i],tms))
            else:
                lcon_den[:i] =-Miusea(cens,tms[:i])+Miusea(tms[i-1],tms[:i])-Phi(cens,tms)+Phi(tms[i-1],tms)

        mlcon_den = max(lcon_den[:i])
        mlpi = max(lpi[:i])
        lden[i] = np.log(np.average(np.exp(lcon_den[:i]-mlcon_den),weights=np.exp(lpi[:i]-mlpi)))+mlcon_den
        res =res-lden[i]
        lpimo[:i] = lpi[:i]
        i = i+1
        
    return res

#%%%

def main(n):
    simres={}
    parares={}
    data=[]
#    p0=np.log(np.array([2,0.5,0.5,0.5,0.005,50,0.1],dtype='float64'))  
    

    event=simRhawkes() 
    for i in range(n):        
        simres[i]=event.simevents()
        data.append(pd.DataFrame({'{}'.format(i):simres[i]}))   
    data=pd.concat(data,axis=1)  
    return data

if __name__=="__main__":     
    res=main(2)
    res.to_csv('res100_500.csv')
      

        
 
      
