# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:57:41 2021

@author: hanzh
"""

##para=[scale,shape,p_h,eta,a,b,c]
import numpy as np
from scipy.stats import expon
import pandas as pd 
from scipy.optimize import minimize
import csv
import argparse
import pickle
from functools import reduce
from scipy.interpolate import BSpline

def poeBS1(lo,up,alpha,tau,knots,order,coef):
    k=order-1
    res=0
    i=1
    con=1
    lim=np.array([lo,up])
    while i<=k:
        spl=BSpline(knots,coef,order-1)
        con=con/(alpha+i)
#        print(np.diff((lim-tau)**(alpha+i)))
#        print(np.diff((lim-tau)**(alpha+i)*spl(lim,nu=i-1)))
        res=res+(-1)**(i-1)*con*np.diff((lim-tau)**(alpha+i)*spl(lim,nu=i-1))
        i+=1

    con=con/(alpha+i)
    between=np.searchsorted(knots,lim)
    i=between[0]
    while i<between[1]:
        spl=BSpline(knots,coef,order-1)
        res=res+(-1)**k*spl(knots[i-1],nu=k)*con*np.diff((np.array([max(knots[i-1],lo),knots[i]])-tau)**(alpha+k+1))
        i+=1
    res=res+(-1)**k*spl(knots[i-1],nu=k)*con*np.diff((np.array([max(knots[i-1],lo),min(knots[i],up)])-tau)**(alpha+k+1))
    return res

def mllbsp(p,knots,order,tms,cens): 
    alpha=p[0]
    beta=p[1]
    fa=alpha/(beta**alpha)
    gamma_scale=p[2]
    eta=p[3]
    coef=p[4:]
    k=order-1
#    knots=np.linspace(0,cens,8)
    
    def miu(t):
        return (p[0]/(p[1]**p[0]))*(t**(p[0]-1))
        
    def seatrend(t):
        spl=BSpline(knots,coef,order-1)
        return spl(t)
    
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
    lden[0]=-poeBS1(0,tms[0],alpha-1,0,knots,order,coef)*alpha/(beta**alpha)+np.log(miu(tms[0]))+np.log(seatrend(tms[0]))
    res=-lden[0]
    i=1
    while i<=n:
        if i==1:
            if i<=n:                    
                lcon_den[0]=-poeBS1(tms[0],tms[1],alpha-1,tms[0],knots,order,coef)*alpha/(beta**alpha)-Phi(tms[1],tms)+np.log(miu(tms[1]-tms[0])*seatrend(tms[1])+phi(tms[1],tms))
            else:
                lcon_den[0]=-poeBS1(tms[0],cens,alpha-1,tms[0],knots,order,coef)*alpha/(beta**alpha)-Phi(cens,tms)
                
        else:
            ph = phi(tms[i-1],tms)   
            m = miu(tms[i-1]-tms[:(i-1)])*seatrend(tms[i-1])
            lpi[:(i-1)] = np.log(ph)-np.log(ph+m)+lcon_den[:(i-1)]+lpimo[:(i-1)]-lden[i-1]
            mx = max(lpimo[:(i-1)]+lcon_den[:(i-1)])
            
#            if miu(tms[i-1]-tms[:(i-1)]).all()<0:
#                print(m)
#                break
            lpi[i-1] = np.log(np.average(m/(ph+m),weights=np.exp(lcon_den[:(i-1)]+lpimo[:(i-1)]-mx)))
            
            if i<=n-1:
                temp1=-Phi(tms[i],tms)+Phi(tms[i-1],tms)                
                seatemp=seatrend(tms[i])
                for j in range(i):
                    lcon_den[j]= -poeBS1(tms[i-1],tms[i],alpha-1,tms[j],knots,order,coef)*fa+temp1+np.log(miu(tms[i]-tms[j])*seatemp+ph)
            else:
                temp1=-Phi(cens,tms)+Phi(tms[i-1],tms)               
                for j in range(i):
                    lcon_den[j] =-poeBS1(tms[i-1],cens,alpha-1,tms[j],knots,order,coef)*fa+temp1

        mlcon_den = max(lcon_den[:i])
        mlpi = max(lpi[:i])
        lden[i] = np.log(np.average(np.exp(lcon_den[:i]-mlcon_den),weights=np.exp(lpi[:i]-mlpi)))+mlcon_den
        res =res-lden[i]
        lpimo[:i] = lpi[:i]
        i = i+1
        
    return res#lp_a,U,n
p0=np.array([0.5,0.5,0.5,0.5,1,1,1,1],dtype='float64')
print(mllbsp(p0,np.linspace(0,50,8),4,test,50))
#p0=np.log(np.array([2,0.5,0.5,0.5,6,0.5,3],dtype='float64'))
#res3=minimize(mlla,p0,method='BFGS',args=(np.array(d.iloc[:,1].dropna()),50))

#def main(start,n):
#    rawdata=pd.read_csv('res100_500.csv',index_col=0)
#    p=pd.read_csv('para100_500.csv',index_col=0,header=None)
#    data=rawdata.iloc[:,n*start:start+n]
#    para={}    
#    se={}
#    
#    for i in range(n):
#        d=np.array(data.iloc[:,i].dropna())
#        p0=np.log(np.array(p.iloc[i,:],dtype='float64'))
#        res=minimize(mll,p0,method='BFGS',args=(d,100)) 
#        f=open('/home/z5050419/Desktop/result/weibull100s/res/res_{}.pkl'.format(start),'ab')
#        pickle.dump(res,f)
#        f.close()
#        para[start+i]=np.exp(res.x)
#        se[start+i]=np.exp(res.x)*np.sqrt(np.diag(res.hess_inv))        
#        pd.DataFrame(res.hess_inv).to_csv('/home/z5050419/Desktop/result/weibull100s/hess/hess_{}.csv'.format(start),mode='w+',float_format='%.20f')
#    para=pd.DataFrame(para)
#    se=pd.DataFrame(se)
#    para.T.to_csv('para100_500_2.csv',mode='a+',header=0,float_format='%.20f' )
#    se.T.to_csv('se100_500.csv',mode='a+',header=0,float_format='%.20f' )
#    
#  
##    return para,hess
#
#
###parser.add_argument('--end', '-e', help='year 属性，非必要参数，但是有默认值',  required=True)
###parser.add_argument('--body', '-b', help='body 属性，必要参数', required=True)
#
#
#if __name__=='__main__':
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--start', '-s',type=int, required=True)
#    parser.add_argument('--n', '-n', type=int,required=True)
#    args = parser.parse_args()   
#    main(args.start,args.n)
#para,hess=main(0,2)

#    para,hess=main(args.start)
#    para.to_csv('partest.csv','w')
#    file=open('hess_4.pkl','a+')
#    pickle.dump(hess,file)
#    file.close()

#]: files=open('fuck2.csv','ab')
#
#np.savetxt(files,hess[1],delimiter=',')
#
#files.close()

# fw = open("test.txt",'w+')
#fw.write(str(dic))      #把字典转化为str
#fw.close()   
#fr = open("test.txt",'r+')
#dic = eval(fr.read())   #读取的str转换为字典
#print(dic)
#fr.close()
