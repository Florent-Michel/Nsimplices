#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
import sys
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
from sklearn.decomposition import PCA
#

use_kurtosis=False


""" nSimplex Volume
    Cayley-Menger formula     
"""

def nSimplexVolume(indices,squareDistMat,exactdenominator=True):
    
    n = np.size(indices) - 1
    restrictedD = squareDistMat[:,indices][indices,:]
    CMmat = np.vstack(((n+1)*[1.] , restrictedD))
    CMmat = np.hstack((np.array([[0.]+(n+1)*[1.]]).T , CMmat))
    # NB : missing (-1)**(n+1)  ; but unnecessary here since abs() is taken afterwards
    if exactdenominator:
        denominator = float(2**n *(math.factorial(n))**2)
    else:
        # if calculation of n*Vn/Vn-1 then n*sqrt(denominator_n/denominator_{n-1})
        # simplify to 1/sqrt(2), to take into account in CALLING function.
        denominator = 1.
    VnSquare=np.linalg.det(CMmat**2) # or numpy.linalg.slogdet
    return np.sqrt(abs(VnSquare/denominator))


'''
Draw B groups of (n-1) points to create B nsimplices containing i, to calculate the heights of the point i
'''
def DrawNSimplices(data,N,B,i,n):

    
    hcollection=[]
    countVzero = 0
    for b in range(B):
        indices  = alea.sample( [x for x in range(N) if x != i] , (n-1)+1 )
        Vn       = nSimplexVolume( [i]+indices , data, exactdenominator=False)
        Vnm1     = nSimplexVolume( indices, data, exactdenominator=False)
        if Vnm1!=0:
            hcurrent =  Vn / Vnm1 / np.sqrt(2.) 
            hcollection.append( hcurrent )
        else:
            countVzero += 1
    
    B = B - countVzero
    
    return B,hcollection

""" """
""" Determination of the height of each point   
                                                                  
Iteration on each point of the dataset, and the height is the median of heights of the points in B n-simplices

"""
def nSimplwhichh(N,data,trim,n,seed=1,figpath=os.getcwd(), verbose=False):
    alea.seed(seed)
    h=N*[float('NaN')]
    hs=N*[float('NaN')]
    Jhn=[]
    
    
    # Computation of h_i for each i
    for i in range(N):
        
        B=100
        #we draw B groups of (n-1) points, to create n-Simplices and then compute the height median for i
        (B,hcollection)=DrawNSimplices(data,N,B,i,n)
        
        
        #we here get h[i] the median of heights of the data point i
        h[i] = np.median(hcollection)
        
    
    return h,hs,Jhn



""" 
    Classical multidimensional scaling
""" 

def cMDS(D,alreadyCentered=False):
    """
    D: distance / dissimilarity matrix, square and symetric, diagonal 0
    """
    (p,p2)=np.shape(D)
    if p != p2:
        sys.exit("D must be symetric...")
        
    
    # Double centering
    if not alreadyCentered:
        J=np.eye(p)-np.ones((p,p))/p
        Dcc=-0.5*np.dot(J,np.dot(D**2,J))
    else: 
        # allows robust centering with mediane and mad
        # outside this routine
        Dcc=D
    
    # Eigenvectors
    evals, evecs = np.linalg.eigh(Dcc)
    
    # Sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    #idx = np.argsort(evals)[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]    
    
    # Undelying coordinates 
    idxPos,=np.where(evalst>0)
    Xe=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
    
    return evalst[idxPos], evecst[:,idxPos], Xe

"""
    Detect the outliers in dimension n using the defined criteria, and give the outlier list
"""

def DetectOutliers_n(N,data,trim,cutoff,n,Med):
    
    h,hs,Jhn = nSimplwhichh(N,data,trim,n,seed=n+1)
    
    #Computation of h_i / median(delta_.,i), to detect outliers                                                   
    honmediandist = (h / Med)**2  
    
    list_outliers = list(np.where( honmediandist >cutoff*np.sqrt(2.)/n**2*n/2))[0]
    
    return list_outliers,h,hs,Jhn
    
"""
    Correct the distance matrix, to reduce the outlying-ness
"""

def CorrectDistances(N,distances,list_outliers,n,h):
    cdata=1.0*distances
    for i in list_outliers:
        
        for j in [x for x in range(N) if x!=i]:
            
            cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(cdata[i,j]**2 - h[i]**2))))
            
            
    return cdata


"""
    Other correction, most used:
    
    Correct the coordinates matrix, by projecting the outliers on the subspace
"""

def CorrectProjection(N,Data,list_outliers,rdim):
    
    d=Data.shape[1]
    Data_corr=Data*1.0
    
    #remove outliers for the PCA?
    remove=True
    
    if remove:
        Data_pca=np.delete(Data,list_outliers,0)
    else:
        Data_pca=1.0*Data
    
    pca_method = PCA(n_components=rdim-1)
    method = pca_method.fit_transform(Data_pca) 
    vectors=pca_method.components_
    
    Mean=np.mean(Data_pca,0)
    print(Mean)
    for v in vectors:
        Mean=Mean-np.dot(Mean,v)*v
    
    for i in list_outliers:
        outlier=Data[i]
        sum=0
        projection=np.zeros(d)
        for v in vectors:
            projection+=np.dot(outlier,v)*v
        Data_corr[i,:]=projection+Mean
    
    
    Distances_corr=squareform(pdist(Data_corr))
    #Then, the distances data is prepared for MDS.
    
    return Distances_corr,Data_corr
    

def nSimpl_RelevantDim_ScreePlot(coord,data,cutoff,trim,n0=2,nf=6):
    
    N=np.shape(data)[0]
    dico_outliers   = {}
    dico_h          = {}
    
    stop=False
    
    nb_outliers = np.zeros((nf-n0))
    
    Med=np.median(data,axis=0)
    
    # Iteration on n: determination of the screeplot nb_outliers function of the dimension tested
    for n in range(n0,nf):
        
        if not stop:
            
            #outlier detection
            list_outliers,h,hs,Jhn = DetectOutliers_n(N,data,trim,cutoff,n,Med) 
            dico_outliers[n]=list_outliers
            nb=len(list_outliers)
            nb_outliers[n-n0]=nb
            dico_h[n] = h
            
            print ("Test dim"+str(n)+ ": Il y a "+str(nb)+" outliers : "+str(dico_outliers[n]))
            
            if (nb==0):
                stop=True
            
    
    dimension=np.array(range(n0,nf),dtype=float)
    
    #Determination of the relevant dimension with a 2 piecewise linear function
    
    # def piecewise_linear(x, x0, y0, k1, k2):
    #     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    # 
    # print(dimension)
    # print(nb_outliers)
    # 
    # p , e = optimize.curve_fit(piecewise_linear, dimension-dimension[0], nb_outliers)
    # xd = np.linspace(dimension[0], dimension[-1], 100)
    # 
    # plt.figure()
    # plt.plot(dimension,nb_outliers, "x")
    # plt.plot(xd, piecewise_linear(xd-dimension[0], *p))
    # plt.show()
    # 
    # if (p[1]<1.):
    #     rdim=round(n0+p[0]-1)
    # else:
    #     rdim=round(p[0]+dimension[0])
    # print(rdim)
    # rdim=2
    
    
    #Determination of the relevant dimension with a sigmoid-like function
    
    def sigmoid(x, A, lambd, B,C):
        
        return (A/(1.0+B*np.exp(lambd*x))+C)
    
    print(dimension)
    print(nb_outliers)
    
    p0=[nb_outliers[0],1,0,0]
    
    #fitting the function on the points
    p , e = optimize.curve_fit(sigmoid, dimension, nb_outliers,p0,maxfev=50000)
    xd = np.linspace(dimension[0], dimension[-1], 100)
    
    plt.figure()
    plt.plot(dimension,nb_outliers, "x")
    plt.plot(xd, sigmoid(xd, *p))
    plt.show()
    
    print("A/(1+B*exp(lambda*x))")
    print("A="+str(p[0]))
    print("lambda="+str(p[1]))
    print("B="+str(p[2]))
    print("C="+str(p[3]))
    
    A=p[0]
    lambd=p[1]
    B=p[2]
    
    
    #rdim here is the abscissa of the elbow
    p=0.03
    rdim=int(round(1/lambd*np.log((1-p)/p/B)))
    print(1/lambd*np.log((1-p)/p/B))
    
    #Correction of the distances in the relevant dimension
    
    print("correction")
    cdata=1.0*data
    h=dico_h[rdim]
    cdata=CorrectDistances(N,cdata,dico_outliers[rdim],rdim,h)
    
    #correct by projection on the plan (if plan vectors known)
    print("correction")
    cdata_proj,coord_corr=CorrectProjection(N,coord,dico_outliers[rdim],rdim)
    
    
    return nb_outliers, dico_outliers, dico_h , rdim , cdata , cdata_proj,coord_corr
    

