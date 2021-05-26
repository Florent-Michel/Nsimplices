#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
nSimplices on toy example
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.decomposition import PCA
import time


# current nSimplices library
exec(compile(open(r"nSimplices_new.py", encoding="utf8").read(), "nSimplices_new.py", 'exec'))

# set matplotlib default savefig directory
#os.chdir("/data/user/my-directory/")
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above
### test data


data = pd.read_csv(r'..\Databases\bdd_synthetic_rdim40.csv',sep=';',header=None)
data.head()
#df.sample(frac = 0.1)

### dataset treatment


X=data
X.head()

tab=data

D=pdist(tab.copy())
D_TRUE=squareform(D)

###Add outliers

proportion=0.05
N=tab.shape[0]
k=int(np.ceil(proportion*N))
# random draw of some points, to become outliers
indices=np.sort(alea.sample(range(N),k))
for n in indices:
    horsplan=alea.uniform(-50,50)

    #keep the rdim components, and change one other component
    i=alea.randint(40,48)
    tab.loc[n,i] = horsplan
    print(str(n)+" "+str(i)+" becomes "+ str(horsplan))






###Distances

""" euclidean distances """
N=tab.shape[0]
D=pdist(tab)
DSO=squareform(D)
# """ A few outliers along new axis """
# seed= 112 ; alea.seed(seed)
# sD=squareform(D)
# N=np.shape(sD)[0]
# pc=0.05 #proportion of outliers
# k=int(np.ceil(pc*N))
# sort(alea.sample(range(N),k))
# DSO=1.*sD# Tirage aléatoire de quelques points hors plan
# indices=np.
# for n in indices:
#     horsplan=50*alea.random()
#     print ("n,horsplan:"+str(n)+","+str(horsplan))
#     for m in [x for x in range(N) if x !=n]:
#         DSO[n,m]=DSO[m,n]=np.sqrt(DSO[n,m]**2+horsplan**2)

""" n = 3 , DSO """
# n=3
# Vn=[]
# seed=245124512 ; alea.seed(seed)
# for i in range(1000):
#     indices2=alea.sample(range(N),n+1)
#     Vn.append(nSimplexVolume(indices2,DSO))

#plt.hist(Vn) ; plt.show() # majorité nuls ou presque, un très petit nb ressort
#                          #   (comme ci-dessus, surement dû au bruit).


# ###nSimplices method
# """ Parameters and formatting of input data """
# cutoff=0.5*2*np.sqrt(2)
# trim=0.9
#
# # En entrée : DSO, qui contient quelques outliers hors-plan
# lDSO=squareform(DSO) # shape DSO as other matrices, i.e. as a N*(N-1)/2-sized flat matrix.
# data=squareform(lDSO)    #squareform( DNd ) #squareform(DNne)
# #(D+1.*np.array(Noise)) ou 1e-6,1e-5,... 1., 10. *Noise
# #D, Dd, DO, DOd, DSO, DNd. (Ddne)
#
#
# """ Applications of nSimplices :
#     - dimension detection
#     - outlier detection
#     - outliers are projected into relevant dimension
#     - result : distance matrix to be used in classical MDS, for instance.
# """
# print("\n Application of nSimplex \n ")
# t1=time.time()
# resu=nSimplwhichdim(data,cutoff,trim,ngmetric="rkurtosis",nf=5)
# t2=time.time()
# var = np.array(resu[0][3][0])**2 / (2*np.mean(data,0))
# print (np.std(resu[0][3][0]) , np.std(resu[0][3][0] / np.sqrt(2*np.mean(data,0))), np.std( var ), 1.4826*np.median(abs(var-np.median(var))))
#
#
# """ cMDS and plot """
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))
# #fig.suptitle('Horizontally stacked subplots')
# #
# va, ve, Xe = cMDS(D_TRUE)
# ax1.plot(Xe[:,0],Xe[:,1],'.')
# ax1.set_title("TRUE")
# #
# va, ve, Xe = cMDS(DSO)
# ax2.plot(Xe[:,0],Xe[:,1],'.', color='orange')
# ax2.set_title("Contaminated")
# #
# va, ve, Xe = cMDS(resu[3][3])
#
# ax3.plot(Xe[:,0],Xe[:,1],'.', color='green')
# ax3.set_title("Corrected")
#
# #ax3.legend(targets)
# ax3.grid()
# plt.show()
#
#
# # ax3.plot(Xe[:,0],Xe[:,1],'.', color='green')
# # ax3.set_title("Restituted")
# # plt.show()
#
#
# ### PCA
#
# pca_method = PCA(n_components=2)
# t3=time.time()
# principalComponents_test1 = pca_method.fit_transform(data_test1)
# t4=time.time()
# Df_pc_test1=pd.DataFrame(data = principalComponents_test1
#              , columns = ['principal component 1', 'principal component 2'])
#
# # pca_method.components_
#
# #plot le Df de PCA
#
# Df_pc_test1['type']="normal"
#
# for i in indices:
#
#     Df_pc_test1.loc[i,'type']="contaminated"
#
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# #targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# #colors = ['r', 'g', 'b']
#
#
# targets = ["normal","contaminated"]
# colors = ['b','r']
#
# for target, color in zip(targets,colors):
#     indicesToKeep = Df_pc_test1['type'] == target
#     ax.scatter(Df_pc_test1.loc[indicesToKeep, 'principal component 1']
#                , Df_pc_test1.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()
#
# ###Temps d'exécution
# print(" ")
# T1=t2-t1
# print(f"nSimplex tourne en {T1} s")
# T2=t4-t3
# print(f"PCA tourne en {T2} s")


###ScreePlot Method
cutoff=0.5
trim=0.9
coord=np.array(tab)
T1=time.time()
nb,dico_outlier,dico_h,rdim,cdata,cdata_proj,coord_corr=nSimpl_RelevantDim_ScreePlot(coord,DSO,cutoff,trim,n0=2,nf=49)
T2=time.time()
print(T2-T1)

###Points graphe
inlist=dico_outlier[rdim]
notinlist=[i for i in range(200) if i not in inlist]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))
#fig.suptitle('Horizontally stacked subplots')
#
va, ve, Xe = cMDS(D_TRUE)
ax1.plot(Xe[notinlist,0],Xe[notinlist,1],'.')
ax1.plot(Xe[inlist,0],Xe[inlist,1],'.',color='red')
# ax1.plot(Xe[:,0],Xe[:,1],'.')
ax1.set_title("TRUE")
ax1.grid()
#
va, ve, Xe = cMDS(DSO)
ax2.plot(Xe[notinlist,0],Xe[notinlist,1],'.', color='orange')
ax2.plot(Xe[inlist,0],Xe[inlist,1],'.',color='red')
# ax2.plot(Xe[:,0],Xe[:,1],'.',color='orange')
ax2.set_title("Contaminated")
#
va, ve, Xe = cMDS(cdata_proj)   #cdata_proj)

#Xe=pd.DataFrame(np.asarray([Xe[:,1],Xe[:,2]]).T, columns = ['principal component 1', 'principal component 2'])
ax3.plot(Xe[notinlist,0],Xe[notinlist,1],'.', color='green')
ax3.plot(Xe[inlist,0],Xe[inlist,1],'.',color='red')
# ax3.plot(Xe[:,0],Xe[:,2],'.',color='green')
ax3.set_title("Corrected")
#ax3.legend(targets)
ax3.grid()
plt.show()

###Représentation 3D

ttab=np.array(coord)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
for e in ttab:

    ax.scatter(e[0],e[1],e[2], s = 5)


plt.show()

# ###Correction
#
# a,b=CorrectProjection(200,coord,[ 34,  99, 121, 178],4)
# c=b[66].reshape((20,2))
# d=coord[66].reshape((20,2))
# plt.figure()
# plt.plot(c[:,0],c[:,1],label="corrected")
# plt.plot(d[:,0],d[:,1],label="true cell")
# plt.plot()
# plt.legend()
# # for e in c:
# #     plt.scatter(e[0],e[1],s=5)
#
# plt.show()
