#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:49:30 2020

@author: zyntec
"""

import numpy as np
import matplotlib.pyplot as plt
import tt
from tt.amen import amen_solve
import time

#------------------Finite Difference 3D------------------

#defining Borders
a=-1
b=1

#Number of dimensions
d=3


#------------------creating Grid------------------

#Grid length
n=640
#stepsize
h=(b-a)/n

#creating 3d Grid
x = np.linspace(a, b, n+1)
y = np.linspace(a, b, n+1)
z = np.linspace(a, b, n+1)
X, Y, Z = np.meshgrid(x, y, z)


#------------------defining reference solution------------------

u_ref1 = lambda x,y,z: np.sin(np.pi*x)* np.sin(np.pi*y)*np.sin(np.pi*z)
u_ref2 = lambda x,y,z: 1/(1+x**2+y**2+z**2)

uref1=u_ref1(X,Y,Z)
uref2=u_ref2(X,Y,Z)


#------------------creating right-hand-side------------------

#defining right-hand side vector f
f1 = -d*np.pi**2*u_ref1(X,Y,Z)
f2 = ((-2*(uref2**-2))+8*X**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-2))+8*Y**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-2))+8*Z**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4
f2 = 2*(X**2+Y**2+Z**2-3)/((X**2+Y**2+Z**2 +1)**3)
#set the boundaries to zero

#set the yz-plane to zero
f1[0,:,:]=0  
f1[-1,:,:]=0  

f2[0,:,:]=0  
f2[-1,:,:]=0  

#set the xz-plane to zero

f1[:,0,:]=0   
f1[:,-1,:]=0  

f2[:,0,:]=0   
f2[:,-1,:]=0  

#set the xy-plane to zero

f1[:,:,0]=0
f1[:,:,-1]=0

f2[:,:,0]=0
f2[:,:,-1]=0

       

#defining boundary vector g

g1 = u_ref1(X,Y,Z)
g2 = u_ref2(X,Y,Z)
#adding g and f up to right-hand side b

g2_tt = tt.tensor(g2)
f2_tt = tt.tensor(f2)

g1_tt = tt.tensor(g1)
f1_tt = tt.tensor(f1)


#------------------creating laplacian matrix------------------

#defining laplacian matrix
L = np.zeros((n+1, n+1))

for i, v in enumerate((-2,1)):
     np.fill_diagonal(L[:,i:], v)
for i, v in enumerate((1,)):
     np.fill_diagonal(L[-n:,i:], v)
     


L=L/h/h


L[0,:] = 0
L[-1,:] = 0
# L[0,0] = 1
# L[-1,-1] = 1

#creating Identity matrix for kronecker product
Identity=np.identity(n+1)
Identity[0,0] = 0
Identity[-1,-1] = 0

#------------------converting to TT-format------------------

#convert right-hand side b to TT-format


#convert Identity matrix to TT-

Identity_tt=tt.matrix(Identity)

#convert laplacian matrix to TT-format
L_tt=tt.matrix(L)


#------------------building 3d laplacian matrix in TT-format with TT-kronecker product------------------

L_tt1 = tt.kron(tt.kron(L_tt,Identity_tt),Identity_tt)
L_tt2 = tt.kron(tt.kron(Identity_tt,L_tt),Identity_tt)
L_tt3 = tt.kron(tt.kron(Identity_tt,Identity_tt),L_tt)
L_tt= L_tt1+L_tt2+L_tt3
Id2 = np.zeros((n+1, n+1))
Id2[0,0] = 1
Id2[-1,-1] = 1
Lbd_tt = tt.kron(tt.kron(tt.matrix(Id2),tt.eye([n+1])),tt.eye([n+1]))+tt.kron(tt.kron(tt.eye([n+1]),tt.matrix(Id2)),tt.eye([n+1]))+tt.kron(tt.kron(tt.eye([n+1]),tt.eye([n+1])),tt.matrix(Id2))


#righthand-side vector f
# f2_tt=tt.matvec(L_tt,f2_tt)
# Lbd=Lbd_tt.full()
# Ltt = L_tt.full()
# f2_ttf=f2_tt.full()

g2_tt=tt.matvec(Lbd_tt,g2_tt)
g2_ttf=g2_tt.full()
L_tt = L_tt + Lbd_tt
b2_tt = f2_tt + g2_tt
b1_tt = f1_tt + tt.matvec(Lbd_tt,g1_tt)


#------------------solving higher order linear system in TT-format------------------

#defining initial guess vector
x = tt.ones(n+1,d)

#solving the higher order linear system with AMEN
print("")
print("Solving Problem 1 with AMEN")
print("")
u1=amen_solve(L_tt,b1_tt,x,1e-10)

time.sleep(0.01)
print("")
print("Solving Problem 2 with AMEN")
print("")
u2=amen_solve(L_tt,b2_tt,x,1e-10,nswp=40)


#------------------calculating RMS-Error for the Solution------------------



RMSE1=np.sqrt(1/n*    np.sum((u1.full()-uref1)**2)   )
RMSE2=np.sqrt(1/n/n/n*np.sum(u2.full()-uref2)**2)
L2_1 = np.sqrt(np.sum((u1.full()-uref1)**2) / np.sum(u1.full()**2))
L2_2 = np.sqrt(np.sum((u2.full()-uref2)**2) / np.sum(u2.full()**2))
time.sleep(0.01)
#print the Error
print("")
print("The RMS-Error for Problem 1 is:",L2_1)
print("The RMS-Error for Problem 2 is:",L2_2)


#--------------------------------------------------------------------

# #plotting both functions
# plt.figure()
# plt.plot(grid,u)
# plt.plot(grid,uref)
# plt.show()
plt.figure()
plt.imshow(uref1[:,:,2])

plt.figure()
plt.imshow(u1.full()[:,:,2])


plt.figure()
plt.imshow(uref2[:,:,2])

plt.figure()
plt.imshow(u2.full()[:,:,2])

#plt.figure()
#plt.imshow(b2_tt.full()[:,:,2])



