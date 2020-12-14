#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:49:40 2020

@author: zyntec
"""

import numpy as np
import matplotlib.pyplot as plt
import tt
from tt.amen import amen_solve
import time
from scipy.linalg import toeplitz

#------------------Finite Difference 3D------------------

#defining Borders
a=-1
b=1

#Number of dimensions
d=3


#------------------creating Grid------------------

#xx= np.arange(anfang, m)
xx = np.array([3,7,15,31,63])
#xx = np.array([31])
errs1=[]
errs2=[]

#yy= np.zeros(m-anfang)

for n in xx:


    #stepsize
    h=(b-a)/n
    print('')
    print('-------------------------------------------------------------------')
    print('')
    print('n =',n,' h =',h)
    #creating 3d Grid
    x = np.linspace(a, b, n+1)
    y = np.linspace(a, b, n+1)
    z = np.linspace(a, b, n+1)
    X, Y, Z = np.meshgrid(x, y, z)
    
    #create qtt-arrray
    qtt_matrix=np.ones(int(np.log2(n+1)*(d-1)))*2
    qtt_tensor=np.ones(int(np.log2(n+1)*(d)))*2
    qtt_matrix=qtt_matrix.astype(int)
    qtt_tensor=qtt_tensor.astype(int)
    #------------------defining reference solution------------------
    
    u_ref1 = lambda x,y,z: np.sin(np.pi*x)* np.sin(np.pi*y)*np.sin(np.pi*z)
    u_ref2 = lambda x,y,z: 1/(1+x**2+y**2+z**2)
    
    uref1=u_ref1(X,Y,Z)
    uref2=u_ref2(X,Y,Z)
    
  
    #------------------creating right-hand-side------------------
    
    #defining right-hand side vector f
    f1 = -d*np.pi**2*u_ref1(X,Y,Z)
    #f2 = ((-2*(uref2**-2))+8*(X)**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-2))+8*Y**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-2))+8*Z**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4
    f2 = 2*(X**2+Y**2+Z**2-3)/((X**2+Y**2+Z**2 +1)**3)

                    
    #defining boundary vector g
    g1 = u_ref1(X,Y,Z)
    g2 = u_ref2(X,Y,Z)


    #------------------creating laplacian matrix------------------
    
    #defining laplacian matrix with toeplitz function, where -2 is the main diagonal and 1 the two on both sides
    L= toeplitz([-2,1,*np.zeros(n-1)])
    
    #set borders to zero
    L[0,:] = 0
    L[-1,:] = 0
    L=L*(1/h/h)
    
    
    #------------------creating L_f matrix------------------
    
    Lf = np.eye(n+1)
    
    #set borders to zero
    Lf[0,:] = 0
    Lf[-1,:] = 0
    
    #devide through dimension to maintain an normalized Lf matrix
    Lf=Lf*(1/d)
    
    #------------------creating Lbd matrix------------------
    
    Lbd = np.eye(n+1) * 0 
    
    #set borders to one to create boundary matrix
    Lbd[0,0] = 1
    Lbd[-1,-1] = 1
    Lbd=Lbd*(1/h/h)
    
    #------------------creating Identity matrix for kronecker product------------------
    
    Identity=np.eye(n+1)
    
    #set 0.0 and -1,-1 to zero for ignoring the borders
    Identity[0,0] = 0
    Identity[-1,-1] = 0
    
    #identity for boundary
    identity_bd=np.eye(n+1)
    
    #------------------reshaping to modesize 2------------------
    
    #reshape
    L_qtt = np.reshape(L,qtt_matrix)
    Lbd_qtt = np.reshape(Lbd,qtt_matrix)
    Lf_qtt = np.reshape(Lf,qtt_matrix)
    f1_qtt = np.reshape(f1,qtt_tensor)
    g1_qtt = np.reshape(g1,qtt_tensor)
    f2_qtt = np.reshape(f2,qtt_tensor)
    g2_qtt = np.reshape(g2,qtt_tensor)
    Identity_qtt = np.reshape(Identity,qtt_matrix)
    Identity_bd_qtt=np.reshape(identity_bd,qtt_matrix)
    

    #------------------converting to TT-format------------------

    # #converting rightside to TT-format
    # #problem 1
    f1_qtt = tt.tensor(f1_qtt)
    g1_qtt = tt.tensor(g1_qtt )
    
    # #problem 1
    f2_qtt = tt.tensor(f2_qtt)
    g2_qtt = tt.tensor(g2_qtt)
    
    # #convert Identity matrix to TT-format
    Identity_qtt = tt.matrix(Identity_qtt) 
    Identity_bd_qtt=tt.matrix(Identity_bd_qtt)
    
    # #convert laplacian matrix to TT-format
    L_qtt = tt.matrix(L_qtt)
    Lbd_qtt = tt.matrix(Lbd_qtt)
    Lf_qtt = tt.matrix(Lf_qtt)
    

    
    
    #------------------building 3d Lbd_tt matrixmatrix in TT-format with TT-kronecker product------------------
    
    Lbd_qtt= tt.kron(tt.kron(Lbd_qtt,Identity_bd_qtt),Identity_bd_qtt)+tt.kron(tt.kron(Identity_bd_qtt,Lbd_qtt),Identity_bd_qtt)+tt.kron(tt.kron(Identity_bd_qtt,Identity_bd_qtt),Lbd_qtt)
        
    
    #------------------building 3d lf matrix in TT-format with TT-kronecker product
    
    Lf_qtt = tt.kron(tt.kron(Lf_qtt,Identity_qtt),Identity_qtt)+tt.kron(tt.kron(Identity_qtt,Lf_qtt),Identity_qtt)+tt.kron(tt.kron(Identity_qtt,Identity_qtt),Lf_qtt)
    

    #------------------building 3d laplacian matrix in TT-format with TT-kronecker product------------------
    
    L_qtt = tt.kron(tt.kron(L_qtt,Identity_qtt),Identity_qtt)+tt.kron(tt.kron(Identity_qtt,L_qtt),Identity_qtt)+tt.kron(tt.kron(Identity_qtt,Identity_qtt),L_qtt)
  
    # #adding boundary matrix to laplacian matrix
    L_qtt = L_qtt+Lbd_qtt
    
    # #matrix-vector multiplication to maintain f and g and sum both to get the complete righside b
    b1_qtt = tt.matvec(Lf_qtt,f1_qtt) + tt.matvec(Lbd_qtt,g1_qtt)
    b2_qtt = tt.matvec(Lf_qtt,f2_qtt) + tt.matvec(Lbd_qtt,g2_qtt)
    

    #------------------solving higher order linear system in TT-format------------------
    
    #defining initial guess vector
    x_qtt= tt.ones(2,int(np.log2(n+1)*(d)))

    #solving the higher order linear system with AMEN
    print("")
    print("Solving Problem 1 with AMEN")
    print("")
    u1_qtt=amen_solve(L_qtt,b1_qtt,x_qtt,1e-10,nswp=100)
    
    time.sleep(0.01)
    print("")
    print("Solving Problem 2 with AMEN")
    print("")
    u2_qtt=amen_solve(L_qtt,b2_qtt,x_qtt,1e-10,nswp=100)
    u2_qtt=u2_qtt.round(1e-10,8)

    #------------------calculating RMS-Error for the Solution------------------
    
    u1_qtt=np.reshape(u1_qtt.full(),(n+1,n+1,n+1))
    u2_qtt=np.reshape(u2_qtt.full(),(n+1,n+1,n+1))

    
    L2_1_qtt = np.sum((u1_qtt-uref1)**2) / n/n/n
    L2_2_qtt = np.sum((u2_qtt-uref2)**2) / n/n/n
   
    time.sleep(0.01)
    #print the Error
    print("")
    print("The RMS-Error for Problem 1 with qtt is:",L2_1_qtt)
    print("The RMS-Error for Problem 2 with qtt is:",L2_2_qtt)
    
    errs1.append(L2_1_qtt)
    errs2.append(L2_2_qtt)
    

#------------------plotting the Solution------------------------------------
   
#create error vector
errs1 = np.array(errs1)    
errs2 = np.array(errs2)  

#loglog plot error
plt.figure()
plt.loglog(xx, errs1, linewidth=2.5)  
plt.loglog(xx, errs2, linewidth=2.5)  
plt.xlabel(r'number of grid points')
plt.ylabel(r'L2 Error')
plt.show() 

#plot reference solution
plt.figure()
plt.imshow(uref2[:,:,2])
 
 
#plot calculated qtt solution    
plt.figure()
plt.imshow(u2_qtt[:,:,2])  
    
    
# plt.figure()
# plt.imshow(b2[:,:,2])

# plt.figure()
# ax = plt.gca()  
# xlog = np.log(xx)
# ylog = np.log(yy)
# plt.loglog(2/xx, yy, linewidth=2.5, color='navy')
# plt.xlabel(r'log(number of grid points)')
# plt.ylabel(r'log(L2 Error)')
# ax.grid(True)
# plt.show()
