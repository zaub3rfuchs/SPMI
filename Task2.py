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
n=10

#stepsize
h=2/n

#creating 3d Grid
x = np.linspace(-1, 1, n+1)
y = np.linspace(-1, 1, n+1)
z = np.linspace(-1, 1, n+1)
X, Y, Z = np.meshgrid(x, y, z)


#------------------defining reference solution------------------

u_ref1 = lambda x,y,z: np.sin(np.pi*x)* np.sin(np.pi*y)*np.sin(np.pi*z)
u_ref2 = lambda x,y,z: 1/(1+np.sqrt((x**2+y**2+z**2))**2)

uref1=u_ref1(X,Y,Z)
uref2=u_ref2(X,Y,Z)


#------------------creating right-hand-side------------------

#defining right-hand side vector f
f1 = -d*np.pi**2*u_ref1(X,Y,Z)
f2 = ((-2*(uref2**-1))+8*x**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-1))+8*y**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-1))+8*z**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4

#set the boundaries to zero
for i in range(0, 11):
    for j in range(0, 11):
        for k in range(0, 11):
            if(i == 0 or j == 0 or k == 0 or i == 10 or j == 10 or k == 10):         
                f1[i][j][k]=0
                f2[i][j][k]=0

#defining boundary vector g
g1 = np.zeros((n+1, n+1,n+1))
g2 = np.zeros((n+1, n+1,n+1))

#set the boundaries to uref
for i in range(0, 11):
    for j in range(0, 11):
        for k in range(0, 11):
            if(i == 0 or j == 0 or k == 0 or i == 10 or j == 10 or k == 10):
                g1[i][j][k]=u_ref1(x[i],y[j],z[k])
                g2[i][j][k]=u_ref2(x[i],y[j],z[k])

#adding g and f up to right-hand side b
b1=g1+f1
b2=g2+f2


#------------------creating right-hand-side------------------

#defining right-hand side vector f
f1 = -d*np.pi**2*u_ref1(X,Y,Z)
f2 = ((-2*(uref2**-1))+8*x**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-1))+8*y**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4+((-2*(u_ref2(X,Y,Z)**-1))+8*z**2*u_ref2(X,Y,Z)**-1)/u_ref2(X,Y,Z)**-4

#set the boundaries to zero
for i in range(0, 11):
    for j in range(0, 11):
        for k in range(0, 11):
            if(i == 0 or j == 0 or k == 0 or i == n or j == n or k == n):         
                f1[i][j][k]=0
                f2[i][j][k]=0

#defining boundary vector g
g1 = np.zeros((n+1, n+1,n+1))
g2 = np.zeros((n+1, n+1,n+1))

#set the boundaries to uref
for i in range(0, 11):
    for j in range(0, 11):
        for k in range(0, 11):
            if(i == 0 or j == 0 or k == 0 or i == n or j == n or k == n):
                g1[i][j][k]=u_ref1(x[i],y[j],z[k])
                g2[i][j][k]=u_ref2(x[i],y[j],z[k])

#adding g and f up to right-hand side b
b1=g1+f1
b2=g2+f2


#------------------creating laplacian matrix------------------

#defining laplacian matrix
L = np.zeros((n+1, n+1))

for i, v in enumerate((-2,1)):
     np.fill_diagonal(L[:,i:], v)
for i, v in enumerate((1,)):
     np.fill_diagonal(L[-n:,i:], v)
     
L = L

#L[0,:] = 0
#L[-1,:] = 0
#L[0,0] = 1
#L[-1,-1] = 1

#creating Identity matrix for kronecker product
Identity=np.identity(n+1)


#------------------converting to TT-format------------------

#convert right-hand side b to TT-format
b1_tt=tt.vector(b1)
b2_tt=tt.vector(b2)

#convert Identity matrix to TT-format
Identity_tt=tt.matrix(Identity)

#convert laplacian matrix to TT-format
L_tt=tt.matrix(L)


#------------------building 3d laplacian matrix in TT-format with TT-kronecker product------------------

L_tt1 = tt.kron(tt.kron(L_tt,Identity_tt),Identity_tt)
L_tt2 = tt.kron(tt.kron(Identity_tt,L_tt),Identity_tt)
L_tt3 = tt.kron(tt.kron(Identity_tt,Identity_tt),L_tt)
L_tt= L_tt1+L_tt2+L_tt3

#------------------solving higher order linear system in TT-format------------------

#defining initial guess vector
x = tt.ones(11,d)

#solving the higher order linear system with AMEN
print("")
print("Solving Problem 1 with AMEN")
print("")
u1=amen_solve(L_tt,b1_tt,x,1e-6)

time.sleep(0.01)
print("")
print("Solving Problem 2 with AMEN")
print("")
u2=amen_solve(L_tt,b2_tt,x,1e-6)



#------------------calculating RMS-Error for the Solution------------------

RMSE1=np.sqrt(1/n*np.sum(u1.full()-uref1)**2)
RMSE2=np.sqrt(1/n*np.sum(u2.full()-uref2)**2)
time.sleep(0.01)
#print the Error
print("")
print("The RMS-Error for Problem 1 is:",RMSE1)
print("The RMS-Error for Problem 2 is:",RMSE2)


#--------------------------------------------------------------------

# #plotting both functions
# plt.figure()
# plt.plot(grid,u)
# plt.plot(grid,uref)
# plt.show()























     