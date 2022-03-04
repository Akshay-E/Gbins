from matplotlib import pyplot as plt
import numpy as np

# Inputs
x = np.arange(1,101,1)
h = np.arange(1,11,1)
L = 18

y = overlapSave(x,h,L)

# Comparison Plots
plt.subplot(1,2,1)
plt.plot(y,color='r')
plt.title('Overlap Save Method')
plt.subplot(1,2,2)
plt.plot(np.convolve(x,h),color='b')
plt.title('Direct Method')
plt.show()

def overlapSave(x,h,L):
    
    import numpy as np
    import numpy.fft as FFT

    Lx = len(x)
    M = len(h)
    N = L+M-1

    hp = np.append(h,[0]*(L-1))
    X = getBlockMatrix(x,L,M)
    FX = FFT.fft(X,axis=0)
    Fhp = FFT.fft(hp)
    FHp = np.transpose(np.tile(Fhp,(X.shape[1],1)))
    FY = np.multiply(FX,FHp)
    Y_aliased = FFT.ifft(FY,axis=0).real # can discard imaginary because it is of the order of 1e-14
    Y = Y_aliased[np.arange(M-1,N),:]
    y_vector = np.ravel(Y,order='F')
    y = y_vector[0:Lx+M-1]

    return(y)

def getBlockMatrix(x,L,M):

    import numpy as np 
    
    Lx = len(x)
    N = L+M-1
    Ly = Lx+M-1
    xc = np.append(x,[0]*(L+M-1))
    xp = np.insert(xc,0,[0]*(M-1),axis=0)

    if Ly%L != 0:
        ncols = (Ly//L) + 1
    else:
        ncols = (Ly//L)

    X = np.zeros([N,ncols])
    count = 0
    for i in np.arange(0,Ly-1,L):
        X[:,count] = xp[i:i+N]
        count = count + 1
    
    return(X)
