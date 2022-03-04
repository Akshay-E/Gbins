import matplotlib.pyplot as plt 
import os
import sys
import time 
try:
    import cupy as xp
except ImportError:
    import numpy as xp


def time_delay(band_w, centr_f,channels):
    
    f_start=centr_f-band_w/2.0
    f_end=f_start+band_w
    f_arr=np.linspace(f_start, f_end, channels)

    chan_bw=band_w/channels
    chan_centr=f_arr+chan_bw/2.0
#     chan_centr=np.asarray(chan_centr)
    D=4.148808e3
    DM=100#26.7641
    time_delay= D*DM*(chan_centr[0]**-2 - chan_centr**-2)
    print(time_delay)
    
    return(sample_delay(time_delay))
    

def sample_delay(time_delay):
#     psr_period=0.716574
    sampling_time=20.48e-6
    samples_to_shift=time_delay//sampling_time
    print((samples_to_shift))
    
    return(samples_to_shift)
    

def read_block(N):
    
    guppi_loc='/home/arunm77/AMITY_INDIA/arunm77/broadbandinjection/B0329_ae_20.guppi.0000.raw'
    chan_array=np.arange(0,2047)

    with open(guppi_loc,'rb') as file:

        r_skip=file.read((6560+96+134217728)*(N-1)) #skipping 14 blocks

        r_head=file.read(6560+96) #skipping header 
        r2=np.frombuffer(file.read(134217728), dtype=np.int8)
        r2 = np.array(r2, dtype=int)

        channelised=r2.reshape(2048, r2.shape[0]//2048)
        return(channelised)
        
        
def BB_inject_sample_shifting():
    
    sample_shift=time_delay(100,600,2048)
    block=read_block(15)
    
    with open('/home/eakshay/ea/b_band/data/sample_shift.0000.raw','wb') as g:
        with open('/home/eakshay/ea/b_band/guppi_header.txt', 'r') as hdr:
        
            h_info=hdr.read(6560+96)
            g.write(h_info.encode())

            for i in range(2048):

                v=np.roll(channelised[1025], int(sample_shift[i]))
#                 v=np.roll(channelised[1025], -1*int(sample_shift[i])) #artificial DM 1 
                
#             for i in range(2047,-1,-1):
#                 v=np.roll(channelised[1025], int(sample_shift[i]))    #artificial DM 2 
#                 v=np.roll(channelised[1025], -1*int(sample_shift[i]))    #artificial DM 3 
                
#                 plt.plot(v,linewidth=0.5)
#                 plt.show()

                g.write(np.array(v, dtype=np.int8).tobytes())
                

########################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################
import numpy as np

def gaussian(x, mu, std, C):
    denominator = np.sqrt(2*np.pi*std**2)
    numerator = np.exp(-1*(x - mu)**2/(2*std**2))
    return C*(numerator/denominator)

def simulate_pulse(data_points, width):
    x = np.arange(width) #Span of Gaussian Pulse
    gaussian_window1 = gaussian(x, int(width/4), 200, 500)
    gaussian_window2 = gaussian(x, int(3*width/4), 200, 1000)
    gaussian_window = 1+ gaussian_window1 + gaussian_window2
    scale_factors = np.sqrt(2)*gaussian_window
    noise = np.random.normal(0,1,data_points)
    for i in range(width):
        if scale_factors[i] >= 1:
            noise[(int)(data_points/2) + i] *= scale_factors[i]
        else:
            continue
    real = noise
    imag = noise
    final_arr = np.zeros(2*data_points)
    final_arr[::2] = real
    final_arr[1::2] = imag
    ax1 = plt.subplot(211)
    ax1.plot(1+ gaussian_window1 + gaussian_window2)
    ax2 = plt.subplot(212)
    ax2.plot(final_arr)
    plt.show()
    return final_arr


#def overalp_save():
#ARUN_M
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


def compute_tf(fcen, bw, n_coarse, n_fine, dm ):
    
#     D=4.148808e9
    f0=fcen-bw/2.0
    chan_tot= n_coarse*n_fine
    bw_fine=bw/chan_tot
        
    K=2 * xp.pi * 4.148808e9 * dm / f0**2
    fl = xp.linspace(0,bw,chan_tot)
    V=fl**2/(fl+f0)
    
    H= xp.exp(1j*K*V) 
    

def complex_multiply(block):
    
    fft_data = xp.fft.fftshift(xp.fft.fft(block))
    H=trial_tf_kde(600,100,2048,32768,30) #f0 is 550; fl is fine chan start
    
    fft_data_long=xp.repeat(fft_data,2048, 0 )
    fft_long_corrected=H*fft_data_long
    print(fft_long_corrected.shape)
    fft_reshape = fft_long_corrected.reshape(2048,32768)
    return(fft_reshape)

def phase_correction(block):
    
    reshaped=complex_multiply(block)
    with open('/home/eakshay/ea/b_band/data/phase+.0000.raw','wb') as g:
        with open('/home/eakshay/ea/b_band/guppi_header.txt', 'r') as hdr:
        
            h_info=hdr.read(6560+96)
            g.write(h_info.encode())
            ifft_split=xp.empty(65536,float)
            
            for i in range(2047,-1,-1):
            
                ifft=xp.fft.ifft(xp.fft.ifftshift(fft_reshape[i]))
                ifft_split[0::2]=ifft.real
                ifft_split[1::2]=ifft.imag
                    
                g.write(xp.array(ifft_split, dtype=xp.int8).tobytes())
    
        
    
    
    
    
channelised_data=read_block(15)
block = channelised_data[1025][0::2]+1j*channelised_data[1025][1::2]
phase_correction(block)

# complex_multiply(block)
