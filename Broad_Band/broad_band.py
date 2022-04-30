import matplotlib.pyplot as plt 
import os
import sys
import time 
import shutil 

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np

from setigen.voltage import raw_utils


class broadband(object):
    
    def __init__(self, 
                 input_file_stem,
                 pulse_time,
                 dm=10, 
                 width=1000, 
                 snr=100         
                ):
    
        self.raw_params = raw_utils.get_raw_params(input_file_stem=input_file_stem)
        self.obs_bw= raw_params['num_chans']*np.abs(raw_params['chan_bw'])
        
        self.smear=self._smear_() 
        self.input_file_stem=input_file_stem
        self.dm=dm
        self.width=width
        self.snr=snr
        self.D=4.148808e3
        self.pulse_time=pulse_time
        
        assert self.pulse_time<self.raw_params['obs_length'], 'injection time exceeds length of file'
        assert self.smear<self.raw_params['obs_length'], 'smearing across band exceeds length of file'
        
        
        self.blocks_per_file = raw_utils.get_blocks_per_file(input_file_stem)
        self.time_per_block = self.raw_params['block_size'] / (self.raw_params['num_antennas'] * self.raw_params['num_chans'] *                               self.raw_params['num_bits']) * self.raw_params['tbin']
        

        shutil.copyfile(self.input_file_stem, f'{self.input_file_stem}_dispersed.0000.raw')
        
        self.blocks_to_read=np.ceil(self.smear/self.time_per_block)
        self.start_block=np.floor(self.pulse_time/self.time_per_block)

        header=raw_utils.read_header(self.input_file_stem)
        self.header_size= int(512 * np.ceil((80 * (len(header) + 1)) / 512))
        self.data_size= int(self.raw_params['block_size'])
        self.chan_size=int(data_size/self.raw_params['num_chans'])
        self.block_read_size =  header_size + data_size
        
        slef.sim_len=blocks_to_read * data_size
        
        

#dispersion by sample shifting 

    def chan_time_delay(self,x):

        f_chan_arr= self.raw_params['fch1']+xp.linspace(0, self.obs_bw, self.raw_params['num_chans'] )
        chan_centr=f_chan_arr+self.raw_params['chan_bw']/2.0
        
        time_delay= self.D*self.DM*(chan_centr[0]**-x - chan_centr**-x)
        samples_to_shift=time_delay//self.raw_params['tbin']

        return(samples_to_shift)


    def sample_shift(self,x=2 ):

        sample_shift=self.chan_time_delay(x)

        for i in range (self.raw_params['num_chans']):
            
            gen_pulse=self.simulate_pulse(self.sim_len)
            
            v=xp.roll(gen_pulse, -2*slef.raw_params['num_pols']*int(sample_shift[i]))
            self._write_(data, i)
            

#dispersion by convolution 

    def gaussian(self, x, mu, std, C):
        denominator = xp.sqrt(2*xp.pi*std**2)
        numerator = xp.exp(-1*(x - mu)**2/(2*std**2))
        return C*(numerator/denominator)

    def simulate_pulse(self, data_points):
        x = xp.arange(width) #Span of Gaussian Pulse
        gaussian_window1 = gaussian(x, int(width/4), 200, 50)
        gaussian_window2 = gaussian(x, int(3*width/4), 200, 100)
        gaussian_window = 1+ gaussian_window1 + gaussian_window2
        scale_factors = xp.sqrt(2)*gaussian_window
        noise = xp.random.normal(0,1,data_points)
        for i in range(width):
            if scale_factors[i] >= 1:
                noise[(int)(data_points/2) + i] *= scale_factors[i]
            else:
                continue
        real = noise
        imag = noise
        final_arr = xp.zeros(2*data_points)
        final_arr[::2] = real
        final_arr[1::2] = imag
        ax1 = plt.subplot(211)
        ax1.plot(xp.asnumpy(1+ gaussian_window1 + gaussian_window2))
        ax2 = plt.subplot(212)
        ax2.plot(xp.asnumpy(final_arr))
        plt.show()
        return final_arr

    
    def clear_cache(self):
        
        mempool = xp.get_default_memory_pool()
        print(mempool.used_bytes()/(1024*1024))             
        print(mempool.total_bytes()/(1024*1024)) 

        mempool.free_all_blocks()
        print(mempool.used_bytes()/(1024*1024))              
        print(mempool.total_bytes()/(1024*1024)) 
    
        
    def _smear_(self):
        
        f_start=self.raw_params['fch1']
        f_end=self.raw_params['fch1'] + self.obs_bw
        
        self.smear= self.D*self.dm*(f_start**-2 - f_end**-2)
        
        
    def impulse_length(self):
        
        f_chan_arr= self.raw_params['fch1']+xp.linspace(0, self.obs_bw, self.raw_params['num_chans'] )
        chan_centr=f_chan_arr+abs(self.raw_params['chan_bw'])/2.0
        
        for i in range(self.raw_params['num_chans']-1):
            
            t_d= self.D*self.DM*(chan_centr[i]**-2 - chan_centr[i+1]**-2)
        
        imp_length=t_d//self.raw_params['tbin']

        n=xp.floor(xp.log(2*imp_length)/xp.log(2))
        nfft=2**n
        
        print(imp_length, nfft)

        return((imp_length), (nfft))


    def disperse(self):
        
        f_coarse_dev=xp.linspace(0,self.obs_bw,self.raw_params['num_chans'], endpoint=False)
        sim_len=blocks_to_read * data_size

        imp_length, nfft=self.impulse_length()

        with open(f'{self.input_file_stem}.0000.raw', 'rb') as org:
            with open(f'{self.input_file_stem}_dispersed.0000.raw', 'wb') as f:

                data=xp.empty(chan_size,xp.int8)

                for i in range self.raw_params['num_chans']:

                    h=self.imp_res(f_coarse_dev[i], imp_length[i]) 

    #                 data_chan_cmplx=overlapSave(pulse_complex,h,nfft)
                    data_chan_cmplx=xp.convolve(self.simulate_pulse(sim_len),h, mode='same')

                    data[::2]=data_chan_cmplx.real
                    data[1::2]=data_chan_cmplx.imag
                    self._write_(data, i)
                    
    
    def _write_(self,chan_data, chan_no):
        
        chan_data.reshape(slef.blocks_to_read, slef.chan_size)
        
        with open(f'{self.input_file_stem}.0000.raw', 'rb') as org:
            with open(f'{self.input_file_stem}_dispersed.0000.raw', 'wb') as f:
                
                for j in range(self.blocks_to_read):

                    f.seek((slef.start_block-1+j)*self.block_read_size, 0) 
                    org.seek((self.start_block-1+j)*self.block_read_size, 0)

                    f.seek(self.header_size, 1)
                    org.seek(self.header_size, 1)

                    block= xp.frombuffer(org.read(self.data_size), dtype=xp.int8).reshape(self.raw_params['num_chans'], self.chan_size)
            #                        
                    dispersed_chunk=block[chan_no]+chan_data[j]

                    f.seek(chan_no*self.chan_size, 1)
                    f.write(xp.array(dispersed_chunk, dtype=xp.int8).tobytes())

                

    def imp_res(self, f_coarse_dev, imp_length):

        f0=self.raw_params['fch1']
        K=2 * xp.pi * self.D * self.dm / f0**2
        
        fl = xp.linspace(0,self.raw_params['chan_bw'], imp_length, endpoint=False) + f_coarse_dev
        V=fl**2/(fl+f0)

        H= xp.exp(1j*K*V) 
        return(xp.fft.ifft(H))


    def overlapSave(self, x,h,N,mode='full'): # N: number of FFT points

    #     Inputs:
    #     x - input sequence
    #     h - input impulse response
    #     N - number of FFT points
    #     mode - {'full','valid'}
    #            'full': outputs len(x)+len(h)-1
    #            'vaild': outputs data excludes leading and trailing zeros
    #                    the output length will be max(len(x),len(h)) - min(len(x),len(h)) + 1

        Lx = len(x)
        M = len(h)
        L = N-M+1

        hp = xp.append(h,xp.zeros(L-1))
        X = getBlockMatrix(x,L,M)
        FX = xp.fft.fft(X,axis=0)
        Fhp = xp.fft.fft(hp)
        FHp = xp.transpose(xp.tile(Fhp,(X.shape[1],1)))
        FY = xp.multiply(FX,FHp)
        Y_aliased = xp.fft.ifft(FY,axis=0) 
        Y = Y_aliased[xp.arange(M-1,N),:]
        y_vector = xp.ravel(Y,order='F')
    #     y = y_vector[0:Lx]
    #     print(len(y_vector))
    #     y = y_vector[2852:Lx+2852]

        if mode=='full':
            y = y_vector[0:Lx+M-1]
        elif mode=='valid':
            N_valid = max(len(x),len(h)) - min(len(x),len(h)) + 1 
            N_extra = N - N_valid
            N_lead = int(xp.floor(N_extra/2))
            #N_trail = int(xp.ceil(N_extra/2))
            y = y_vector[0:Lx+M-1]
    #         y = y1[xp.arange(N_lead,N_valid+1)]  
    #         y = y1[xp.arange(0,N_valid+1)]
        return(y)

    def getBlockMatrix(self, x,L,M):

        Lx = len(x)
        N = L+M-1
        Ly = Lx+M-1

        xc = xp.append(x,xp.zeros(L+M-1))
        xd = xp.append(xp.zeros(M-1),xc)


        if Ly%L != 0:
            ncols = (Ly//L) + 1
        else:
            ncols = (Ly//L)

        X = xp.zeros([N,ncols], dtype=complex)
        count = 0
        for i in xp.arange(0,Ly-1,L):
            X[:,count] = xd[i:i+N]
            count = count + 1

        return(X)
