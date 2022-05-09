import matplotlib.pyplot as plt 
import os
import sys
import time 
import shutil 
from setigen.voltage import raw_utils

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np


class broadband(object):
    
    def __init__(self, 
                 input_file_stem,
                 pulse_time,
                 dm=10, 
                 width=1000, 
                 snr=100         
                ):
    
        self.raw_params = raw_utils.get_raw_params(input_file_stem=input_file_stem)
        self.raw_params['fch1'], self.raw_params['chan_bw'], self.raw_params['center_freq'] = self.raw_params['fch1']/1e6, self.raw_params['chan_bw']/1e6, self.raw_params['center_freq']/1e6
        self.obs_bw= self.raw_params['num_chans'] * self.raw_params['chan_bw']
        if self.raw_params['ascending']:
            self.f_low= self.raw_params['fch1']
        else:
            self.f_low= self.raw_params['fch1'] + self.obs_bw
        
        self.input_file_stem=input_file_stem
        self.dm=dm
        self.width=width
        self.snr=snr
        self.D=4.148808e3
        self.pulse_time=pulse_time
        
        assert self.pulse_time > 0 , f"injection time cannot be 0"
        assert self.pulse_time < self.raw_params['obs_length'], f"injection time cannot be greater than length of file( {self.raw_params['obs_length']}) "
        
        self.blocks_per_file = raw_utils.get_blocks_per_file(input_file_stem)
        self.time_per_block = self.raw_params['block_size'] / (self.raw_params['num_antennas'] * self.raw_params['num_chans'] * (2 * self.raw_params['num_pols'] * self.raw_params['num_bits'] // 8)) * self.raw_params['tbin']
        
        self.calc_smear() 
        
        self.adjust_time()
        
#         self.blocks_to_read=int(np.ceil(self.smear/self.time_per_block))
        
#         self.start_block= int(np.ceil(self.pulse_time/self.time_per_block))
        
#         if self.raw_params['ascending']:
#             self.adjusted_pulse_time=(self.start_block) * self.time_per_block
#             assert self.adjusted_pulse_time - self.smear>=0, f"smearing across band({self.smear}) exceeds length of file({self.raw_params['obs_length']}) "
#         else:
#             self.adjusted_pulse_time=(self.start_block-1) * self.time_per_block
#             assert self.adjusted_pulse_time + self.smear<self.raw_params['obs_length'], f"smearing across band({self.smear}) exceeds length of file({self.raw_params['obs_length']}) "

        
        header=raw_utils.read_header(f'{self.input_file_stem}.0000.raw')
        
        self.header_size= int(512 * np.ceil((80 * (len(header) + 1)) / 512))
        self.data_size= int(self.raw_params['block_size'])
        self.chan_size=int(self.data_size/self.raw_params['num_chans'])
        self.block_read_size =  self.header_size + self.data_size
        
        self.sim_len= int(self.blocks_to_read * self.chan_size/2)
        shutil.copyfile(f'{self.input_file_stem}.0000.raw', f'{self.input_file_stem}_dispersed.0000.raw')
        
    def adjust_time(self):
        
        try :
            assert self.adjusted_pulse_time > self.smear and self.adjusted_pulse_time < self.raw_params['obs_length'] 
        
        except AssertionError:
            self.pulse_time=self.adjusted_pulse_time + self.smear
            assert self.pulse_time  < self.raw_params['obs_length'], f"Smearing across the band exceeds length of file from adjusted pulse time. Try changing DM. "
            print("WARNING: Smearing exceeds length of file. Adjusting pulse time...")
            self.calc_smear()
            
        print(f"Adjusted injection time for channel {self.f_low} MHz {round(self.adjusted_pulse_time,3)}")
        

    def calc_smear(self,x=2):
        
        f_end=self.f_low + abs(self.obs_bw)
        
        self.smear= self.D*self.dm*(self.f_low**-x - f_end**-x)
        
#         self.blocks_per_file = raw_utils.get_blocks_per_file(self.input_file_stem)
#         self.time_per_block = self.raw_params['block_size'] / (self.raw_params['num_antennas'] * self.raw_params['num_chans'] * (2 * self.raw_params['num_pols'] * self.raw_params['num_bits'] // 8)) * self.raw_params['tbin']
        
        self.blocks_to_read=int( 1 + np.ceil((self.smear-self.time_per_block/2) / self.time_per_block))
        self.start_block= int(np.ceil(self.pulse_time/self.time_per_block))
        
        self.adjusted_pulse_time=(self.start_block) * self.time_per_block - self.time_per_block/2.0
        
        print(self.smear)
#         return(smear)
    
    

    def gaussian(self, x, mu, std, C):
        
        denominator = xp.sqrt(2*xp.pi*std**2)
        numerator = xp.exp(-1*(x - mu)**2/(2*std**2))
        return C*(numerator/denominator)
    
    def simulate_pulse(self,  cmplx=True):
        
        x = xp.arange(self.width) #Span of Gaussian Pulse
        gaussian_window1 = self.gaussian(x, int(self.width/4), 50, 10000)
        gaussian_window2 = self.gaussian(x, int(3*self.width/4), 50, 20000)
        gaussian_window = 1+ gaussian_window1 + gaussian_window2
        scale_factors = xp.sqrt(2)*gaussian_window
        noise = xp.random.normal(0,1,self.sim_len)
        
#         if self.raw_params['ascending']:
#             loc=self.sim_len-2*self.width
#         else:
#             loc=2*self.width
        loc=self.sim_len/2
            
        for i in range(self.width):
            if scale_factors[i] >= 1:
                noise[(int)(loc) + i] *= scale_factors[i]
            else:
                continue
        
        if cmplx:
            final_arr = noise+1j*noise
        else:
            final_arr = xp.zeros(2*self.sim_len)
            final_arr[::2] = noise
            final_arr[1::2] = noise
        
        ax1 = plt.subplot(211)
        ax1.plot(xp.asnumpy(gaussian_window))
        ax2 = plt.subplot(212)
        ax2.plot(xp.asnumpy(final_arr.real))
        plt.show()
        
        return final_arr
    
    
    def _write_(self,chan_data, chan_no, reverse=False):
         
        if reverse:
            chan_no= self.raw_params['num_chans'] - 1 - chan_no
        
        print(f"writing chan {chan_no}")
        
        blocks=self.blocks_to_read
        
        if chan_data==self.chan_size:
            chan_data=np.tile(chan_data,2)
            blocks+=1
            
        chan_data.reshape(blocks, self.chan_size)
        print(chan_data.shape)

        with open(f'{self.input_file_stem}_dispersed.0000.raw', 'r+b') as f:

            for j in range(self.blocks_to_read):

                f.seek((self.start_block-1-j)*self.block_read_size, 0) 
                f.seek(self.header_size, 1)
#                 print(f.tell()-self.header_size)

                block= xp.frombuffer(f.read(self.data_size), dtype=xp.int8).reshape(self.raw_params['num_chans'], self.chan_size)
#                 plt.plot(xp.asnumpy(chan_data))
#                 plt.show()

                dispersed_chunk=block[chan_no].astype(int)+chan_data[j]
                plt.plot(xp.asnumpy(dispersed_chunk).astype(xp.int8))
                plt.show()

                f.seek(-self.data_size, 1)
#                 print(f.tell()-self.header_size)
                f.seek(chan_no*self.chan_size, 1)
#                 print(f.tell()-self.header_size)

                f.write(xp.array(dispersed_chunk, dtype=xp.int8).tobytes())

    def clear_cache(self):

        mempool = xp.get_default_memory_pool()
        print(mempool.used_bytes()/(1024*1024))             
        print(mempool.total_bytes()/(1024*1024)) 

        mempool.free_all_blocks()
        print(mempool.used_bytes()/(1024*1024))              
        print(mempool.total_bytes()/(1024*1024)) 

        
#dispersion by sample shifting 

    def chan_time_delay(self,x):

        f_chan_arr= self.f_low+xp.linspace(0, abs(self.obs_bw), self.raw_params['num_chans'], endpoint=False )
        print(f_chan_arr)
        chan_centr=f_chan_arr+abs(self.raw_params['chan_bw']/2.0)
        print(chan_centr)
        
        time_delay= self.D*self.dm*(chan_centr[0]**-x - chan_centr**-x)
        print(time_delay, len(time_delay))
        samples_to_shift=np.ceil(time_delay/self.raw_params['tbin'])
        print(samples_to_shift)
        return(samples_to_shift)


    def sample_shift(self,x=2 , b_type='N' ):
        
        order=-1
#         if not self.raw_params['ascending']:
#             order=-1        
        
        if b_type=='N':
            flip = not self.raw_params['ascending']
        elif b_type=='A1':
            flip = self.raw_params['ascending']
        elif b_type=='A2':
            order, flip = -1*order, self.raw_params['ascending']
        elif b_type=='A3':
            order, flip = -1*order, not self.raw_params['ascending']
            
        if x!=2:
            
            smear_x=self.calc_smear(x)
            self.adjust_time()
        
#         sample_shift=self.chan_time_delay(x)
#         gen_pulse=self.simulate_pulse(cmplx=False)
        
#         print(order, flip)
        
#         for i in range (self.raw_params['num_chans']):
            
#             roll_samples= 2 * order * self.raw_params['num_pols'] * int(sample_shift[i])
            
#             v=xp.roll(gen_pulse, roll_samples)
#             self._write_(v, i, flip)
            

#dispersion by convolution 

    def impulse_length(self):
        
        if self.raw_params['ascending']:
            f_lower= self.raw_params['fch1']
        else:
            f_lower= self.raw_params['fch1'] + self.obs_bw
#             reverse=True
            
        
        f_chan_start= f_lower + xp.linspace(0, np.abs(self.obs_bw), self.raw_params['num_chans'], endpoint=False)
        print(f_chan_start)

        t_d=self.D*self.dm*(f_chan_start**-2 - (f_chan_start+np.abs(self.raw_params['chan_bw']))**-2)
    
        print(t_d, len(t_d))

        imp_length=np.ceil(t_d/self.raw_params['tbin'])

        n=xp.floor(xp.log(2*imp_length)/xp.log(2))
        nfft=2**n
        
        return(imp_length, nfft)

    def disperse(self):
        
        f_coarse_dev=xp.linspace(0,np.abs(self.obs_bw),self.raw_params['num_chans'], endpoint=False)
        print(f_coarse_dev)

#         imp_length, nfft=self.impulse_length()
#         print(imp_length, max(imp_length))
        
        data=xp.empty(self.chan_size,float)
        
        gen_pulse=self.simulate_pulse()

        qq=int(np.ceil(self.smear/self.raw_params['tbin']))
        print(qq)
        
        for i in range (self.raw_params['num_chans']):
#         for i in range (0,self.raw_params['num_chans'],20):

            h=self.imp_res(f_coarse_dev[i], qq) 

#                 data_chan_cmplx=overlapSave(pulse_complex,h,nfft)
            data_chan_cmplx=xp.convolve(gen_pulse,h, mode='same')
#             ddd=xp.convolve(gen_pulse_complex,h, mode='full')

            data[::2]=data_chan_cmplx.real
            data[1::2]=data_chan_cmplx.imag
            
#             plt.plot(xp.asnumpy(xp.abs(data_chan_cmplx)), label=str(i))
#             plt.legend(title=str(i))
#             plt.plot(xp.asnumpy(xp.fft.fftshift(h)))
#             plt.show()
            self._write_(data, i,not self.raw_params['ascending'])
                    
    
    def imp_res(self, f_coarse_dev, imp_length):
        
#         print(f_low)
        
        K=2 * xp.pi * self.D *1e6 * self.dm / self.f_low**2
        
        fl = xp.linspace(0,np.abs(self.raw_params['chan_bw']), imp_length, endpoint=False) + f_coarse_dev
#         print(fl)
        V=fl**2/(fl + self.f_low)

        H= xp.exp(1j * K * V) 
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