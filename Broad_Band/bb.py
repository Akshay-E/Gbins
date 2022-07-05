import matplotlib.pyplot as plt 
import os
import time 
import shutil 
from setigen.voltage import raw_utils
import warnings
import astropy.units as u
import scipy

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np


class broadband(object):
    
    def __init__(self, 
                 input_file_stem,
                 pulse_time,
                 dm=100, 
                 width=1000, 
                 snr=10      
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
        
        
        self.calc_smear() 
        self.adjust_time()
        self.file_params()
        
        print(f" Adjusted injection time for channel {self.f_low + abs(self.obs_bw)} MHz {round(self.adjusted_pulse_time,3)}")
        
    
    def calc_smear(self,x=2):
        
        f_end=self.f_low + abs(self.obs_bw)
        
        self.smear= self.D*self.dm*(self.f_low**-x - f_end**-x)
        
    
    def adjust_time(self):
        
        self.blocks_per_file = raw_utils.get_blocks_per_file(self.input_file_stem)
        self.time_per_block = self.raw_params['block_size'] / (self.raw_params['num_antennas'] * self.raw_params['num_chans'] * (2 * self.raw_params['num_pols'] * self.raw_params['num_bits'] // 8)) * self.raw_params['tbin']

        self.blocks_to_read=int( np.ceil(self.smear/self.time_per_block))
        self.start_block= int(np.ceil(self.pulse_time/self.time_per_block))
        
        self.adjusted_pulse_time=(self.start_block - 1) * self.time_per_block 

        try :
            
            assert self.adjusted_pulse_time + self.smear < self.raw_params['obs_length'] 
        
        except AssertionError:
            
            self.pulse_time=self.raw_params['obs_length'] - self.smear
            
            assert self.pulse_time  > 0 , f"Smearing across the band ({round(self.smear, 3)}) exceeds length of file ({round(self.raw_params['obs_length'], 3)}) from adjusted pulse time ({round(self.adjusted_pulse_time, 2)}). Try changing DM. "
            
            warnings.warn('Smearing exceeds length of the file. Re-adjusting pulse time...')
            
            self.adjust_time()
            

    def file_params(self):
        
        header=raw_utils.read_header(f'{self.input_file_stem}.0000.raw')
        
        self.header_size= int(512 * np.ceil((80 * (len(header) + 1)) / 512))
        self.data_size= int(self.raw_params['block_size'])
        self.chan_size=int(self.data_size/self.raw_params['num_chans'])
        self.block_read_size =  self.header_size + self.data_size
        

# Dispersion by sample shifting 

    def chan_time_delay(self,x):

        f_chan_arr= self.f_low+np.linspace(0, abs(self.obs_bw), self.raw_params['num_chans'], endpoint=False )
        chan_centr=f_chan_arr+abs(self.raw_params['chan_bw']/2.0)
        
        time_delay= self.D*self.dm*(chan_centr**-x - chan_centr[-1]**-x)
        samples_to_shift=np.ceil(time_delay/self.raw_params['tbin'])
        
        return(samples_to_shift)

    
    @classmethod
    def gauss(cls, x=None, x0=None, fwhm=None, a=None, width=None):
#         print(x, x0, fwhm, a)
        if x is None:
            x=np.arange(width)
        if x0 is None:
            x0=width/2
        if fwhm is None:
            fwhm=width/2
            
        sigma = (fwhm/2) / np.sqrt(2*np.log(2))
        
        if a is None:
            a= 1/(sigma*np.sqrt(2*np.pi))
        
        G= a  * np.exp(-(x-x0)**2 / (2*sigma**2))
#         plt.plot(G)
#         plt.show()
        return G
    
    def sample_shift(self, x=2 , b_type='N', op_dir=None, profile=None):
        
        path, pulse_profile= self.dispatcher(op_dir, profile)
        chan_flip=False
        
        if b_type=='N':
            flip = not self.raw_params['ascending']
        elif b_type=='A1':
            flip = self.raw_params['ascending']
        elif b_type=='A2':
            chan_flip, flip = not chan_flip, self.raw_params['ascending']
        elif b_type=='A3':
            chan_flip, flip = not chan_flip, not self.raw_params['ascending']
        else:
            raise Exception("Invalid plot type ")
            
        if x!=2:
            self.calc_smear(x)
            self.adjust_time()
        
        td_in_samps=self.chan_time_delay(x)
        
        samples_per_chan=int(self.chan_size/(2*self.raw_params['num_pols']))

        if flip:  
            td_in_samps=np.flip(td_in_samps)
        if chan_flip:
            td_in_samps=samples_per_chan*self.blocks_to_read - td_in_samps -self.width
        
        with open(path, 'r+b') as self.file_handler:

            self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)

            print(f" At {self.file_handler.tell()}")

            block_cmplx=self.collect_data()
            
            pulse_profile=broadband.gauss(a=self.snr, width=self.width)

            for i in range(self.raw_params['num_chans']):
            
                s= int(td_in_samps[i])
                e= int(td_in_samps[i]+self.width)

                block_cmplx[i][s:e]*=pulse_profile
            
            self.write_blocks(block_cmplx)
        

    @classmethod
    def disperse_filterbank(cls, frame, params, b_type='N', save=True):
        
        width, snr, t0, dm,x = params['width'], params['snr'], params['t0'], params['dm'], params.get('x',2)
        
        assert t0<frame.ts[-1], f"Start time {t0} seconds exceeds length of the file, {frame.ts[-1]} seconds"

        rms  = frame.get_intensity(snr=snr)
        fch1 = frame.get_frequency(frame.fchans-1)

        width_in_chans = width / frame.dt
        t0_in_samps = (t0 / frame.dt) - frame.ts[0]
        
        # tdel_in_samps = 4.15e-3 * dm * ((fch1/1e9)**(-2) - (frame.fs/1e9)**(-2)) / frame.dt
    
        tdel_in_samps = 4.15e-3 * dm * ((frame.fs/1e9)**(-x) - (fch1/1e9)**(-x)) / frame.dt
        t0_in_samps = t0_in_samps + tdel_in_samps 

        t = np.arange(frame.tchans)

        t2d, tdel2d = np.meshgrid(t, t0_in_samps)

        profile = broadband.gauss(t2d, tdel2d, width_in_chans, rms)
        
        if b_type=='N':
            pass
        if b_type=='A1':
            profile=np.flip(profile)
        if b_type=='A2':
            profile=np.flip(profile, axis=0)
        if b_type=='A3':
            profile=np.flip(profile, axis=1)
        
        
        frame.data +=profile.T

        plt.figure(figsize=(10,6))
        frame.plot()
        
        if save:
            frame.save_fil(filename='./data/dispersed_frame.fil')

        return(frame.data)


#Dispersion by Convolution
    
    def imp_res(self, imp_length):
        
        f_coarse_dev=np.linspace(0,np.abs(self.obs_bw),self.raw_params['num_chans'], endpoint=False)
        H=np.empty((self.raw_params['num_chans'], imp_length), dtype=complex)
        
        for i in range(self.raw_params['num_chans']):
            
            K=2 * xp.pi * self.D *1e6 * self.dm / self.f_low**2

            fl = xp.linspace(0,np.abs(self.raw_params['chan_bw']), imp_length, endpoint=False) + f_coarse_dev[i]
            V=fl**2/(fl + self.f_low)

            H[i]= xp.asnumpy ( xp.fft.ifft ( xp.exp(1j * K * V ) ))

        if not self.raw_params['ascending']:
            H=np.flip(H, axis=0)
            
        return(H)
    

    def dispatcher(self, op_dir, profile):
        
        if profile is None:
            profile=broadband.gauss(a=self.snr, width=self.width)
        else:
            self.width=len(profile)
        
        if op_dir is None:
            path=os.path.join(os.getcwd(), f'{self.input_file_stem}_dispersed.0000.raw')
        else:
            path=os.path.join(op_dir, f'{self.input_file_stem}_dispersed.0000.raw')
        
        shutil.copyfile(f'{self.input_file_stem}.0000.raw', path )
        
        return(path, profile)
    
        
    def disperse(self, op_dir=None, profile=None, plot=False, plot_alt=None):
        
        path, pulse_profile= self.dispatcher(op_dir, profile)
        
        impulse_len=int(np.ceil(self.smear/self.raw_params['tbin']))

        print(f"imp length:{impulse_len}")

        with open(path, 'r+b') as self.file_handler:
            
            self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)

            print(f" At {self.file_handler.tell()}")
    
            block_cmplx=self.collect_data()
            
            if self.start_block - self.blocks_to_read<1:
                block_cmplx[ :,(block_cmplx.shape[1])//2:(block_cmplx.shape[1])//2+self.width ]*=pulse_profile
            else:
                block_cmplx[ :, :self.width ]*=pulse_profile
                
            block_cmplx, conv_mode=self.pad(block_cmplx, impulse_len)
            h=self.imp_res( impulse_len) 
            
            dispersed_ts=scipy.signal.fftconvolve(block_cmplx, h, mode=conv_mode, axes=1)
#             print(block_cmplx.shape, h.shape, dispersed_ts.shape)

            if plot:
                if plot_alt is None:
                    plot_alt=1
        
                for k in range(0,self.raw_params['num_chans'],plot_alt):
                
                    plt.figure(figsize = (10,6))
                    a1=plt.subplot(311)
                    a1.title.set_text(f'Generated pulse, channel {k}')
                    a1.plot(block_cmplx[k].real)

                    a2=plt.subplot(312)
                    a2.title.set_text('Impulse response of ISM')
                    a2.plot( h[k].real)

                    a3=plt.subplot(313)
                    a3.title.set_text('Dispersed pulse')
                    a3.plot(dispersed_ts[k].real)
                    
                    plt.setp((a1,a2,a3), xticks=[])
                    plt.show()
            
            self.write_blocks(dispersed_ts)


    def collect_data(self, _from=None, to=None):
        
        if _from is None:
            _from=self.start_block
        if to is None:
            to=self.blocks_to_read
        
        block= (np.frombuffer(self.file_handler.read(self.block_read_size),offset=self.header_size
                              , dtype=np.int8)).reshape(self.raw_params['num_chans'],self.chan_size)
        block=block.astype(float)
        print(f"seek first bl {self.file_handler.tell()}")

        for i in range(1,self.blocks_to_read):

            nxt_block= (np.frombuffer(self.file_handler.read(self.block_read_size),offset=self.header_size
                          , dtype=np.int8)).reshape(self.raw_params['num_chans'],self.chan_size)
            nxt_block=nxt_block.astype(float)

            block=np.hstack((block, nxt_block))
        
        print(f"seek end {self.file_handler.tell()}")
        
        block_cmplx=block[: , ::2] + 1j*block[: , 1::2]
        
        return(block_cmplx)
        
        
    def pad(self, L, il):
        
        print('pad ',self.file_handler.tell())
        pad_start_block= self.start_block - self.blocks_to_read
        
        if pad_start_block<1:
            warnings.warn('Incomplete data points for padding. Boundary effects of convolution will be visible.')
            mode='same'
            return(L, mode)
       
        else:
            self.file_handler.seek((pad_start_block-1) * self.block_read_size, 0)
            cmplx_data=self.collect_data(pad_start_block)
            
            M_1= cmplx_data[:, 1-il:]
            
            padded_block=np.hstack((M_1, L) )
            mode='valid'
            return(padded_block, mode)
    
    
    def write_blocks(self, data_chunk):
        
        self.file_handler.seek((self.start_block-1) * self.block_read_size, 0)
        
        final_data=np.empty((self.data_size))
        print('writing',self.file_handler.tell())

        for i in range(self.blocks_to_read):
            
            self.file_handler.seek(self.header_size, 1)
            
            s=int(i*self.chan_size/2)
            e=int(i*self.chan_size/2 + self.chan_size/2)

            block_i=data_chunk[:, s:e]
            
            final_data[::2]=np.ravel(block_i.real)
            final_data[1::2]=np.ravel(block_i.imag)
            
            self.file_handler.write(np.array(final_data, dtype=np.int8).tobytes())
            print('writ block i',self.file_handler.tell())
            
            
            
