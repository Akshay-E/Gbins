#!/usr/bin/env python
# coding: utf-8




import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import blimpy as bl
import setigen as stg
import sys

# # Create synthetic data with gaussian nois

sample_rate = 1024e6 #taken as 2*BW
num_taps = 8
num_branches = 1024

num_chans=int(sys.argv[1])
num_pols=int(sys.argv[2])

chan_bw = sample_rate / num_branches


antenna = stg.voltage.Antenna(sample_rate=sample_rate, 
                              fch1=500*u.MHz,
                              ascending=True,
                              num_pols=num_pols)


antenna.x.add_noise(v_mean=0, 
                    v_std=1)


antenna.x.add_constant_signal(f_start=500.5e6, 
                              drift_rate=-1*u.MHz/u.s, 
                              level=5)

digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps, 
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                           num_bits=8)



rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=0,
                                    num_chans=num_chans,
                                    block_size=512,
                                    blocks_per_file=64,
                                    num_subblocks=32)


print('samp rate ',antenna.sample_rate)
#print(rvb.sample_rate/1e6)

print('branches',rvb.num_branches)
print('chn bw',rvb.chan_bw)
print('tbin',rvb.tbin)
#print(rvb.fch1/1e6)
print('final channel freq in mhz',(rvb.fch1+chan_bw*rvb.num_chans)/1e6)


# In[11]:

op_file=sys.argv[3]
num_blocks=int(sys.argv[4])

rvb.record(output_file_stem=op_file,
           num_blocks=num_blocks, 
           length_mode='num_blocks',
           header_dict={'TELESCOP': 'GBT'},
           verbose=False)



#reading the generated file 

start_chan = 0

raw_params = stg.voltage.get_raw_params(input_file_stem=op_file,
                                        start_chan=start_chan)





print(raw_params)




