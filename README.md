# G_BINS
Program for injecting natural and artificial broad band signals at raw volatge level into guppi raw files. 

## Dispersion by convolution :

Broadband signal injected by convolving complex raw volatges with the transfer function of ISM. 

Dispersed Pulse             |  De-dispersed Pulse
:-------------------------:|:-------------------------:
![by convolution](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_conv.png)|![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_conv_de.png)

## Dispersion by sample shifting 
Injecting natural and artificial broadband signals by computing delay between channels. 

#### Generating natural signal
Dispersed Pulse             |  De-dispersed Pulse
:-------------------------:|:-------------------------:
![by convolution](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_typeN.png)|![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_typeN_dedi.png)


#### Generating artificial signals with flipped axis
A1|A2|A3
:-------------------------:|:-------------------------:|:-------------------------:
![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_typeA1.png)|![by convolution](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_typeA2.png)|![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/neg5_typeA3.png)

#### Generating artificial signals following non-natural power law
x=1.8|x=2|x=2.2
:-------------------------:|:-------------------------:|:-------------------------:
![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/pl_1.8.png)|![by convolution](https://github.com/Akshay-E/G_BINS/blob/main/images/pl_2.png)|![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/pl_2.2.png)

## Injection in intensity domain :
Injection of broadband signal onto filterbank files

Natural and artificial dispersed pulse             |  Signals with non-natural power law
:-------------------------:|:-------------------------:
![by convolution](https://github.com/Akshay-E/G_BINS/blob/main/images/from_fil.png)|![by convolution_de-dispersed](https://github.com/Akshay-E/G_BINS/blob/main/images/from_fil_varying_exponents.png)







