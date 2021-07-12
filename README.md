# Heat-source-detection

## Overview
The code performs deblurring of images using the MCMC approach designed based on the mixed norm theorem presented in the paper.

## Description

### Image Deblurring experiments
The main file is  deblurring2d.ipynb that performs deblurring on various images provided in the paper and it's appendix.
The images  are contained in the image folder. 
The main function that performs deblurring is in Deblurring.py, which utilizes several other functions in Utils.py. 


### Heat reconstruction experiments
The main file is  Heat1d.ipynb that recovers random heat sources from it's diffused measurements and plots the accuracy of reconstruction using errorbars. This file also contains 
experiments of deblurring 1D signals with small TV norm.
The main function that performs the recontruction is HSRC.py.

![Results](heat_source_rec.svg)


