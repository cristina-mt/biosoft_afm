
def WT_gauss_fixed_2D(mat_img,a_scale,display_img):
    # Symmetric Gaussian in 2D for use in the Wavelet Transform at a fixed scale
    # It returns the WT_modulus, normalised by the maximum values
    # IMPORTANT: not properly WT, as no norm is used to normalised
    #            the norm (L1/L2) needs to be added dependent on the case
    #        NOTE:  for future use of proper WT, add correct normalisation  (29-06-2016, CMT)
    
    import numpy as np
    
    # Shifted FFT of the image
    fft_img=np.fft.fftshift(np.fft.fft2(mat_img))    
    # x and y vectors for WT, centered at 0    
    y=np.arange(-mat_img.shape[0]/2,mat_img.shape[0]/2)         
    x=np.arange(-mat_img.shape[1]/2,mat_img.shape[1]/2)   
    # x and y grid for 2D computation
    xg=np.tile(x,(len(y),1));yg=np.tile(y,(len(x),1)).transpose()   
    
    # Gaussian function scaled by a_scale (symmetric) and its (shifted) FFT
    # The way the power 2 is computed (x*x) is to avoid numerical errors that arise otherwhise
    phi=np.exp(-((xg/a_scale)*(xg/a_scale)+(yg/a_scale)*(yg/a_scale))/2) 
    phi_g=phi/np.sum(phi.flatten())
    fft_phi=np.fft.fftshift(np.fft.fft2(phi_g))          
    # WT at scale a_scale. Normalized WT modulus, where 1=WT_max
    WT=np.fft.ifftshift(np.fft.ifft2(fft_phi*fft_img))
    WT_mod=1/(2*np.pi)*np.abs(WT)  #WT_mod=WT_mod/np.max(WT_mod.flatten())        
    
    # If display_img=1, then it plots the image in real and fourier space, as well as the WT, and the filter used
    if display_img==1:
        import matplotlib.pyplot as plt
       
        fig=plt.figure(figsize=(10,8),facecolor='k')
        
        # Image in real space
        ax1=plt.subplot(221)
        plt.imshow(mat_img,cmap='gray_r',aspect='equal',interpolation='none',origin='upper')
        plt.title('Raw Image (cropped)',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.5);ax1.spines[axis].set_color('white')
        # WT modulus in real space (filtered image by gaussian function)
        ax2=plt.subplot(222)
        plt.imshow(WT_mod,cmap='gray_r',aspect='equal',interpolation='none',origin='upper')
        plt.title('WT mod (Filtered Image)',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(1.5);ax2.spines[axis].set_color('white')
        # Image in Fourier space, log scale
        ax3=plt.subplot(223)
        plt.imshow(np.log10(np.abs(fft_img)),cmap='inferno',aspect='equal',interpolation='none',origin='upper',vmin=0,vmax=np.max(np.log10(np.abs(fft_img))))
        plt.title('FFT Image, log scale',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(1.5);ax3.spines[axis].set_color('white')
        # Gaussian filter in Fourier space
        ax4=plt.subplot(224)
        plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(WT_mod)))),cmap='inferno',aspect='equal',interpolation='none',origin='upper',vmin=0,vmax=np.max(np.log10(np.abs(fft_img))))
        plt.title('FFT gaussian filter, log scale',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax4.spines[axis].set_linewidth(1.5);ax4.spines[axis].set_color('white')
        plt.show()
        
    return WT_mod


def WT_first_der_gauss_fixed_2D(mat_img,a_scale,display_img):

    # Symmetric Gaussian in 2D for use in the Wavelet Transform at a fixed scale
    # It returns the WT_modulus, normalised by the maximum values
    # IMPORTANT: not properly WT, as no norm is used to normalised
    #            the norm (L1/L2) needs to be added dependent on the case
    #        NOTE:  for future use of proper WT, add correct normalisation  (29-06-2016, CMT)
    
    import numpy as np
    
    # Shifted FFT of the image
    fft_img=np.fft.fftshift(np.fft.fft2(mat_img))    
    # x and y vectors for WT, centered at 0    
    y=np.arange(-mat_img.shape[0]/2,mat_img.shape[0]/2)         
    x=np.arange(-mat_img.shape[1]/2,mat_img.shape[1]/2)   
    # x and y grid for 2D computation
    xg=np.tile(x,(len(y),1));yg=np.tile(y,(len(x),1)).transpose()   
    
    # Gaussian function scaled by a_scale (symmetric) and its (shifted) FFT
    # The way the power 2 is computed (x*x) is to avoid numerical errors that arise otherwhise
    phi_g=np.exp(-((xg/a_scale)*(xg/a_scale)+(yg/a_scale)*(yg/a_scale))/2) 
    fphi_x=-(xg/a_scale)*phi_g;fphi_x=fphi_x/np.sum(fphi_x.flatten())
    fphi_y=-(yg/a_scale)*phi_g;fphi_y=fphi_y/np.sum(fphi_y.flatten())
    fft_fphix=np.fft.fftshift(np.fft.fft2(fphi_x));fft_fphiy=np.fft.fftshift(np.fft.fft2(fphi_y))
    
    # WT at scale a_scale. Normalized WT modulus, where 1=WT_max
    fWT_x=np.fft.ifftshift(np.fft.ifft2(fft_fphix*fft_img));fWT_y=np.fft.ifftshift(np.fft.ifft2(fft_fphiy*fft_img))
    fWT_mod=np.sqrt(np.abs(fWT_x)**2+np.abs(fWT_y)**2)  #fWT_mod=fWT_mod/np.max(fWT_mod.flatten())
    fWT_arg=np.angle(fWT_x+fWT_y*1j,deg=True)
    # If display_img=1, then it plots the image in real and fourier space, as well as the WT, and the filter used
    if display_img==1:
        import matplotlib.pyplot as plt
       
        fig=plt.figure(figsize=(10,8),facecolor='k')
        
        # Image in real space
        ax1=plt.subplot(221)
        plt.imshow(mat_img,cmap='gray',aspect='equal',interpolation='none',origin='upper')
        plt.title('Raw Image (cropped)',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.5);ax1.spines[axis].set_color('white')
        # WT modulus in real space (filtered image by gaussian function)
        ax2=plt.subplot(222)
        plt.imshow(fWT_mod,cmap='gray_r',aspect='equal',interpolation='none',origin='upper')
        plt.title('WT mod (Edges)',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(1.5);ax2.spines[axis].set_color('white')
        # Image in Fourier space, log scale
        ax3=plt.subplot(223)
        plt.imshow(np.log10(np.abs(fft_img)),cmap='inferno',aspect='equal',interpolation='none',origin='upper',vmin=0,vmax=np.max(np.log10(np.abs(fft_img))))
        plt.title('FFT Image, log scale',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(1.5);ax3.spines[axis].set_color('white')
        # WT argument
        ax4=plt.subplot(224)
        plt.imshow(fWT_arg,cmap='inferno',aspect='equal',interpolation='none',origin='upper',vmin=0,vmax=np.max(np.log10(np.abs(fft_img))))
        plt.title('WT argument',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax4.spines[axis].set_linewidth(1.5);ax4.spines[axis].set_color('white')
        plt.show()
        
    return fWT_mod,fWT_arg
    
    
