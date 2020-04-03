import numpy as np;import matplotlib.pyplot as plt

from fiber_tracking_AFM_v0 import *

filename="filename.txt"


# ====================================================================
# INPUT VARIABLES:
# --------------------------------------------------------------------
# n_iter: number of iterations for the "multiscale" gaussian smoothing
# pth: treshold to binarise the image (in fraction of the mean intensity)
# th_lchain: Minimum length of the chains to be considered (in pixels)
# ====================================================================

header_size=4
n_iter=2;pth=1e-2;th_lchain=5
img_info=[]

with open(filename,'r') as f:
    for iheader in range(0,header_size):
        line=f.readline()
        if ('Channel' or 'units') in line:
            img_info.append(line[line.find(':')+2:-1])
        else:
            img_info.append(line[line.find(':')+2:line.find('.')+3])
 
Mat_full_img=np.loadtxt(filename)*1e9
npix_ydir,npix_xdir=Mat_full_img.shape

xcal_factor=float(img_info[1])*1e3/npix_xdir;ycal_factor=float(img_info[2])*1e3/npix_ydir

mask_WT_aa,sk_WT_aa=img_bin_skeleton(Mat_full_img,n_iter,pth,0)
xsl_chain,ysl_chain,idl_chain,xj,yj=fiber_label_junction_det(sk_WT_aa,th_lchain)
xm,ym,m,im=fiber_local_slope_lin(xsl_chain,ysl_chain,idl_chain)

Fiber_info,Back_info,Profile_info=fiber_profile(Mat_full_img,mask_WT_aa,xm,ym,m,im)

[fiber_min,fiber_max,fiber_mean,fiber_x1,fiber_x2,fiber_y1,fiber_y2,fiber_xc,fiber_yc,fiber_slope,fiber_number,fiber_width,fiber_sum,fiber_corr_int_int_mean]=Fiber_info

[bleft_min,bleft_max,bleft_mean,bleft_sum,bleft_x1,bleft_x2,bleft_y1,bleft_y2,bright_min,bright_max,bright_mean,bright_sum,bright_x1,bright_x2,bright_y1,bright_y2]=Back_info
    
[sec_mean_range,sec_max_range,sec_mean_back_int_int,sec_length_back_int_int_mean]=Profile_info

fiber_width_nm=np.array(fiber_width)*(xcal_factor)
cal_bar_1um=np.arange(0,1000/xcal_factor)

plt.subplot(221)
plt.imshow(Mat_full_img,cmap='inferno',clim=[-2,np.max(Mat_full_img.flatten())])
plt.plot(cal_bar_1um+50,450*np.ones(len(cal_bar_1um)),lw=5,c='w')
plt.xlim([0,512]);plt.ylim([512,0])

plt.subplot(223)
plt.imshow(Mat_full_img,cmap='gray_r',clim=[-2,np.max(Mat_full_img.flatten())],interpolation='none')
plt.scatter(fiber_xc,fiber_yc,c=fiber_number,edgecolor='none',s=5)
plt.plot(cal_bar_1um+50,450*np.ones(len(cal_bar_1um)),lw=5,c='k')
plt.xlim([0,512]);plt.ylim([512,0])



plt.subplot(322)
plt.scatter(fiber_width_nm,fiber_max,c=fiber_number,edgecolor='none',s=10)

plt.subplot(426)
plt.scatter(fiber_number,fiber_max,c=fiber_number,s=25,alpha=0.25)
plt.grid(True)
plt.subplot(428)
plt.scatter(fiber_number,np.array(fiber_max)/np.array(fiber_width_nm),c=fiber_number,s=25,alpha=0.25)
plt.grid(True)


