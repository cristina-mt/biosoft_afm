# ====================================================================
# Code to extract the information related to intensity profiles of filament-like structures
# Algorithm based on "multi-scale smoothing" and skeletonization of the image
# Initially written for STEM images, adapted for AFM
# --------------------------------------------------------------------
# Requires: Python 3 or higher
#           - numpy, matplotlib, PIL, pickle
# --------------------------------------------------------------------
#
# ---------------------------------------------------------------------
# 
# ---------------------------------------------------------------------
# Version 0 @Cristina_MT (August 29th, 2016)
# ====================================================================

# ====================================================================
# IMPORT REQUIRED MODULES AND FUNCTIONS
# ====================================================================
def WT_multiscale_gauss(mat_img,n_iter):
    from wt_functions import WT_gauss_fixed_2D
    import numpy as np
    print("\n Please wait, computing multiscale smoothing \n")
    a_scale=1.
    WT_aa=np.ones(mat_img.shape)
    
    for nit in range(n_iter):
        a_scale=a_scale+1.
        WT_mod=WT_gauss_fixed_2D(mat_img,a_scale,0)
        WT_mod=WT_mod/np.max(WT_mod.flatten())
        WT_aa=WT_aa*WT_mod
    print('Multiscale smoothing done \n')
    return WT_aa

def img_bin_skeleton(mat_img,n_iter,pth,display_step):
    # ====================================================================
    # IMAGE READING/PROCESSING AND FIBERS SKELETON DETECTION
    # ====================================================================
    
    import numpy as np; import matplotlib.pyplot as plt
    from skimage.morphology import skeletonize
    from fiber_tracking_AFM_v0 import WT_multiscale_gauss
    
    # Image "multiscale smoothing" and binarization based on threshold 
    WT_aa=WT_multiscale_gauss(mat_img,n_iter)
    mask_WT_aa=np.zeros(WT_aa.shape);mask_WT_aa[WT_aa>=pth]=1
    mask_WT_aa[:n_iter,:]=0; mask_WT_aa[-n_iter:,:]=0;mask_WT_aa[:,:n_iter]=0;mask_WT_aa[:,-n_iter:]
    # Note: Borders equal to zero to avoid edge effects due to smoothing

    # Skeletonize and extract coordinates
    sk_WT_aa=skeletonize(mask_WT_aa)
    ys,xs=np.unravel_index(np.nonzero(sk_WT_aa.flatten())[0],mask_WT_aa.shape)
    xs=np.array(xs);ys=np.array(ys)

    # Display this step:
    if display_step==1:
        plt.figure(figsize=(12,8),facecolor='k')
        ax1=plt.subplot(221)
        plt.imshow(mat_img,cmap='inferno',aspect='equal',interpolation='none')
        plt.title('Raw Image (cropped)',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.5);ax1.spines[axis].set_color('white')
        ax2=plt.subplot(222)
        plt.imshow(mask_WT_aa,cmap='gray_r',aspect='equal',interpolation='none')
        plt.title('Binary mask',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(1.5);ax2.spines[axis].set_color('white')
        ax3=plt.subplot(223)
        plt.scatter(xs,ys,edgecolor='k',s=5,c='w')
        plt.title('Skeleton',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(1.5);ax3.spines[axis].set_color('white')
        ax4=plt.subplot(224)
        plt.imshow(mat_img,cmap='inferno',aspect='equal',interpolation='none')
        plt.scatter(xs,ys,edgecolor='none',s=5,c='y')
        plt.title('Superposition',color='white')
        plt.tick_params(axis='x',colors='white');plt.tick_params(axis='y',colors='white')
        for axis in ['top','bottom','left','right']:
            ax4.spines[axis].set_linewidth(1.5);ax4.spines[axis].set_color('white')
        plt.show()
    
    return mask_WT_aa,sk_WT_aa

def fiber_label_junction_det(sk_WT_aa,th_lchain):
    # ====================================================================
    # LABELLING OF FIBERS AND JUNCTION DETECTION
    # --------------------------------------------------------------------
    # xj,yj : x,y coordinates of the junction points
    # xsl_chain, ysl_chain: x,y, coordinates of the selected fibers >= th_lchain
    # idl_chain: 'id' number of the selected fibers. Same length as xsl_chain,ysl_chain
    # ====================================================================
    import numpy as np
    
    ys,xs=np.unravel_index(np.nonzero(sk_WT_aa.flatten())[0],sk_WT_aa.shape)
    xs=np.array(xs);ys=np.array(ys)
    
    # Detection of junction points based on the sum of points over a 3x3 square
    xj=[];yj=[]
    for ij in range(len(xs)):
        mask_jp = sk_WT_aa[ys[ij]-1:ys[ij]+2,xs[ij]-1:xs[ij]+2]
        if np.count_nonzero(mask_jp)>=4:
            xj.append(xs[ij]);yj.append(ys[ij])
            xs[ij]=0;ys[ij]=0

    # Connection of points inside a fiber and labelling of fibers
    # Connection based on distance between points <= 2 pixels
    id_chain=[0];xs_chain=[0];ys_chain=[0];        
    idc=1;ipoint=0;ilook=1;
    print('\n Connecting points and labelling fibers.. \n')
    while ilook==1:
        if np.count_nonzero(xs)<2:
            ilook=0
        dist=np.sqrt((xs-xs[ipoint])**2+(ys-ys[ipoint])**2)
        dist[ipoint]=10000
        if np.min(dist)>0 and np.min(dist)<=2:
            id_chain.append(idc);xs_chain.append(xs[ipoint]);ys_chain.append(ys[ipoint])
            xs[ipoint]=0;ys[ipoint]=0
            ipoint=np.argmin(dist)
        elif np.count_nonzero(xs)>1:
            xs[ipoint]=0;ys[ipoint]=0;
            ipoint=np.nonzero(xs)[0][0]
            idc=id_chain[-1]+1
        else:
            ilook=0;
    xs_chain=np.array(xs_chain);ys_chain=np.array(ys_chain);id_chain=np.array(id_chain)        
    print('Done \n')

    # Discard the chains with less than "th_lchain" pixels
    nchain,ind_rev,lchain=np.unique(id_chain,return_counts=True,return_inverse=True)
    nchain=np.arange(len(lchain));nchain[np.where(lchain<th_lchain)]=0
    selec_chains=nchain[ind_rev];idl_chain=selec_chains[np.nonzero(selec_chains)]
    xsl_chain=xs_chain[np.nonzero(selec_chains)];ysl_chain=ys_chain[np.nonzero(selec_chains)]

    return xsl_chain,ysl_chain,idl_chain,xj,yj

def fiber_local_slope_lin(xsl_chain,ysl_chain,idl_chain):
    # ====================================================================
    # COMPUTATION OF THE LOCAL SLOPE BASED ON A LINEAR FIT
    # --------------------------------------------------------------------
    # xm,ym:  x,y coordinates of the selected points
    # m:      local slope
    # im:     re-labelling of the fibers
    # ====================================================================
    
    import numpy as np;import warnings
    m=[];xm=[];ym=[];im=[]
    isc_new=0
    for isc in np.unique(idl_chain):
        isc_new=isc_new+1
        xp=xsl_chain[np.where(idl_chain==isc)];
        yp=ysl_chain[np.where(idl_chain==isc)];
        for iss in range(1,int(np.floor(len(xp)/5))-1):
            xf=xp[(iss-1)*5:(iss-1)*5+10];yf=yp[(iss-1)*5:(iss-1)*5+10]
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    m0,b0=np.polyfit(xf,yf,1)
                    x1=(np.min(xf));x2=(np.max(xf))
                    x0=np.arange(x1,x2);y0=b0+m0*x0
                    m.extend(m0*np.ones(len(x0)))
                    xm.extend(x0);ym.extend(y0)
                    im.extend(isc_new*np.ones(len(x0)))
                except np.RankWarning:
                    if len(np.unique(xf))>2:
                        m.extend(0);xm.extend(0);ym.extend(0);im.extend(0);
                    else:
                        y0=np.arange(np.min(yf),np.max(yf))
                        x0=np.unique(xf)[0]*np.ones(len(y0))
                        m.extend(m0*np.ones(len(x0)))
                        xm.extend(x0);ym.extend(y0)
                        im.extend(isc*np.ones(len(x0)))
                    
    xm=np.array(xm);ym=np.array(ym);m=np.array(m);im=np.array(im)  
    
    return xm,ym,m,im


def fiber_profile(mat_image,mask_WT_aa,xm,ym,m,im):
    # ====================================================================
    # INTENSITY PROFILE EXTRACTION AND COMPUTATION OF ALL ITS RELARED INFO
    # ====================================================================
    import numpy as np
    
    # Initialisation of the variables
    bleft_min=[];bleft_max=[];bleft_mean=[];bleft_sum=[];bleft_x1=[];bleft_x2=[];bleft_y1=[];bleft_y2=[];
    bright_min=[];bright_max=[];bright_mean=[];bright_sum=[];bright_x1=[];bright_x2=[];bright_y1=[];bright_y2=[];
    fiber_min=[];fiber_max=[];fiber_mean=[];fiber_sum=[];fiber_x1=[];fiber_x2=[];fiber_y1=[];fiber_y2=[];
    fiber_width=[];fiber_xc=[];fiber_yc=[];fiber_slope=[];fiber_number=[];fiber_corr_int_int_mean=[];
    sec_mean_range=[];sec_max_range=[];sec_mean_back_int_int=[];sec_length_back_int_int_mean=[];

    #Profile extraction 
    x_pline=np.arange(0,mat_image.shape[1])
    for icn in np.unique(im):
        # Select points from a given chain
        xcn=xm[np.where(im==icn)];ycn=ym[np.where(im==icn)]; mcn=m[np.where(im==icn)]
        # For each point in the chain:
        for ix in range(len(xcn)):
            if ycn[ix]>=10 and xcn[ix]>=10:     # To avoid border effects
                # Trace a line that passes over the selected point, perpendicular to the local fiber orientation
                y_pline=ycn[ix]-1/mcn[ix]*(-xcn[ix]+x_pline)
                if len(np.where(np.isinf(y_pline))[0])==0:
                    # Define the borders of the line, so it's contained in the image
                    if y_pline[1]>y_pline[-1]:
                        if np.max(y_pline)>=mat_image.shape[0]:
                            ind_y1=np.where(y_pline>=mat_image.shape[0])[0][-1]+1
                        else:ind_y1=1
                        if np.min(y_pline)<0:
                            ind_y2=np.where(y_pline<0)[0][0]-1
                        else:ind_y2=mat_image.shape[1]+1
                    else: 
                        if np.max(y_pline)>=mat_image.shape[0]:
                            ind_y2=np.where(y_pline>=mat_image.shape[0])[0][0]-1
                        else:ind_y2=mat_image.shape[1]-1
                        if np.min(y_pline)<0:
                            ind_y1=np.where(y_pline<0)[0][-1]-1
                        else:ind_y1=1
            
                    # Define the x,y coordinates for the profile to be extracted
                    if ind_y2-ind_y1>3:
                        x_profile=[int(x) for x in x_pline[ind_y1:ind_y2]]
                        y_profile=[int(x) for x in y_pline[ind_y1:ind_y2]]
                    elif mcn[ix]<1e-10:
                        y_profile=np.arange(1,mat_image.shape[0]-1)
                        x_profile=np.array([int(x) for x in xcn[ix]*np.ones(len(y_profile))])
                    x_profile=np.array(x_profile);y_profile=np.array(y_profile)
            
                    # If there are more than 5 pixels in the profile, then continue
                    if len(x_profile)>=5:
                        # Extract profile (both binary, and intensity)
                        mask_profile=np.array(mask_WT_aa[y_profile,x_profile])
                        int_profile=np.array(mat_image[y_profile,x_profile])
                        # Find the position of the selected point in the profile
                        ind_max=np.where(np.abs(x_profile-xcn[ix])<1)[0][0]
                
                        # Assure that the selected point corresponds to a detected fiber
                        if mask_profile[ind_max]==1:
                            # Detect the left and right borders (if any)
                            if len(np.where(mask_profile[:ind_max]==0)[0])>0:
                                b_left=np.where(mask_profile[:ind_max]==0)[0][-1]-1
                            else: b_left=0;
                            if len(np.where(mask_profile[ind_max:]==0)[0])>0:
                                b_right=np.where(mask_profile[ind_max:]==0)[0][0]+ind_max+1
                            else: b_right=len(int_profile);
                            # Assure that there are enough background pixels on the left/right side
                            if b_right-b_left>4 and np.abs(b_right-b_left)<=300 and b_left>10 and b_right<len(int_profile)-10:
                                # Assure that the detected left/right borders are about the same distance from the selected point, difference +-20%
                                if np.abs(x_profile[ind_max]-x_profile[b_left])<=1.5*np.abs(x_profile[ind_max]-x_profile[b_right]) and np.abs(x_profile[ind_max]-x_profile[b_left])>=0.5*np.abs(x_profile[ind_max]-x_profile[b_right]) :
                            
                                    # Section the profile in left/right background, and fiber sections
                                    back_left=int_profile[b_left-10:b_left]
                                    back_right=int_profile[b_right:b_right+10]
                                    fiber_sec=int_profile[b_left+2:b_right-2]
                        
                                    # Assure that the background is homogeneous, I(left)~ I(right) , e= +-10%
                                    
                                
                                    # Compute the distance between points in the profile: step in pixels
                                    dy=y_profile[b_left+3:b_right-1]-y_profile[b_left+2:b_right-2];
                                    dx=x_profile[b_left+3:b_right-1]-x_profile[b_left+2:b_right-2];
                                    dl=np.sqrt(dy**2+dx**2)
                                
                                    # Info computed for the background section on the left of the fiber
                                    bleft_min.append(np.min(back_left));bleft_max.append(np.max(back_left))
                                    bleft_mean.append(np.mean(back_left));bleft_sum.append(np.sum(back_left))
                                    bleft_x1.append(x_profile[b_left-10]);bleft_x2.append(x_profile[b_left])
                                    bleft_y1.append(y_profile[b_left-10]);bleft_y2.append(y_profile[b_left])
                                    # Info for the background on the right side
                                    bright_min.append(np.min(back_right));bright_max.append(np.max(back_right))
                                    bright_mean.append(np.mean(back_right));bright_sum.append(np.sum(back_right))
                                    bright_x1.append(x_profile[b_right]);bright_x2.append(x_profile[b_right+10])
                                    bright_y1.append(y_profile[b_right]);bright_y2.append(y_profile[b_right+10])
                                    # Info for the fiber
                                    fiber_min.append(np.min(fiber_sec));fiber_max.append(np.max(fiber_sec))
                                    fiber_mean.append(np.mean(fiber_sec));fiber_sum.append(np.sum(fiber_sec*dl))
                                    fiber_x1.append(x_profile[b_left+2]);fiber_x2.append(x_profile[b_right-2])
                                    fiber_y1.append(y_profile[b_left+2]);fiber_y2.append(y_profile[b_right-2])
                                    fiber_width.append(np.sqrt((fiber_x2[-1]-fiber_x1[-1])**2+(fiber_y2[-1]-fiber_y1[-1])**2))
                                    fiber_xc.append(x_profile[ind_max]);fiber_yc.append(y_profile[ind_max])
                                    fiber_slope.append(mcn[ix]);fiber_number.append(icn)
                                    # Info for the full profile
                                    sec_mean_range.append(fiber_mean[-1]-np.mean([bleft_mean[-1],bright_mean[-1]]))
                                    sec_max_range.append(fiber_max[-1]-np.mean([bleft_mean[-1],bright_mean[-1]]))
                                    # Integrated Intensity related values
                                    sec_mean_back_int_int.append(np.mean([bleft_sum[-1],bright_sum[-1]]))
                                    sec_length_back_int_int_mean.append(np.mean([bleft_mean[-1],bright_mean[-1]])*fiber_width[-1])
                                    fiber_corr_int_int_mean.append(fiber_sum[-1]-sec_length_back_int_int_mean[-1])
                                    
    Fiber_info=[fiber_min,fiber_max,fiber_mean,fiber_x1,fiber_x2,fiber_y1,fiber_y2,fiber_xc,fiber_yc,fiber_slope,fiber_number,fiber_width,fiber_sum,fiber_corr_int_int_mean]
    
    Back_info=[bleft_min,bleft_max,bleft_mean,bleft_sum,bleft_x1,bleft_x2,bleft_y1,bleft_y2,bright_min,bright_max,bright_mean,bright_sum,bright_x1,bright_x2,bright_y1,bright_y2]
    
    Profile_info=[sec_mean_range,sec_max_range,sec_mean_back_int_int,sec_length_back_int_int_mean]
    
    return Fiber_info,Back_info,Profile_info
    
    

# # ====================================================================
# # Save information and figure with the detected points (fiber center, left/right border)
# # ====================================================================

# fig2=plt.figure(figsize=(12,4),facecolor='w')
# plt.subplot(121)
# plt.imshow(mat_cropped_image,cmap='gray',interpolation='none')
# plt.scatter(fiber_x1,fiber_y1,c='y',edgecolor='none',s=2)
# plt.scatter(fiber_x2,fiber_y2,c='orange',edgecolor='none',s=2)
# plt.scatter(xj,yj,c='m',edgecolor='k',s=5)
# plt.subplot(122)
# plt.scatter(fiber_width_nm,fiber_corr_int_int_mean,s=2,c='k',edgecolor='none')
# plt.xlabel('Fiber width (nm)')
# plt.ylabel('Integrated Intensity')
# plt.show()

