"""
Created on Wed Sep 20 12:07:38 2017

@author: Cristina MT
"""

import numpy as np
import matplotlib.pyplot as plt

def find_maxpeak(signal, dpix):
    ind_max = np.arange(len(signal), step = dpix)[np.r_[True, signal[1::dpix]>signal[:-1:dpix]] & np.r_[signal[:-1:dpix] > signal[1::dpix], True]]
    return ind_max
    
def find_minpeak(signal, dpix):
    ind_min = np.arange(len(signal), step = dpix)[np.r_[True, signal[1::dpix]<signal[:-1:dpix]] & np.r_[signal[:-1:dpix] < signal[1::dpix], True]]
    return ind_min


filename = "filename_cross_sections.txt"

raw_data = np.loadtxt(filename, skiprows = 3)
xdata_nm = raw_data[:,0::2]*1e9
ydata_nm = raw_data[:,1::2]*1e9

ydata_nm[ydata_nm == 0] = float('NaN')
xdata_nm[xdata_nm == 0] = float('NaN')


height_diff = []
height_corr = []
height_nocorr = []

for nprof in range(xdata_nm.shape[1]):
    ind_min = find_minpeak(ydata_nm[:,nprof],1)
    ind_max = find_maxpeak(ydata_nm[:,nprof],1)
    ydata_corr = ydata_nm[:,nprof] - np.mean(ydata_nm[ind_min, nprof])
    
    for ipeak in ind_max:
        try: min1 = ydata_nm[ind_min[np.where(ind_min<ipeak)[0][-1]],nprof]
        except IndexError: min1 = 0
        try: min2 = ydata_nm[ind_min[np.where(ind_min>ipeak)[0][0]],nprof]
        except IndexError: min2 = 0
        height_diff.append(ydata_nm[ipeak,nprof]-np.mean([min1,min2]))
        height_corr.append(ydata_corr[ipeak])
        height_nocorr.append(ydata_nm[ipeak, nprof])
        
plt.figure()
plt.hist(height_diff, bins = 20, label = 'diff', alpha = 0.5)
plt.hist(height_corr, bins = 20, label = 'corr', alpha = 0.5)
plt.hist(height_nocorr, bins = 20, label = 'raw', alpha = 0.5)
plt.legend()
    
