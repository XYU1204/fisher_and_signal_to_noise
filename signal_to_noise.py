import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.ticker as mticker
from scipy.special import comb
from CovMat import *
from cl_and_cov import *


def BoundloUp(i, binl, ell):
    """for a givin binning central ls and the index of bin, this function gives the upper and lower ls of each bin.
    the lowest is set to be 2 since we found that the data array has all 0 for l=0 and l=1"""
    if i == 0:
        lo = 2;
        up = np.round((binl[1]+binl[0])/2);
    elif i == len(binl)-1:
        lo = np.round((binl[-2]+binl[-1])/2)
        up = ell[-1]
    else:
        lo = np.round((binl[i-1]+binl[i])/2)
        up = np.round((binl[i]+binl[i+1])/2)
    return lo, up

def num_cov(bins_in_z):
    return bins_in_z + comb(bins_in_z, 2)

def binnedCl(bins_in_z, binl, cl_o, ell):
    """bin the cl signals according to array of ls"""
    Cl_binned = np.zeros([int(num_cov(bins_in_z)), int(len(binl))])
    for i in range(len(binl)):
        Cl_binned[:, i] = np.mean(cl_o[:, int(BoundloUp(i, binl, ell)[0]):int(BoundloUp(i, binl, ell)[1])], axis=1)
    return Cl_binned

def binnedCov(bins_in_z, binl, cov_o, ell):
    """bin the covariance according to array of binning ls"""
    Cov_binned = np.zeros([ int(len(binl)), int(num_cov(bins_in_z)), int(num_cov(bins_in_z))])
    for i in range(len(binl)):
        lo, up = BoundloUp(i, binl, ell)
        Cov_binned[i, :, :] = np.sum(cov_o[int(lo):int(up), :, :], axis=0)/(up-lo)**2
    return Cov_binned

def SignalToNoise(bins_in_z, binl, cl, cov, ell):
    """calculate signal to noise for givien redshift bins, bins in l, data vector and covariance matrix"""
    cl_binned = binnedCl(bins_in_z, binl, cl, ell)
    cov_binned = binnedCov(bins_in_z, binl, cov, ell)
    stn_sq = 0
    for i in range(len(binl)):
        stn_sq = stn_sq + np.matmul(cl_binned[:, i].T, 
                                    np.matmul(np.linalg.inv(cov_binned[i,:,:]), cl_binned[:, i]))
    stn = np.sqrt(stn_sq)
    return stn

def SignalToNoise_o(cl, cov):
    """calculate the signal to noise for the given cl signal, and covariance array, without binning in l"""
    stn_sq = 0
    for i in range(cl.shape[1]):
        stn_sq = stn_sq + np.matmul(cl[:, i].T, 
                                    np.matmul(np.linalg.inv(cov[i,:,:]), cl[:, i]))
    stn = np.sqrt(stn_sq)
    return stn

