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

def GalaxyDistr(zi, zf, n_eff, z0, alpha):
    """this function creates a galaxy redshift distribution for given inputs: starting redshift zi, and final redshift zf, source effective number density n_eff per square arcminute, and parameters z0 and alpha. The output array has redshift in the first column and normalized galaxy number density in the second column"""
    dndz = np.ones([400, 2])
    dndz[:,0] = np.linspace(zi, zf, 400)
    #create the redshift array
    dndz[:,1] = dndz[:,0]**2.0 * np.exp(-(dndz[:,0]/z0)**alpha)
    #conventional fitting formula for galaxy number density, from Eq (5), page 47, section D1.1 of LSST science book 2018 (1809.01669)
    dndz[:,1] = dndz[:,1]/sum(dndz[:,1])
    #normalize galaxy number density function
    dndz[:,1] = dndz[:,1] * n_eff * 4*np.pi*((360 / (2*np.pi))*60)**2
    #multiplying by n_eff times the number of square arc minutes over the entire sky
    dndz=dndz[dndz[:,0].argsort()]
    return dndz

def Numberdensity(i_lim, f_mask):
    """Produce number of galaxy per arcminute in the sky with luminosity less than i band limit. Input i_lim for i band limit and f_mask for reduction factors for masks due to various image defects, bright sky, and etc"""
    N = 42.9*(1-f_mask)*10**(0.359*(i_lim-25))
    return N


def PlotGalaxyDistr(dndz):
    """Plot galaxy distribution for given galaxy distribution array as a function of redshifts"""
    plt.plot(dndz[:,0], dndz[:,1])
    plt.xlabel(r'$z$',fontsize=20)
    plt.ylabel(r'$dN/dz$',fontsize=20)
    plt.title("galaxy number density distribution by redshift")
    plt.show()

def normalizing(cls):
    """Add normalizing factor (proportional to angular mode l) for signal"""
    N=np.size(cls)
    clsn=np.zeros(N)
    for i in range(N):
        clsn[i]=i*(i+1)*cls[i]/(2*np.pi)
    return clsn

def CreatelensZ(n_bins, dndz, cosmo):
    """Create lensing object for a given galaxy-number-density vs.redshifts distribution by diving the distribution into bins of equal redshift range. Input n_bins for the number of bins to be divided into, dndz for the galaxy distribution, and cosmo for cosmology object. Return an array that stores lensing object, and another array that stores the starting and ending redshifts for each tomographic bin."""
    n = n_bins
    inc = (dndz[ :, 0].max()-dndz[ :, 0].min())/n
    #this calculates the increment in refshifts for each bin
    redshifts = [dndz[0][0]] 
    #create a list that display the starting and ending redshifts for each bin
    lenses = [[]]*n;
    #create an array of lensing objects
    for i in range(n):
        dndz_cut = dndz[np.logical_and(dndz[:,0]>inc*i, dndz[:,0]<inc*(i+1))]
        #the ith bin galaxy distribution is cut from the total galaxy distribution array by choosing galaxies with redshift range of (i*inc, (i+1)*inc)
        lenses[i] = ccl.WeakLensingTracer(cosmo, dndz=(dndz_cut[:,0], dndz_cut[:,1]))
        redshifts.append(dndz[0][0]+inc*(i+1))
    return lenses, redshifts

def cdf(pdf):
    """calculating cumultative density function for a given probability density function"""
    cf = np.zeros(len(pdf));
    for i in range(len(pdf)):
        cf[i] = sum(pdf[:i])
    return cf

def CreatelensN(n_bins, dndz, cosmo):
    """Create lensing objects for a given galaxy-number-density vs.redshifts distribution by diving the distribution into bins of equal galaxy number. Input n_bins for the number of bins to be divided into, dndz for the galaxy distribution, and cosmo for cosmology object, Return an array that stores lensing object, and another array that stores the starting and ending redshifts for each tomographic bin."""
    n = n_bins
    pdf = dndz[:, 1]
    cf = cdf(pdf)
    #find the cumulative density function
    inc = cf[-1]/n
    #cd[-1] gives the total number of galaxies in the sky. each bin has total/n galaxies
    lenses = [[]]*n;
    #create an array of lensing objects
    redshifts = [dndz[0][0]];
    #create a list that display the starting and ending redshifts for each bin
    for i in range(n):
        dndz_cut = dndz[np.logical_and(cf>inc*i, cf<inc*(i+1))]
        #the ith bin galaxy distribution is cut from the total galaxy distribution array by chopping the distribution into equal galaxy number bins
        lenses[i] = ccl.WeakLensingTracer(cosmo, dndz=(dndz_cut[:,0], dndz_cut[:,1]))
        redshifts.append(dndz_cut[-1,0])
        #the ending redshift for the ith bin is added to the redshift array
    return lenses, redshifts

def getLensingCRmatrix(lenses, ell, cosmo):
    """This function calculats auto- and cross-power spectra for given lensing objects in tomographic bins. Input the lensing objects and ls' to integrate over. It returns a matrix that stores the cross power spectra between different bins"""
    n = len(lenses);
    lensCL = np.zeros([n, n, len(ell)]);
    #create an empty n-by-n array that stores the lensing cross power spectrum between the ith and jth bin
    for i in range(n):
        lensCL[i, i, :] = ccl.angular_cl(cosmo, lenses[i], lenses[i], ell);
    for j in range(n-1):
        for k in range (j+1, n):
            lensCL[j, k, :] = ccl.angular_cl(cosmo, lenses[j], lenses[k], ell)
            lensCL[k, j, :] = lensCL[j, k, :]
    return lensCL

def PlotLensingAngPowerSpec(ell, lensCL, redshifts):
    """plot lensing angular auto power spectra for given spectra matrix, redshifts limits for tomographic bins, and angular modes ls' to intergrate over"""
    plt.figure(figsize=(16,9))
    mpl.rcParams.update({'font.size': 20})
    for i in range(len(redshifts)-1):
        plt.loglog(ell, normalizing(lensCL[i, i, :]), label= "Z=%.2f-%.2f"%(redshifts[i],redshifts[i+1]))
    plt.legend(loc='lower right', fontsize=18)
    plt.title('lensing angular power spectrum binned by redshift')
    plt.xlabel(r'$\ell$')
    plt.xlim(ell.min(), ell.max()*2)
    plt.ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi$')
    plt.show() 

def PlotLensingAngCRPowerSpec(n, ell, lensCL, redshifts):
    """plot lensing angular cross power spectra for given spectra matrix, redshifts limits for tomographic bins, and angular modes ls' to intergrate over"""
    plt.figure(figsize=(16,9))
    mpl.rcParams.update({'font.size': 20})
    for i in range(n-1):
        for j in range (i+1, n):
            plt.loglog(ell, normalizing(lensCL[i, j, :]), label= "Z=%.2f-%.2f"%(redshifts[i],redshifts[i+1])+ " and %.2f-%.2f"%(redshifts[j],redshifts[j+1]))
    plt.legend(loc='lower right', fontsize=18)
    plt.title('lensing cross power spectrum binned by redshift')
    plt.xlabel(r'$\ell$')
    plt.xlim(ell.min(), ell.max()*2)
    plt.ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi$')
    plt.show()

def getCovMat(fsky, n_bins, ClMat, ell, numdenPerStr):
    l = []
    for j in range(n_bins):
        l.extend([[j, j+i] for i in range(n_bins-j)])
    cl=[]
    for j in range(n_bins):
        cl.extend(ClMat[j, j+i] for i in range(n_bins-j))
    cl_bin = np.vstack((ell, cl)).T
    cl_arr = np.array(cl)
    numden = [numdenPerStr]*n_bins
    cov_arr = np.array(multi_bin_cov(fsky, cl_bin, np.array(l), numden))
    return cov_arr, cl_arr

def BoundloUp(i, binl, elll):
    """for a givin binning central ls and the index of bin, this function gives the upper and lower ls of each bin.
    the lowest is set to be 2 since we found that the data array has all 0 for l=0 and l=1"""
    if i == 0:
        lo = 2;
        up = np.round((binl[1]+binl[0])/2);
    elif i == len(binl)-1:
        lo = np.round((binl[-2]+binl[-1])/2)
        up = elll[-1]
    else:
        lo = np.round((binl[i-1]+binl[i])/2)
        up = np.round((binl[i]+binl[i+1])/2)
    return lo, up

def num_cov(bins_in_z):
    return bins_in_z + comb(bins_in_z, 2)

def binnedCl(bins_in_z, binl, cl_o, elll):
    """bin the cl signals according to array of ls"""
    Cl_binned = np.zeros([int(num_cov(bins_in_z)), int(len(binl))])
    for i in range(len(binl)):
        Cl_binned[:, i] = np.mean(cl_o[:, int(BoundloUp(i, binl, elll)[0]):int(BoundloUp(i, binl, elll)[1])], axis=1)
    return Cl_binned

def binnedCov(bins_in_z, binl, cov_o, elll):
    """bin the covariance according to array of binning ls"""
    Cov_binned = np.zeros([ int(len(binl)), int(num_cov(bins_in_z)), int(num_cov(bins_in_z))])
    for i in range(len(binl)):
        lo, up = BoundloUp(i, binl, elll)
        Cov_binned[i, :, :] = np.sum(cov_o[int(lo):int(up), :, :], axis=0)/(up-lo)**2
    return Cov_binned

def SignalToNoise(bins_in_z, binl, cl, cov, ell):
    cl_binned = binnedCl(bins_in_z, binl, cl, ell)
    cov_binned = binnedCov(bins_in_z, binl, cov, ell)
    stn_sq = 0
    for i in range(len(binl)):
        stn_sq = stn_sq + np.matmul(cl_binned[:, i].T, 
                                    np.matmul(np.linalg.inv(cov_binned[i,:,:]), cl_binned[:, i]))
    stn = np.sqrt(stn_sq)
    return stn

def SigToNos(rbins, rbin_type, dndz1):
    """input rbins for number of tomographic bins in redshift, 
    rbin_type for how to divide the tomographic bins in redshifts: input string Z for bins of equal redshifts, 
    and N for bins of equal galaxy number"""
    elll = np.arange(0, 10001)
    if rbin_type == "z":
        lens, redshifts = CreatelensZ(n_bins = rbins, dndz = dndz1, cosmo = cosmo)
    else:
        lens, redshifts = CreatelensN(n_bins = rbins, dndz = dndz1, cosmo = cosmo)
    Cl = getLensingCRmatrix(lenses = lens, ell = elll, cosmo = cosmo)
    cov_arr, cl_arr = getCovMat(fsky=0.4, n_bins=rbins, ClMat=Cl, ell=elll, numdenPerStr = num_den_per_str)
    stnsq = 0
    for i in range(2,10001):
        stnsq = stnsq + np.matmul(cl_arr[:, i].T, np.matmul(np.linalg.inv(cov_arr[i,:,:]), cl_arr[:, i]))
        
    StoN = np.sqrt(stnsq)
    return StoN, redshifts

def getDataArray(rbins, rbin_type, cosmo, dndz, ell, fsky, num_den_per_str):
    """input cosmo for cosmological object, dndz for galaxy redshift distribution, rbins for number of tomographic 
    bins in redshift, rbin_type for how to divide the tomographic bins in redshifts: input string z for bins of equal redshifts,
    and n for bins of equal galaxy number. This function will return the covariance array, cl data array, and the redshift
    range for each tomographic bins"""
    if rbin_type == "z":
        lens, redshifts = CreatelensZ(n_bins = rbins, dndz = dndz, cosmo = cosmo)
    else:
        lens, redshifts = CreatelensN(n_bins = rbins, dndz = dndz, cosmo = cosmo)
    Cl = getLensingCRmatrix(lenses = lens, ell = ell, cosmo = cosmo)
    cov_arr, cl_arr = getCovMat(fsky=fsky, n_bins=rbins, ClMat=Cl, ell=ell, numdenPerStr = num_den_per_str)
    return cov_arr, cl_arr, redshifts

def SignalToNoise_o(cl, cov):
    """calculate the signal to noise for the given cl signal, and covariance array, without binning in l"""
    stn_sq = 0
    for i in range(cl.shape[1]):
        stn_sq = stn_sq + np.matmul(cl[:, i].T, 
                                    np.matmul(np.linalg.inv(cov[i,:,:]), cl[:, i]))
    stn = np.sqrt(stn_sq)
    return stn