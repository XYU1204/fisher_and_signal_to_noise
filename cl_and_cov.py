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

def normalizing(cls, ell):
    """Add normalizing factor (proportional to angular mode l) for signal, input cls and ell must be the same length"""
    N=np.size(cls)
    clsn=np.zeros(N)
    for i in range(N):
        clsn[i]=ell[i]*(ell[i]+1)*cls[i]/(2*np.pi)
    return clsn

# create bins of dndz using A.equal redshift bins, and B.equal galaxy number bins
def sliced_equal_z(dndz, nbins_z):
    """create bins of dndz using equal redshift bins. The output will be a dictionary with the bin numbers as keys, and number density versus redshifts in that bin as values"""
    n = nbins_z
    min_z = dndz[ :, 0].min()
    inc = (dndz[ :, 0].max()-min_z)/n
    #this calculates the increment in refshifts for each bin
    dndz_cut = {}
    #create an empty dictionary for sliced galaxy number density distribution
    for i in range(n):
        dndz_cut["bin_{}".format(i+1)]= dndz[np.logical_and(dndz[:,0]>=inc*i+min_z, dndz[:,0]<inc*(i+1)+min_z)]
    #add to the last bin
    if dndz_cut["bin_{}".format(i+1)][-1][0] < dndz[ :, 0].max():
        new = np.vstack([dndz_cut["bin_{}".format(i+1)], dndz[-1, :]])
        dndz_cut["bin_{}".format(i+1)] = new
    return dndz_cut

def sliced_equal_n(dndz, nbins_z):
    """create bins of dndz using equal galaxy number bins. The output will be a dictionary with the bin numbers as keys, and number density versus redshifts in that bin as values"""
    n = nbins_z
    cdf = np.array([np.sum(dndz[:i+1,1]) for i in range(len(dndz[:,1]))])
    #find the cumulative density function
    inc = cdf[-1]/n
    #cdf[-1] gives the total number of galaxies in the sky. each bin has total/n galaxies
    dndz_cut = {}
    #create an empty dictionary for sliced galaxy number density distribution
    for i in range(n):
        dndz_cut["bin_{}".format(i+1)] = dndz[np.logical_and(cdf>inc*i, cdf<=inc*(i+1))]
    #new = np.vstack([dndz_cut["bin_{}".format(i+1)], dndz[-1, :]])
    #dndz_cut["bin_{}".format(i+1)] = new
    return dndz_cut

def getCl(cosmo, dndz_sliced, ell):
    """This function calculats auto- and cross-power spectra for given sliced galaxy-redshift distribution."""
    n = len(dndz_sliced);
    lens = [[]]*n
    for i, dndz in enumerate(dndz_sliced.values()):
        lens[i] = ccl.WeakLensingTracer(cosmo, dndz=(dndz[:,0], dndz[:,1]))
    #create an empty n-by-n array that stores the lensing cross power spectrum between the ith and jth bin
    cl=[]
    for j in range(n):
        cl.extend(ccl.angular_cl(cosmo, lens[j], lens[j+i], ell) for i in range(n-j))
    cl_arr = np.array(cl)
    return cl_arr

def num_den(dndz_sliced, numdenPerStr):
    numden = np.array([np.sum(dndz_sliced["bin_{}".format(i+1)][:, 1]) for i in range(len(dndz_sliced))])
    new = numden/np.sum(numden)
    new = numdenPerStr*new
    return new

def getCovMat(fsky, n_bins, cl, dndz_sliced, numdenPerStr, ell, shape_noise=None, shot_noise=False):
    """dndz1 is the original galaxy redshift distribution, whereas dndz is the sliced distribution in different redshift bins"""
    l = []
    for j in range(n_bins):
        l.extend([[j, j+i] for i in range(n_bins-j)])
    cl_bin = np.vstack((ell, cl)).T
    #for number density, divide galaxy numbers into each tomographic bin
    numden = num_den(dndz_sliced, numdenPerStr)
    cov_arr = np.array(multi_bin_cov(fsky, cl_bin, np.array(l), numden, shape_noise, shot_noise))
    return cov_arr

def getDataArray(n_bins, bin_type, cosmo, ell, dndz, numdenPerStr, fsky, shape_noise=None, shot_noise=False):
    """input cosmo for cosmological object, dndz for galaxy redshift distribution, rbins for number of tomographic bins in redshift, rbin_type for how to divide the tomographic bins in redshifts: input string z for bins of equal redshifts, and n for bins of equal galaxy number. This function will return the covariance array, cl data array, and the redshift range for each tomographic bins"""
    if bin_type == 'z':
        dndz_cut = sliced_equal_z(dndz, n_bins)
    elif bin_type == 'n':
        dndz_cut = sliced_equal_n(dndz, n_bins)
    else:
        print("Enter 'z' for bins of equal redshifts, and 'n' for bins of equal galaxy number")
        return(None)
    cl_arr = getCl(dndz_sliced=dndz_cut, cosmo=cosmo, ell=ell) 
    cov_arr = getCovMat(fsky, n_bins, cl_arr, dndz_cut, numdenPerStr, ell, shape_noise, shot_noise)
    redshifts=[]
    for x in dndz_cut.values():
        redshifts.append(x[1][0])
    redshifts.append(x[-1][0])
    return cov_arr, cl_arr, redshifts, dndz_cut