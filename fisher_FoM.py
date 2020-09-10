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
from signal_to_noise import *
from initialize import *
import numdifftools as nd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

param_order = ['Omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'Omega_b', 'h']
param_labels = [r'$\Omega_m$', r'$\sigma_8$', r'$n_s$', r'$w_0$', r'$w_a$', r'$\Omega_b$', r'$h$']

def getC_ellOfSigma8(sigma8, dndz_sliced, ell):
    """create cl w.r.t to Sigma8, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfOmegab(Omega_b, dndz_sliced, ell):
    """create cl w.r.t to Omega_baryon, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfh(h, dndz_sliced, ell):
    """create cl w.r.t to hubble's constant, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfn_s(n_s, dndz_sliced, ell):
    """create cl w.r.t to n_s constant, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfOmegam(Omega_m, dndz_sliced, ell):
    """create cl w.r.t to Omega_m (matter density component) constant, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfw0(w_0, dndz_sliced, ell):
    """create cl w.r.t to w_0 constant, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, w0=w_0, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl

def getC_ellOfwa(w_a, dndz_sliced, ell):
    """create cl w.r.t to w_a constant, while keeping other parameters fixed"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, w0=w_a, transfer_function= transfer_function)
    cl = getCl(ell = ell, cosmo = cosmo, dndz_sliced = dndz_sliced)
    return cl
        
def fisher_matrix(covariance, dndz_sliced, ell):
    """calculate fisher matrix for 7 cosmological parameters ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h'] for given covariance matrix of lensing signal, and galaxy-redshift distribution in divided redshift bins."""
    
    funcs = {
    'Omega_b': getC_ellOfOmegab,
    'sigma_8': getC_ellOfSigma8,
    'h': getC_ellOfh,
    'n_s': getC_ellOfn_s,
    'Omega_m': getC_ellOfOmegam,
    'w_0': getC_ellOfw0,
    'w_a': getC_ellOfwa}
    vals = {
    'sigma_8': sigma8, 
    'Omega_b': Omega_b, 
    'h': h, 
    'n_s': n_s, 
    'Omega_m': Omega_m,
    'w_0': w_0,
    'w_a': w_a}
    derivs_sig = {}
    for var in funcs.keys():
        if vals[var] == 0:
            f = nd.Derivative(funcs[var], full_output=True, step=0.05)
        else:
            f = nd.Derivative(funcs[var], full_output=True, step=float(vals[var])/10)
        val, info = f(vals[var], dndz_sliced, ell)
        derivs_sig[var] = val
    param_order = ['Omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'Omega_b', 'h']
    fisher = np.zeros((7,7))
    for i, var1 in enumerate(param_order):
        for j, var2 in enumerate(param_order):
            f = []
            for l in range(derivs_sig[var1].shape[1]):
                res = derivs_sig[var1][:, l].T @ np.linalg.inv(covariance[l]) @ derivs_sig[var2][:, l]
                f.append(res)
            fisher[i][j] = sum(f)
    return fisher

def showParameterOrder():
    """print parameter order for fisher matrix"""
    param_order = ['Omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'Omega_b', 'h']
    print(param_order)
    return(None)

def cov_cosmo_para(fisher):
    """calculate the covariance matrix among 7 cosmological parameters for given fisher matrix of the 7 cosmological parameters"""
    return np.linalg.inv(fisher)

#we want to know the correlation matrix of cosmological parameters
def corr_cosmo_para(cov):
    """calculate the correlation matrix among 7 cosmological parameters for given covariance matrix of the 7 cosmological parameters"""
    dim = cov.shape
    correlation = np.zeros(dim)
    for j in range(dim[0]):
        for k in range(dim[1]):
            correlation[j, k] = cov[j,k]/np.sqrt(cov[j,j]*cov[k,k])
    return correlation

def plot_corr(correlation):
    param_labels = [r'$\Omega_m$', r'$\sigma_8$', r'$n_s$', r'$w_0$', r'$w_a$', r'$\Omega_b$', r'$h$']
    plt.matshow(correlation)
    plt.colorbar()
    plt.xticks(np.arange(7), param_labels)
    plt.yticks(np.arange(7), param_labels)
    for i in range(7):
        for j in range(7):
            c = correlation[j,i]
            plt.text(i, j, "%.4f"%(c), va='center', ha='center', fontsize=6)
    return(None)

def subcov(cov, para_a, para_b):
    """find the sub covariance matrix between two parameters given the full covariance matrix"""
    param_order = ['Omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'Omega_b', 'h']
    a = param_order.index(para_a)
    b = param_order.index(para_b)
    ixgrid = np.ix_([a, b], [a, b])
    sub = cov[ixgrid]
    return sub

def ellipse_parameter(cov):
    """
    find the parameters for confidence ellipse for given covariance matrix
    """
    Sxx=cov[0][0]
    Syy=cov[1][1]
    Sxy=cov[0][1]
    #find ellipse parameters
    a = np.sqrt(0.5*(Sxx + Syy) + np.sqrt(Sxy**2 + 0.25*(Sxx-Syy)**2))
    b = np.sqrt(0.5*(Sxx + Syy) - np.sqrt(Sxy**2 + 0.25*(Sxx-Syy)**2))
    #find ellipse angle by finding the angle of the major axis, which is the same as the eigenvector of the matrix with larger eigenvalue
    eig = np.linalg.eig(np.linalg.inv(cov))
    large_eigenv = eig[1][np.argmax(eig[0])]
    theta = np.arctan2(large_eigenv[1],large_eigenv[0])
    
    return a,b,theta

def confidence_ellipse(mu, a, b, theta, ax, n_std, facecolor='none', **kwargs):
    """add confidence ellipse in the plot for given ellipse parameters and confidence interval"""
    prefactor = {1:1.52, 2:2.48, 3:3.44}
    pre = prefactor[n_std]
    a*= pre
    b*= pre
    ellipse = Ellipse(mu, width=a * 2, height=b * 2, angle=-np.degrees(theta),
                      facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)

def FoM(cov):
    """calculate figure of merit between two comological parameters given a 2 by 2 covariance matrix between the two parameters"""
    return 1/(4*np.sqrt(np.linalg.det(cov)))

def plotConfidenceEllipse(cov, para_a, para_b):
    """plot three confidence ellipses between two cosomlogical parameters for given covariance matrix of all cosmological parameters"""
    vals = {
    'sigma_8': sigma8, 
    'Omega_b': Omega_b, 
    'h': h, 
    'n_s': n_s, 
    'Omega_m': Omega_m,
    'w_0': w_0,
    'w_a': w_a}
    mu = (vals[para_a], vals[para_b])
    sub = subcov(cov, para_a, para_b)
    a, b, theta = ellipse_parameter(sub)
    F_o_M = FoM(sub)
    print("{} = ".format(para_a) + str(mu[0]) + " +- " + str(a) + "\n")
    print("{} = ".format(para_b) + str(mu[1]) + " +- " + str(b) + "\n")
    print("Figure of Merit for {0} and {1} is ".format(para_a, para_b) + str(F_o_M) +'\n')
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    
    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    if (para_a == 'Omega_m') or (para_a == 'Omega_b') or (para_a == 'sigma_8'):
        para_a = "\\"+para_a
    if (para_b == 'Omega_m') or (para_b == 'Omega_b') or (para_b == 'sigma_8'):
        para_b = "\\"+para_b
    ax_nstd.set_title(r'Confidence Ellipse for ${0}$ and ${1}$'.format(para_a, para_b))
    ax_nstd.set_xlabel(r'${0}$'.format(para_a))
    ax_nstd.set_ylabel(r'${0}$'.format(para_b))
    ax_nstd.legend(loc='upper left')
    plt.show()
    return F_o_M

def plotConfidenceEllipse_sub(sub, para_a, para_b):
    """plot three confidence ellipses between two cosomlogical parameters for given covariance matrix of the two"""
    vals = {
    'sigma_8': sigma8, 
    'Omega_b': Omega_b, 
    'h': h, 
    'n_s': n_s, 
    'Omega_m': Omega_m,
    'w_0': w_0,
    'w_a': w_a}
    mu = (vals[para_a], vals[para_b])
    a, b, theta = ellipse_parameter(sub)
    F_o_M = FoM(sub)
    print("{} = ".format(para_a) + str(mu[0]) + " +- " + str(a) + "\n")
    print("{} = ".format(para_b) + str(mu[1]) + " +- " + str(b) + "\n")
    print("Figure of Merit for {0} and {1} is ".format(para_a, para_b) + str(F_o_M) +'\n')
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(mu, a, b, theta, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    
    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    if (para_a == 'Omega_m') or (para_a == 'Omega_b') or (para_a == 'sigma_8'):
        para_a = "\\"+para_a
    if (para_b == 'Omega_m') or (para_b == 'Omega_b') or (para_b == 'sigma_8'):
        para_b = "\\"+para_b
    ax_nstd.set_title(r'Confidence Ellipse for ${0}$ and ${1}$'.format(para_a, para_b))
    ax_nstd.set_xlabel(r'${0}$'.format(para_a))
    ax_nstd.set_ylabel(r'${0}$'.format(para_b))
    ax_nstd.legend(loc='upper left')
    plt.show()
    return F_o_M