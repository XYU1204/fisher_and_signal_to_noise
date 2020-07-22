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

def getC_ellOfSigma8(sigma8):
    """create derivatives of cl w.r.t to Sigma8"""
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(dndz_cut, ell, cosmo)
    return cl

def getC_ellOfOmegab(Omega_b):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr

def getC_ellOfh(h):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr

def getC_ellOfn_s(n_s):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr

def getC_ellOfOmegam(Omega_m):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr

def getC_ellOfw0(w_0):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, w0=w_0, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr

def getC_ellOfwa(w_a):
    cosmo = ccl.Cosmology(Omega_c = Omega_m-Omega_b, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, w0=w_a, transfer_function= transfer_function)
    cl = getCl(dndz_sliced, ell, cosmo)
    return cl_arr
        
def fisher_matrix(covariance):
    funcs = {
    'sigma_8': getC_ellOfSigma8,
    'omega_b': getC_ellOfOmegab,
    'h': getC_ellOfh,
    'n_s': getC_ellOfn_s,
    'omega_m': getC_ellOfOmegam,
    'w_0': getC_ellOfw0,
    'w_a': getC_ellOfwa}
    vals = {
    'sigma_8': sigma8, 
    'omega_b': Omega_b, 
    'h': h, 
    'n_s': n_s, 
    'omega_m': Omega_m,
    'w_0': w_0,
    'w_a': w_a}
    derivs_sig = {}
    for var in funcs.keys():
        if vals[var] == 0:
            f = nd.Derivative(funcs[var], full_output=True, step=0.1)
        else:
            f = nd.Derivative(funcs[var], full_output=True, step=float(vals[var])/10)
        val, info = f(vals[var])
        derivs_sig[var] = val
    param_order = ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h']
    param_labels = [r'$\Omega_m$', r'$\sigma_8$', r'$n_s$', r'$w_0$', r'$w_a$', r'$\Omega_b$', r'$h$']
    fisher = np.zeros((7,7))
    for i, var1 in enumerate(param_order):
        for j, var2 in enumerate(param_order):
            f = []
            for l in range(derivs_sig[var1].shape[1]):
                res = derivs_sig[var1][:, l].T @ np.linalg.inv(cov["n_5"][l]) @ derivs_sig[var2][:, l]
                f.append(res)
            fisher[i][j] = sum(f)
    return fisher