from astropy.table import Table
import numpy as np 
from matplotlib import pyplot as plt
#matplotlib inline
#load_ext autoreload
#autoreload 2
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.stats import iqr
from scipy.stats import dirichlet
from scipy.interpolate import Rbf
from scipy.stats import norm
import pickle
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 900
from scipy.stats import multivariate_normal
import pickle
import json

"""This program predicts the theoretical covariance matrix
for LSS for each row in the input file


USES the cosmosis format

Author: MMRAU
"""

import sys
import numpy as np

#def get_cov_matrix_1bin(l, cl_vals, orderings, fsky): 
#    prefac = 1.0/(2.0*l + 1.0)/fsky


def get_cov_matrix(l, data_order, cl_vals, orderings, fsky):
    """
    Return the covariance matrix
    Parameters:
    l: mode number
    data_order: order of the data vector
    cl_vals: numpy vector shotnoise already added for lookup purposes
    orderings: bin_ordering of the inputs for lookup purposes
    fsky: sky benefit
    number_density values: shot noise component
    """
    prefac = 1.0/(2.0*l + 1.0)/fsky
    out_cov = np.zeros((len(data_order), len(data_order)))
    for z in range(out_cov.shape[0]):
        for y in range(out_cov.shape[1]):
            i = data_order[z][0]
            j = data_order[z][1]
            k = data_order[y][0]
            l = data_order[y][1]

            try:
                cl_ik = cl_vals[orderings.index([i, k])]
            except ValueError:
                #this happens if only autocorrelation is queried
                cl_ik = 0.0

            try:
                cl_jl = cl_vals[orderings.index([j, l])]
            except ValueError:
                cl_jl = 0.0

            try:
                cl_il = cl_vals[orderings.index([i, l])]
            except ValueError:
                cl_il = 0.0

            try:
                cl_jk = cl_vals[orderings.index([j, k])]
            except ValueError:
                cl_jk = 0.0

            out_cov[z, y] = prefac * (cl_ik*cl_jl + cl_il*cl_jk)
    return out_cov

def multi_bin_cov(fsky, Clbins, Cl_ordering, num_dens, *shape_noise): 
        #first column --> l values

    combination_cl = Clbins[:, 0]
    orderings = []
    for i in range(1, Clbins.shape[1]):
        #Cl_orderings has no l entry
        index_comb = Cl_ordering[i-1, :].tolist()
        if index_comb[0] != index_comb[1]:
            #add the entry plus the
            #reversed entry --> would NOT be valid for cross correlations since different
            #bins!!!
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            orderings.append(index_comb)
            orderings.append([index_comb[1], index_comb[0]])
        else:
            #autocorrelations
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            orderings.append(index_comb)

    #remove the first column from combination_cl --> makes it easier since now
    #it corresponds to orderings vector

    combination_cl = combination_cl[:, 1:]

    assert len(orderings) == combination_cl.shape[1]

    #add the shapenoise to each of the cl combinations:
    if shape_noise:
        shapenoise = []
        for el in orderings:
            shapenoise.append(shape_noise/(2*num_dens[int(el[0] - 1)]))  # because ordering starts with 1
        shapenoise = np.array(shapenoise)
        assert len(shapenoise) == combination_cl.shape[1]

        for i in range(combination_cl.shape[1]):
        #only the autocorrelation is affected by shot noise
            if orderings[i][0] == orderings[i][1]:
                combination_cl[:, i] += shapenoise[i]

    #now calculate the covariance matrice for each of the Cl_orderings
    out_mat = []
    for i in range(Clbins.shape[0]):
        curr_l = Clbins[i, 0]
        matrix_out = get_cov_matrix(curr_l, Cl_ordering, combination_cl[i, :], orderings, fsky)
        out_mat.append(matrix_out)
    return out_mat
        #np.savetxt(X=matrix_out, fname=out_filename+str(curr_l)+".mat")

def one_bin_cov(fsky, Clbins, num_dens): 
    added_shotnoise = (Clbins[:, 1] + 1.0/num_dens)**2
    prefactor = 2.0/((2.0 * Clbins[:, 0] + 1.)*fsky) 
    covariance = prefactor * added_shotnoise 
    cov_matrix = np.diag(covariance)
    return cov_matrix
    #np.savetxt(X=cov_matrix, fname=out_filename+"onebin.mat")
    
