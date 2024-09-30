# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carles Roch i Carceller
"""

# The main code begins around line 940
# above that, it's only defining functions 

''' Dependencies '''
import numpy as np
import cvxpy as cp
from cvxpy import *
import time
import chaospy # Needed for Gauss-Radau quadrature weigths and nodes
import scipy.special as sps # used to compute error function
# to integrate
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy.integrate import dblquad

# Moment matrix generators
from MoMPy.MoM import *

import warnings
warnings.filterwarnings("ignore")
''' ------------------------------'''

'''
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
'''

def deltaF(x,xx):

    """ Delta function """
    
    if x == xx:
        return 1.0
    else:
        return 0.0
    
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
    
def BPKS_probs(nB,nX,nY,amp,y1):
    
    y0 = 0.0 # Offset of the binnings in the X quadrature
    if y1 == None:
        y1 = np.sqrt(2.0)*np.abs(amp[0]) - sps.erfinv(0.5*(np.math.erf(np.sqrt(2.0)*np.abs(amp[0])-y0) -1.0 )) # Binnings in the X quadrature
    pbxy = np.zeros((nB,nX,nY))
    if nB == 2: # 2 bins
            
        for x in range(nX):
            
            amp[x] = np.sqrt(eta)*amp[x]
        
            pbxy[0][x][0] = p_alpha(amp[x],'-Inf',y0)
            pbxy[1][x][0] = p_alpha(amp[x],y0,'Inf')
            
    if nB == 4: # 4 bins
        
        for x in range(nX):
       
            amp[x] = np.sqrt(eta)*amp[x]
       
            pbxy[0][x][0] = p_alpha(amp[x],'-Inf',-y1+y0)
            pbxy[1][x][0] = p_alpha(amp[x],-y1+y0,0.0+y0)
            pbxy[2][x][0] = p_alpha(amp[x],0.0+y0,y1+y0)
            pbxy[3][x][0] = p_alpha(amp[x],y1+y0,'Inf')

    if nB == 8: # 8 bins

        for x in range(nX):
            
            amp[x] = np.sqrt(eta)*amp[x]
            
            pbxy[0][x][0] = p_alpha(amp[x],'-Inf',-3.0*y1+y0)
            pbxy[1][x][0] = p_alpha(amp[x],-3.0*y1+y0,-2.0*y1+y0)
            pbxy[2][x][0] = p_alpha(amp[x],-2.0*y1+y0,-y1+y0)
            pbxy[3][x][0] = p_alpha(amp[x],-y1+y0,0.0+y0)
            pbxy[4][x][0] = p_alpha(amp[x],0.0+y0,y1+y0)
            pbxy[5][x][0] = p_alpha(amp[x],y1+y0,2.0*y1+y0)
            pbxy[6][x][0] = p_alpha(amp[x],2.0*y1+y0,3.0*y1+y0)
            pbxy[7][x][0] = p_alpha(amp[x],3.0*y1+y0,'Inf')   
            
        
    return pbxy
        
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
    
def p_alpha(alpha,y,z):
    
    """ Probaiblity of finding a photon in the X quadrature interval x={y,z} from a coherent state |alpha >  """
    alpha = np.real(alpha)
    if y == '-Inf' and z != 'Inf':
        return 0.5*(1.0 - np.math.erf(np.sqrt(2.0)*alpha - z))
    elif y != '-Inf' and z == 'Inf':
        return 0.5*(np.math.erf(np.sqrt(2.0)*alpha - y) + 1.0)
    elif y == 'Inf' and z == 'Inf':
        return 1.0
    else:
        return 0.5*(np.math.erf(np.sqrt(2.0)*alpha - y) - np.math.erf(np.sqrt(2.0)*alpha - z))

def p_alpha_hetero(alpha,x0,xf,y0,yf):
    
    """ Probaiblity of finding a photon in the X quadrature interval x={y,z} from a coherent state |alpha >  """
    alphaR = np.real(alpha)
    alphaI = np.imag(alpha)
    
    xs = ( np.math.erf(np.sqrt(2.0)*alphaR - x0) - np.math.erf(np.sqrt(2.0)*alphaR - xf))
    ys = ( np.math.erf(np.sqrt(2.0)*alphaI - y0) - np.math.erf(np.sqrt(2.0)*alphaI - yf))
    
    return 0.25*xs*ys

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def PHet(r,theta,alpha):
    
    alphaR = np.sqrt(2.0)*np.real(alpha)
    alphaI = np.sqrt(2.0)*np.imag(alpha)
    
    return r*(np.exp(-(r*np.cos(theta)-alphaR)**2.0))*(np.exp(-(r*np.sin(theta)-alphaI)**2.0))/np.pi

def p_alpha_hetero_polar(alpha,theta0,thetaf):
    
    """ Probaiblity of finding a photon in the P_X quadrature with polar coordinates from a coherent state |alpha >  """
    return dblquad(lambda theta, r: PHet(r,theta,alpha), 0, np.inf, lambda theta: theta0, lambda theta: thetaf)[0]

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


def p_click(alpha):
    return 1.0-np.exp(-np.abs(alpha)**2.0)

def p_no_click(alpha):
    return np.exp(-np.abs(alpha)**2.0)

def p_alpha_nphotons(alpha,n):
    return (np.exp(-np.abs(alpha)**2.0))*(np.abs(alpha)**(2.0*(n)))/np.math.factorial(n) 

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def max_witness(nX,nB,nK,omega,monomials,gamma_matrix_els,Q,eps,n_trunc,type_cts):
    
    """
    Compute the maximum success probability using a SDP relaxation with monomial list: {1,rho,M,phi}
    
    Inputs:
        nX:
            Number of state preparations
        nB:
            Number of measurement outcomes
        nK:
            Number of photon number constraints
        omega:
            Bound on the photon number constraints (omega[k][x])
        monomials:
            List of monomials used to build the SDP relaxation
        gamma_matrix_els:
            Set of identities and full matrix from the SDP relaxation after applying the relaxation rules
        Q:
            fixed rate of inconclusive events (if any)
        eps:
            epsilon for almost dimension constraint
        n_trunc:
            truncated average photon number
        type_cts:
            what are we constraining. Can be: 'individual_energy' or 'avg_energy'
    """    
    [w_R,w_B,w_P] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
    if n_trunc > nK:
        return 'ERROR: n_trunc must be lower or equal than nK'
    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#
    
    G_var_vec = {}
    for element in list_of_eq_indices:
        if element == map_table[-1][-1]:
            G_var_vec[element] = 0.0 # Zeros form orthogonal projectors
        else:
            G_var_vec[element] = cp.Variable()
               
    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    G = {}
    lis = []
    for r in range(len(G_new)):
        lis += [[]]
        for c in range(len(G_new)):
            lis[r] += [G_var_vec[G_new[r][c]]]
    G = cp.bmat(lis)
            
    # Localising matrices

    elms = ['R','RR'] # These elements are included in the localising matrices
        
    loc_elms = []
    loc_elms += [ [w_R[x]] for x in range(nX) if 'R' in elms]
    loc_elms += [ [w_R[x],w_R[x]] for x in range(nX) if 'RR' in elms]

    # Localising levels
    loc_lvl = []
    
    # First order
    loc_lvl += [ [w_R[x]] for x in range(nX) ]
    loc_lvl += [ [w_B[y][b]] for y in range(nY) for b in range(nB) ]
    loc_lvl += [ [w_P[k]] for k in range(nK) ]
    
    g_loc = {}
    for elm in loc_elms:
        lis = []
        for r in range(len(loc_lvl)+1):
            lis += [[]]
            for c in range(len(loc_lvl)+1):
                if r == 0 and c > 0:
                    element = elm + reverse_list(loc_lvl[c-1])
                elif r > 0 and c == 0:
                    element = loc_lvl[r-1] + elm
                elif r == 0 and c == 0:
                    element = elm
                else:
                    element = loc_lvl[r-1] + elm + reverse_list(loc_lvl[c-1])

                if fmap(map_table,element) != 'ERROR: The value does not appear in the mapping rule':
                    lis[r] += [G_var_vec[fmap(map_table,element)]]
                else:   
                    checking = check_if_id(element,map_table,rank_1_projectors,commuting_variables,orthogonal_projectors)
                    if checking[0] == True:
                        index_el = checking[2]
                        lis[r] += [G_var_vec[index_el]]
                    elif checking[1] == True:
                        lis[r] += [0.0]
                    else:
                        lis[r] += [cp.Variable()]                

        g_loc[fmap(map_table,elm)] = cp.bmat(lis)

    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
                        
    # Normalisation constraints --------------------------------------------------------------
    for y in range(nY):
        map_table_copy = map_table[:]
        
        identities = [ term[0] for term in map_table_copy]
        norm_cts = normalisation_contraints(w_B[y],identities)
        
        for gg in range(len(norm_cts)):
            the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
            an_element_is_not_in_the_list = False
            for hhh in range(len(the_elements)):
                if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                    an_element_is_not_in_the_list = True
            if an_element_is_not_in_the_list == False:
                ct += [ sum([ G_var_vec[fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[fmap(map_table,norm_cts[gg][nB])] ]
    # ----------------------------------------------------------------------------------------
    
    # Positivity of tracial matrices and localising matrices
    ct += [G >> 0.0]
            
    # Some specific constraints in each corr matrix  -- G
        
    # Localising matrices
    ct += [ g_loc[fmap(map_table,[w_R[x],w_R[x]])] << g_loc[fmap(map_table,[w_R[x]])] for x in range(nX) ]#for xx in range(nX) ]

    # Rank-1 projectors
    ct += [ G_var_vec[fmap(map_table,[w_R[x]])] == 1.0 for x in range(nX)]
    ct += [ G_var_vec[fmap(map_table,[w_P[k]])] == 1.0 for k in range(nK)]
    ct += [ G_var_vec[fmap(map_table,[w_B[y][b]])] == 1.0 for b in range(nB) for y in range(nY)]
   
    # Photon number avg constraint
    if type_cts == 'avg_energy': # (n_trunc+1 because n_trunc = 2 means levels 0, 1 and 2, but python thinks 0 and 1)
        ct += [ sum ([ k*G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] for k in range(n_trunc+1) ]) <= np.abs(amp[x])**2.0 for x in range(nX) ]
        ct += [ sum ([ G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] for k in range(nK) ]) >= 1.0 - eps[x] for x in range(nX) ]
    elif type_cts == 'individual_energy':
        ct += [ G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] >= omega[k][x] for k in range(nK) for x in range(nX) ] 
    else:
        return 'ERROR: type_cts not recognised'
    
    # Fix rate of inconclusive events, if any
    ct += [ sum ([ G_var_vec[fmap(map_table,[w_R[x],w_B[0][b]])]/float(nX) for x in range(nX) ]) == Q for b in range(nX,nB) if nB > nX]
    
    witness = sum([ G_var_vec[fmap(map_table,[w_R[x],w_B[0][x]])]/float(nX) for x in range(nX)])  
   
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#

    obj = cp.Maximize(witness)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    return witness.value


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Shannon_Entropy(nX,nB,nK,m,t,w,pbxy,omega,monomials,gamma_matrix_els,nXstar,nYstar,W,amp,eps,n_trunc,type_cts):
    
    """
    Compute the Shannon entropy using a SDP relaxation with monomial list: {1,rho,M,phi}
    
    Inputs:
        nX:
            Number of state preparations
        nB:
            Number of measurement outcomes
        nK:
            Number of photon number constraints
        m:
            Gauss-Radau quadrature saturation limit
        t:
            Nodes from the Gauss-Radau quadrature
        w:
            Weigths from the Gauss-Radau quadrature
        pbxy:
            Full distribution of observable probabilities
        omega:
            Bound on the photon number constraints (omega[k][x])
        monomials:
            List of monomials used to build the SDP relaxation
        gamma_matrix_els:
            Set of identities and full matrix from the SDP relaxation after applying the relaxation rules
        nXstar:
            Number of settings (preparations) from which we aim to extract the randomness
        nYstar:
            Number of settings (measurements) from which we aim to extract the randomness
        W:
            witness value
        amp:
            coherent state amplitude (for average photon number constraints)
        eps:
            epsilon for almost dimension constraint
        n_trunc:
            truncated average photon number
        type_cts:
            what are we constraining. Can be: 'individual_energy' or 'avg_energy'    
    """
    
    string = f'Running SDP Hierarchy...'
    print('\r'+' '*len(string)*4+'\r', end='')
    print('\r'+string+'\r', end='')
    
    tau = [ w[i]/(t[i]*np.log(2.0)) for i in range(m) ]
    H_out = sum([ w[i]/(t[i]*np.log(2.0)) for i in range(m) ])
    
    [w_R,w_B,w_P] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
    if type_cts == 'avg_energy' and n_trunc > nK:
        return 'ERROR: n_trunc must be lower or equal than nK'
    
    for i in range(m):
            
        #----------------------------------------------------#
        #                  CREATE VARIABLES                  #
        #----------------------------------------------------#
        
        string = f'Creating variables...'
        print('\r'+' '*len(string)*4+'\r', end='')
        print('\r'+string+'\r', end='')
        
        z_var = {}
        h_var = {}
        for o in range(nB):
            z_var[o] = {}
            h_var[o] = {}
            for s in range(nXstar):
                z_var[o][s] = {}
                h_var[o][s] = {}
                for u in range(nYstar):
                    z_var[o][s][u] = cp.Variable(nonpos=True)
                    h_var[o][s][u] = cp.Variable(nonneg=True)
    
        G_var_vec = {}
        for element in list_of_eq_indices:
            if element == map_table[-1][-1]:
                G_var_vec[element] = 0.0 # Zeros form orthogonal projectors
            else:
                G_var_vec[element] = cp.Variable()
            
        zG_var_vec = {}
        hG_var_vec = {}
        for b in range(nB):
            zG_var_vec[b] = {}
            hG_var_vec[b] = {}
            for s in range(nXstar):
                zG_var_vec[b][s] = {}
                hG_var_vec[b][s] = {}
                for u in range(nYstar):
                    zG_var_vec[b][s][u] = {}
                    hG_var_vec[b][s][u] = {}
                    for element in list_of_eq_indices:
                        if element == map_table[-1][-1]:
                            zG_var_vec[b][s][u][element] = 0.0 # Zeros form orthogonal projectors
                            hG_var_vec[b][s][u][element] = 0.0 # Zeros form orthogonal projectors
                        else:
                            zG_var_vec[b][s][u][element] = cp.Variable()
                            hG_var_vec[b][s][u][element] = cp.Variable()
              
        #--------------------------------------------------#
        #                  BUILD MATRICES                  #
        #--------------------------------------------------#
        
        string = f'Building matrices...'
        print('\r'+' '*len(string)*4+'\r', end='')
        print('\r'+string+'\r', end='')
        
        G = {}
        lis = []
        for r in range(len(G_new)):
            lis += [[]]
            for c in range(len(G_new)):
                lis[r] += [G_var_vec[G_new[r][c]]]
        G = cp.bmat(lis)
            
        zG = {}
        hG = {}
        for b in range(nB):
            zG[b] = {}
            hG[b] = {}
            for s in range(nXstar):
                zG[b][s] = {}
                hG[b][s] = {}
                for u in range(nYstar):
                    zlis = []
                    hlis = []
                    for r in range(len(G_new)):
                        zlis += [[]]
                        hlis += [[]]
                        for c in range(len(G_new)):
                            zlis[r] += [zG_var_vec[b][s][u][G_new[r][c]]]
                            hlis[r] += [hG_var_vec[b][s][u][G_new[r][c]]]
                    zG[b][s][u] = cp.bmat(zlis)
                    hG[b][s][u] = cp.bmat(hlis)
                
        # Localising matrices
        
        print('\r'+' '*len(string)*2+'\r', end='')
        string = f'Generating localising matrices...'
        print('\r'+string+'\r', end='')
        
        elms = ['R','RR'] # These elements are included in the localising matrices
            
        loc_elms = []
        loc_elms += [ [w_R[x]] for x in range(nX) if 'R' in elms]
        loc_elms += [ [w_R[x],w_R[x]] for x in range(nX) if 'RR' in elms]
    
        # Localising levels
        loc_lvl = []
        
        # First order
        loc_lvl += [ [w_R[x]] for x in range(nX) ]
        loc_lvl += [ [w_B[y][b]] for y in range(nY) for b in range(nB) ]
        loc_lvl += [ [w_P[k]] for k in range(nK) ]
    
        g_loc = {}
        for elm in loc_elms:
            lis = []
            for r in range(len(loc_lvl)+1):
                lis += [[]]
                for c in range(len(loc_lvl)+1):
                    if r == 0 and c > 0:
                        element = elm + reverse_list(loc_lvl[c-1])
                    elif r > 0 and c == 0:
                        element = loc_lvl[r-1] + elm
                    elif r == 0 and c == 0:
                        element = elm
                    else:
                        element = loc_lvl[r-1] + elm + reverse_list(loc_lvl[c-1])
    
                    if fmap(map_table,element) != 'ERROR: The value does not appear in the mapping rule':
                        lis[r] += [G_var_vec[fmap(map_table,element)]]
                    else:   
                        checking = check_if_id(element,map_table,rank_1_projectors,commuting_variables,orthogonal_projectors)
                        if checking[0] == True:
                            index_el = checking[2]
                            lis[r] += [G_var_vec[index_el]]
                        elif checking[1] == True:
                            lis[r] += [0.0]
                        else:
                            lis[r] += [cp.Variable()]                
    
            g_loc[fmap(map_table,elm)] = cp.bmat(lis)
    
        zg_loc = {}
        hg_loc = {}
        for b in range(nB):
            zg_loc[b] = {}
            hg_loc[b] = {}
            for s in range(nXstar):
                zg_loc[b][s] = {}
                hg_loc[b][s] = {}
                for u in range(nYstar):
                    zg_loc[b][s][u] = {}
                    hg_loc[b][s][u] = {}
                    for elm in loc_elms:
                        zlis = []
                        hlis = []
                        for r in range(len(loc_lvl)+1):
                            zlis += [[]]
                            hlis += [[]]
                            for c in range(len(loc_lvl)+1):
                                if r == 0 and c > 0:
                                    element = elm + reverse_list(loc_lvl[c-1])
                                elif r > 0 and c == 0:
                                    element = loc_lvl[r-1] + elm
                                elif r == 0 and c == 0:
                                    element = elm
                                else:
                                    element = loc_lvl[r-1] + elm + reverse_list(loc_lvl[c-1])
                                    
                                if fmap(map_table,element) != 'ERROR: The value does not appear in the mapping rule':
                                    zlis[r] += [zG_var_vec[b][s][u][fmap(map_table,element)]]
                                    hlis[r] += [hG_var_vec[b][s][u][fmap(map_table,element)]]
                                else:    
                                    checking = check_if_id(element,map_table,rank_1_projectors,commuting_variables,orthogonal_projectors)
                                    if checking[0] == True:
                                        index_el = checking[2]
                                        zlis[r] += [zG_var_vec[b][s][u][index_el]]
                                        hlis[r] += [hG_var_vec[b][s][u][index_el]]
                                    elif checking[1] == True:
                                        zlis[r] += [0.0]
                                        hlis[r] += [0.0]
                                    else:
                                        zlis[r] += [cp.Variable()]                
                                        hlis[r] += [cp.Variable()]                
                                    
                        zg_loc[b][s][u][fmap(map_table,elm)] = cp.bmat(zlis)
                        hg_loc[b][s][u][fmap(map_table,elm)] = cp.bmat(hlis)
    
        #------------------------------------------------------#
        #                  CREATE CONSTRAINTS                  #
        #------------------------------------------------------#
        
        ct = []
        
        print('\r'+' '*len(string)*2+'\r', end='')
        string = f'Generating constraints [normalisation]...'
        print('\r'+string+'\r', end='')
                            
        # Normalisation constraints --------------------------------------------------------------
        for y in range(nY):
            map_table_copy = map_table[:]
            
            identities = [ term[0] for term in map_table_copy]
            norm_cts = normalisation_contraints(w_B[y],identities)
            
            for gg in range(len(norm_cts)):
                the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
                an_element_is_not_in_the_list = False
                for hhh in range(len(the_elements)):
                    if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                        an_element_is_not_in_the_list = True
                if an_element_is_not_in_the_list == False:
                    ct += [ sum([ G_var_vec[fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[fmap(map_table,norm_cts[gg][nB])] ]
                    for o in range(nB):
                        for s in range(nXstar):
                            for u in range(nYstar):
                                ct += [ sum([ zG_var_vec[o][s][u][fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == zG_var_vec[o][s][u][fmap(map_table,norm_cts[gg][nB])] ]
                                ct += [ sum([ hG_var_vec[o][s][u][fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == hG_var_vec[o][s][u][fmap(map_table,norm_cts[gg][nB])] ]
        # ----------------------------------------------------------------------------------------
    
        print('\r'+' '*len(string)*2+'\r', end='')
        string = f'Generating constraints [positivity]...'
        print('\r'+string+'\r', end='')    
        tol = 1e-8
        # Positivity of tracial matrices and localising matrices
        for b in range(nB):
            for s in range(nXstar):
                for u in range(nYstar):
                        
                    ct += [zG[b][s][u] << tol*np.identity(len(G_new))] # uncomment if the result is not tight
                     
                    Gamma = cp.bmat([[ G         ,zG[b][s][u]],
                                     [zG[b][s][u],hG[b][s][u]] ])
                    ct += [Gamma >> -tol*np.identity(len(G_new)*2)]
                    
        print('\r'+' '*len(string)*2+'\r', end='')
        string = f'Generating constraints [other]...'
        print('\r'+string+'\r', end='')
                
        # Some specific constraints in each corr matrix  -- G
    
        # Rank-1 projectors
        ct += [ G_var_vec[fmap(map_table,[w_R[x]])] == 1.0 for x in range(nX)]
        ct += [ G_var_vec[fmap(map_table,[w_P[k]])] == 1.0 for k in range(nK)]
      #  ct += [ G_var_vec[fmap(map_table,[w_B[y][b]])] == 1.0 for b in range(nB) for y in range(nY)]
        
        for o in range(nB):
            for s in range(nXstar):
                for u in range(nYstar):
                    
                    # Big localising matrix
                    if 'R' in elms and 'RR' in elms:
                        for x in range(nX):
                            big_loc_R = cp.bmat([[g_loc[fmap(map_table,[w_R[x]])],           zg_loc[o][s][u][fmap(map_table,[w_R[x]])]],
                                                 [zg_loc[o][s][u][fmap(map_table,[w_R[x]])], hg_loc[o][s][u][fmap(map_table,[w_R[x]])]]])
                            
                            big_loc_RR = cp.bmat([[g_loc[fmap(map_table,[w_R[x],w_R[x]])],           zg_loc[o][s][u][fmap(map_table,[w_R[x],w_R[x]])]],
                                                  [zg_loc[o][s][u][fmap(map_table,[w_R[x],w_R[x]])], hg_loc[o][s][u][fmap(map_table,[w_R[x],w_R[x]])]]])
                            
                            ct += [ big_loc_RR << big_loc_R ]
                    
                    # Some specific constraints in each corr matrix  -- z * G
                    # Rank-1 projectors
                    ct += [ zG_var_vec[o][s][u][fmap(map_table,[w_R[x]])] == z_var[o][s][u] for x in range(nX)]
                    ct += [ zG_var_vec[o][s][u][fmap(map_table,[w_P[k]])] == z_var[o][s][u] for k in range(nK)]
                    #ct += [ zG_var_vec[o][s][u][fmap(map_table,[w_B[y][b]])] == z_var[o][s][u] for b in range(nB) for y in range(nY)]
                    #ct += [ zG_var_vec[o][s][u][fmap(map_table,[w_B[y][b]])] == zG_var_vec[o][s][u][fmap(map_table,[w_B[np.mod(y+1,nY)][np.mod(b+1,nB)]])] for b in range(nB) for y in range(nY)]
                    
                    # Some specific constraints in each corr matrix  -- h * G
                    # Rank-1 projectors
                    ct += [ hG_var_vec[o][s][u][fmap(map_table,[w_R[x]])] == h_var[o][s][u] for x in range(nX)]
                    ct += [ hG_var_vec[o][s][u][fmap(map_table,[w_P[k]])] == h_var[o][s][u] for k in range(nK)]
                    #ct += [ hG_var_vec[o][s][u][fmap(map_table,[w_B[y][b]])] == h_var[o][s][u] for b in range(nB) for y in range(nY)]
                    #ct += [ hG_var_vec[o][s][u][fmap(map_table,[w_B[y][b]])] == hG_var_vec[o][s][u][fmap(map_table,[w_B[np.mod(y+1,nY)][np.mod(b+1,nB)]])] for b in range(nB) for y in range(nY)]
   
        # Photon number avg constraint
        if type_cts == 'avg_energy': # (n_trunc+1 because n_trunc = 2 means levels 0, 1 and 2, but python thinks 0 and 1)
            ct += [ sum ([ k*G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] for k in range(n_trunc+1) ]) <= np.abs(amp[x])**2.0 for x in range(nX) ]
            ct += [ sum ([ G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] for k in range(nK) ]) >= 1.0 - eps[x] for x in range(nX) ]
        elif type_cts == 'individual_energy':
            ct += [ G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] >= omega[k][x] for k in range(nK) for x in range(nX) ] 
        else:
            return 'ERROR: type_cts not recognised'
        
        # Witness or full distribution
        if W == None:
            ct += [ G_var_vec[fmap(map_table,[w_R[x],w_B[y][b]])] == pbxy[b][x][y] for b in range(nB) for x in range(nX) for y in range(nY)]
        else:
            ct += [ sum([ G_var_vec[fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX)]) >= W  for y in range(nY)]
        
        # Shannon entropy
        H = 0.0
        for b in range(nB):
            for x in range(nXstar):
                for y in range(nYstar):
                
                    H += w[i]/(t[i]*np.log(2.0)) * ( 2.0*zG_var_vec[b][x][y][fmap(map_table,[w_B[y][b],w_R[x]])] + 
                                              (1.0-t[i])*hG_var_vec[b][x][y][fmap(map_table,[w_B[y][b],w_R[x]])] + 
                                                   t[i] *hG_var_vec[b][x][y][fmap(map_table,[w_R[x]])] ) / float(nXstar*nYstar)
        
        ct += [H >= -666.0] # To detect unbounded solutions
            
        #----------------------------------------------------------------#
        #                  RUN THE SDP and WRITE OUTPUT                  #
        #----------------------------------------------------------------#
        
        print('\r'+' '*len(string)*2+'\r', end='')
        string = f'Solving {i} of {m}...'
        print('\r'+string+'\r', end='')
    
        obj = cp.Minimize(H)
        prob = cp.Problem(obj,ct)
    
        output = []
    
        try:
            mosek_params = {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
                }
            prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
        except SolverError:
            something = 10
            
        if H.value != None:
            H_out += H.value
            #print('Hvalue',H.value)
        else:
            H_out = None
            break
            
    return H_out


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Min_Entropy(nX,nB,nK,pbxy,omega,monomials,gamma_matrix_els,nXstar,nYstar,W,amp,eps,n_trunc,type_cts):
    
    """
    Compute the Min-entropy using a SDP relaxation with monomial list: {1,rho,M,phi}
    
    Inputs:
        nX:
            Number of state preparations
        nB:
            Number of measurement outcomes
        nK:
            Number of photon number constraints
        pbxy:
            Full distribution of observable probabilities
        omega:
            Bound on the photon number constraints (omega[k][x])
        monomials:
            List of monomials used to build the SDP relaxation
        gamma_matrix_els:
            Set of identities and full matrix from the SDP relaxation after applying the relaxation rules
        nXstar:
            Number of settings (preparations) from which we aim to extract the randomness
        nYstar:
            Number of settings (measurements) from which we aim to extract the randomness
        W:
            witness value
        amp:
            coherent state amplitude (for average photon number constraints)
        eps:
            epsilon for almost dimension constraint
        n_trunc:
            truncated average photon number
        type_cts:
            what are we constraining. Can be: 'individual_energy' or 'avg_energy'    
    """
    
    nL = nB
    xstar = 0
    ystar = 0
    
    string = f'Running SDP Hierarchy...'
    print('\r'+' '*len(string)*4+'\r', end='')
    print('\r'+string+'\r', end='')
    
    [w_R,w_B,w_P] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
    if type_cts == 'avg_energy' and n_trunc > nK:
        return 'ERROR: n_trunc must be lower or equal than nK'
    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#
    
    string = f'Creating variables...'
    print('\r'+' '*len(string)*4+'\r', end='')
    print('\r'+string+'\r', end='')
    
    q = cp.Variable(nL,nonneg=True)
    
    G_var_vec = {}
    for l in range(nL):
        G_var_vec[l] = {}
        for element in list_of_eq_indices:
            if element == map_table[-1][-1]:
                G_var_vec[l][element] = 0.0 # Zeros form orthogonal projectors
            else:
                G_var_vec[l][element] = cp.Variable()
        
    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    string = f'Building matrices...'
    print('\r'+' '*len(string)*4+'\r', end='')
    print('\r'+string+'\r', end='')
    
    G = {}
    for l in range(nL):
        lis = []
        for r in range(len(G_new)):
            lis += [[]]
            for c in range(len(G_new)):
                lis[r] += [G_var_vec[l][G_new[r][c]]]
        G[l] = cp.bmat(lis)
            
    # Localising matrices
    
    print('\r'+' '*len(string)*2+'\r', end='')
    string = f'Generating localising matrices...'
    print('\r'+string+'\r', end='')
    
    elms = ['R','RR'] # These elements are included in the localising matrices
        
    loc_elms = []
    loc_elms += [ [w_R[x]] for x in range(nX) if 'R' in elms]
    loc_elms += [ [w_R[x],w_R[x]] for x in range(nX) if 'RR' in elms]

    # Localising levels
    loc_lvl = []
    
    # First order
    loc_lvl += [ [w_R[x]] for x in range(nX) ]
    loc_lvl += [ [w_B[y][b]] for y in range(nY) for b in range(nB) ]
    loc_lvl += [ [w_P[k]] for k in range(nK) ]

    g_loc = {}
    for l in range(nL):
        g_loc[l] = {}
        for elm in loc_elms:
            lis = []
            for r in range(len(loc_lvl)+1):
                lis += [[]]
                for c in range(len(loc_lvl)+1):
                    if r == 0 and c > 0:
                        element = elm + reverse_list(loc_lvl[c-1])
                    elif r > 0 and c == 0:
                        element = loc_lvl[r-1] + elm
                    elif r == 0 and c == 0:
                        element = elm
                    else:
                        element = loc_lvl[r-1] + elm + reverse_list(loc_lvl[c-1])
    
                    if fmap(map_table,element) != 'ERROR: The value does not appear in the mapping rule':
                        lis[r] += [G_var_vec[l][fmap(map_table,element)]]
                    else:   
                        checking = check_if_id(element,map_table,rank_1_projectors,commuting_variables,orthogonal_projectors)
                        if checking[0] == True:
                            index_el = checking[2]
                            lis[r] += [G_var_vec[l][index_el]]
                        elif checking[1] == True:
                            lis[r] += [0.0]
                        else:
                            lis[r] += [cp.Variable()]                

            g_loc[l][fmap(map_table,elm)] = cp.bmat(lis)

    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
    
    print('\r'+' '*len(string)*2+'\r', end='')
    string = f'Generating constraints [normalisation]...'
    print('\r'+string+'\r', end='')

    ct += [ sum([q[l] for l in range(nL)]) == 1.0 ]                 
       
    # Normalisation constraints --------------------------------------------------------------
    for y in range(nY):
        map_table_copy = map_table[:]
        
        identities = [ term[0] for term in map_table_copy]
        norm_cts = normalisation_contraints(w_B[y],identities)
        
        for gg in range(len(norm_cts)):
            the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
            an_element_is_not_in_the_list = False
            for hhh in range(len(the_elements)):
                if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                    an_element_is_not_in_the_list = True
            if an_element_is_not_in_the_list == False:
                for l in range(nL):
                    ct += [ sum([ G_var_vec[l][fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[l][fmap(map_table,norm_cts[gg][nB])] ]
    # ----------------------------------------------------------------------------------------

    print('\r'+' '*len(string)*2+'\r', end='')
    string = f'Generating constraints [positivity]...'
    print('\r'+string+'\r', end='')    
    tol = 1e-8
    # Positivity of tracial matrices and localising matrices
    
    ct += [G[l] >> 0.0 for l in range(nL)]
                
    print('\r'+' '*len(string)*2+'\r', end='')
    string = f'Generating constraints [other]...'
    print('\r'+string+'\r', end='')
            
    # Some specific constraints in each corr matrix  -- G
    for l in range(nL):
        # Rank-1 projectors
        ct += [ G_var_vec[l][fmap(map_table,[w_R[x]])] == q[l] for x in range(nX)]
        ct += [ G_var_vec[l][fmap(map_table,[w_P[k]])] == q[l] for k in range(nK)]
       # ct += [ G_var_vec[l][fmap(map_table,[w_B[y][b]])] == q[l] for b in range(nB) for y in range(nY)]
        
    # Big localising matrix
    if 'R' in elms and 'RR' in elms:
        for x in range(nX):
            ct += [ g_loc[l][fmap(map_table,[w_R[x],w_R[x]])] << g_loc[l][fmap(map_table,[w_R[x]])] for l in range(nL) ]
    
    # Photon number avg constraint
    if type_cts == 'avg_energy': # (n_trunc+1 because n_trunc = 2 means levels 0, 1 and 2, but python thinks 0 and 1)
        ct += [ sum([ k*G_var_vec[l][fmap(map_table,[w_R[x],w_P[k]])] for k in range(n_trunc+1) for l in range(nL) ]) <= np.abs(amp[x])**2.0 for x in range(nX) ]
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_P[k]])] for k in range(nK) for l in range(nL) ]) >= 1.0 - eps[x] for x in range(nX) ]
    elif type_cts == 'individual_energy':
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_P[k]])] for l in range(nL)]) >= omega[k][x] for k in range(nK) for x in range(nX) ] 
    else:
        return 'ERROR: type_cts not recognised'
    
    # Witness or full distribution
    if W == None:
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_B[y][b]])] for l in range(nL)]) == pbxy[b][x][y] for b in range(nB) for x in range(nX) for y in range(nY)]
    else:
        ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[x],w_B[y][x]])]/float(nX) for x in range(nX) for l in range(nL) ]) >= W  for y in range(nY)]
    
    pg = sum([ G_var_vec[l][fmap(map_table,[w_R[xstar],w_B[ystar][l]])] for l in range(nL) ])
        
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#
    
    print('\r'+' '*len(string)*2+'\r', end='')
    string = f'Solving...'
    print('\r'+string+'\r', end='')

    obj = cp.Maximize(pg)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    if pg.value != None:
        return -np.log2(pg.value)
    else:
        return None

'''
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        MAIN CODE                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
'''

nX = 2 # number of state preparations
nY = 1 # number of measurement settings
nB = 8 # number of measurement outcomes
nK = 1 # number of photon states (for avg energy nK >= nX)
n_trunc = 1 # energy level truncation, n_trunc level included (only relevant for average energy)

type_cts = 'individual_energy'# this can be 'individual_energy' or 'avg_energy'

nXstar = 1 # from how many states you certify randomness
nYstar = 1 #nY # from how many measurements you certify randomness

N = 4 # number of datapoints

# Gauss-Radau quadrature weitghts (w) and nodes (t)
m_in = 4 # half of the quadrature limit ( m = m_in * 2 )
m = int(m_in*2) # quadrature limit
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

vec = np.linspace(0.001,0.02,N) # x-axis vector (sqrt of coherent amplitude)

H_vec = [[],[]] # Store output in this vectors
Hmin_vec = [[],[]] # Store output in this vectors
W_vec = [[],[]] # Store witness in this vectors

#---------------------------------------------------------------------#
#                        Collect all monomials                        #
#---------------------------------------------------------------------#

string = f'Generating matrix of correlations...'
print(string+'\r', end='')

# Track operators in the tracial matrix
w_R = [] # Prepared quantum state
w_B = [] # Measurement
w_P = [] # Observable (photon number projector)

S_1 = [] # List of first order elements
cc = 1

for x in range(nX):
    S_1 += [cc]
    w_R += [cc]
    cc += 1

for y in range(nY):
    w_B += [[]]
    for b in range(nB):
        S_1 += [cc]
        w_B[y] += [cc]
        cc += 1
        
for k in range(nK):
    S_1 += [cc]
    w_P += [cc]
    cc += 1

# Additional higher order elements
S_high = [] # Uncomment if we only allow up to some 2nd order elements in the hierarchy  

# Second order elements
some_second = False
if some_second == True:
    for x in range(nX):
        for xx in range(nX):
            S_high += [[w_R[x],w_R[xx]]]
            
    for x in range(nX):
        for b in range(nB):
            for y in range(nY):
                S_high += [[w_R[x],w_B[y][b]]]
                #S_high += [[w_B[y][b],w_R[x]]]
            
    for k in range(nK):
        for b in range(nB):
            for y in range(nY):
                S_high += [[w_P[k],w_B[y][b]]]
            
    #for x in range(nX):
    #    for k in range(nK):
    #          S_high += [[w_P[k],w_R[x]]]

some_third = True
if some_third == True:
    for x in range(nX):
        for xx in range(nX):
            for xxx in range(nX):
                S_high += [[w_R[x],w_R[xx],w_R[xxx]]]
            
    #for x in range(nX):
    #    for b in range(nB):
    #        for y in range(nY):
    #            for xx in range(nX):
    #                S_high += [[w_R[x],w_B[y][b],w_R[xx]]]

            
    for k in range(nK):
        for b in range(nB):
            for y in range(nY):
                for x in range(nX):
                    S_high += [[w_P[k],w_B[y][b],w_R[x]]]
                    #S_high += [[w_B[y][b],w_P[k],w_R[x]]]


#S_high = []

# Set the operational rules within the SDP relaxation
list_states = [] # operators that do not commute with anything (not important here)

rank_1_projectors = []#w_R
rank_1_projectors += [w_B[y][b] for y in range(nY) for b in range(nB)]
rank_1_projectors += [w_P[k] for k in range(nK)]

orthogonal_projectors = []
orthogonal_projectors += [ w_B[y] for y in range(nY)]
orthogonal_projectors += [ w_P ] 

commuting_variables = [] # commuting elements (wxcept with elements in "list_states"

print('Rank-1 projectors',rank_1_projectors)
print('Orthogonal projectors',orthogonal_projectors)
print('commuting elements',commuting_variables)

# Collect rules and generate SDP relaxation matrix
start = time.process_time()
[G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,S_1,S_high,rank_1_projectors,orthogonal_projectors,commuting_variables,list_states)
end = time.process_time()

print('Gamma matrix generated in',end-start,'s')
print('Matrix size:',np.shape(G_new))

monomials = [w_R,w_B,w_P]
gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]

for jj in range(N):

    eta = 1.0#-vec[jj]# Efficiency (1 - Photon loss)

    alpha = np.sqrt(vec[jj]) # Coherent state amplitude
    
    amp = [ alpha*np.exp(1j* (2.0*np.pi/nX * (2.0*x+1.0)/2.0 - np.pi/2.0) ) for x in range(nX) ]

    theta0_vec = [ 2.0*np.pi/nB*b - np.pi/2.0 for b in range(nB) ]
    thetaf_vec = [ 2.0*np.pi/nB*(b+1) - np.pi/2.0 for b in range(nB) ]    

    # Photon number constraint in each state ( probability |<k|\psi_x>|^2 )
    omega = np.zeros((nK,nX))
    for k in range(nK):
        for x in range(nX):
            omega[k][x] = p_alpha_nphotons(amp[x],k) # Probability of finding k photons in state x
    
    # Full distribution of observed probabilities
    #pbxy = np.zeros((nB,nX,nY)) 
    #for x in range(nX):
    #    for b in range(nB):
    #        pbxy[b][x][0] = p_alpha_hetero_polar(np.sqrt(eta)*amp[x],theta0_vec[b],thetaf_vec[b])

    eps = [1.0-sum([p_alpha_nphotons(np.sqrt(eta)*amp[x],k) for k in range(nK)]) for x in range(nX)]
    #eps = [0.0 for x in range(nX)]
    
    # Witness (if rng is based on observed probabilities, write W = None)
    W = None#max_witness(nX,nB,nK,omega,monomials,gamma_matrix_els,0.0,eps,n_trunc,type_cts)

    pbxy = BPKS_probs(nB,nX,nY,amp,None)
    
    start = time.process_time()
    out_H = Shannon_Entropy(nX,nB,nK,m-1,t,w,pbxy,omega,monomials,gamma_matrix_els,nXstar,nYstar,W,amp,eps,n_trunc,type_cts)
    out_Hmin = Min_Entropy(nX,nB,nK,pbxy,omega,monomials,gamma_matrix_els,nXstar,nYstar,W,amp,eps,n_trunc,type_cts)
    end = time.process_time()

    H_vec[0] += [vec[jj]]
    H_vec[1] += [out_H]
    
    Hmin_vec[0] += [vec[jj]]
    Hmin_vec[1] += [out_Hmin]
    
    W_vec[0] += [vec[jj]]
    W_vec[1] += [W]
    
    print(alpha**2.0)
    print(W,out_H,out_Hmin,'in',np.round(end-start,2),'seconds')
    #print(W,out_H,'in',np.round(end-start,2),'seconds')
    
  #  np.savetxt('data_outputs/H_output.csv', H_vec, delimiter =", ", fmt ='% s') 
  #  np.savetxt('data_outputs/W_output.csv', W_vec, delimiter =", ", fmt ='% s') 
    
