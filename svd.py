import numpy, scipy.sparse
from sparsesvd import sparsesvd
import ConfigParser
import numpy
import codecs
import sys
import time
import random
import math
import scipy
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
import scipy.sparse as ss
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def main():
    
    #===========================================================================
    # mat = numpy.random.rand(300, 300)
    # smat = scipy.sparse.csc_matrix(mat)
    # ut, s, vt = sparsesvd(smat,100)
    # tmp=numpy.diag(s)
    # test=numpy.dot(ut.T, numpy.dot(numpy.diag(s), vt))#vt=(300,300), ut=(300,300), s=(300,1)
    # u2, s2, v2=svds(mat, k=100)
    # 
    # print ""
    #===========================================================================
    #ut, s, vt = sparsesvd(smat,100) # do SVD, asking for 100 factors
    # ut - Unitary matrices.
    #s -The singular values for every matrix, sorted in descending order.
    #vt  - Unitary matrices
    #assert numpy.allclose(mat, numpy.dot(ut.T, numpy.dot(numpy.diag(s), vt)))   #test if mat is close to numpy.dot(ut.T, numpy.dot(numpy.diag(s), vt))
    
    
    ################################################################################################################
        
    mat1=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round_allpats.npz')
    (nrows, ncols) = mat1.get_shape()
    
    #u1, s1, v1 = svds(mat1, k=500)
    u1, s1, v1 = sparsesvd(csc_matrix(mat1), 500) #v1(500,746K), u1(500,746K) s1[500,1]
    reduced_mat=numpy.dot(u1.T,numpy.diag(s1) )
    ss.save_npz('svd_reduced_mat_500_allpats', csr_matrix(reduced_mat))

  
    print "I'm here"
    
#################################################################     
########################

if __name__ == '__main__':
    main()
