import codecs
import sys
import re
import numpy as np
import json
import h5py
from scipy.sparse import csr_matrix, lil_matrix
import h5sparse
from ast import literal_eval
import re
import collections
import scipy.sparse as ss
import scipy
import sys


def main():
   
    mat1=ss.load_npz('mat_06eu.npz')
    mat2=ss.load_npz('mat_06com.npz')
    mat3=ss.load_npz('mat_07.npz')
    mat4=ss.load_npz('mat_08.npz')
    mat5=ss.load_npz('mat_09.npz')
    mat6=ss.load_npz('mat_10.npz')
    mat7=ss.load_npz('mat_11.npz')
    mat8=ss.load_npz('mat_12.npz')
    mat9=ss.load_npz('mat_13.npz')
    mat10=ss.load_npz('mat_14.npz')
    mat11=ss.load_npz('mat_15.npz')
    mat12=ss.load_npz('mat_16.npz')
    
    sum_read=mat1+mat2+mat3+mat4+mat5+mat6+mat7+mat8+mat9+mat10+mat11+mat12 #sum_read is a csr_matrix
    ss.save_npz('final_mat.npz', sum_read)
    
    #print "sum_mat[1,0], ", sum_read[1,0]
    #print "sum_mat[0,1], ", sum_read[0,1]
    # with h5sparse.File("mat1"+".h5") as h5f:
    #     h5f.create_dataset('sparse/matrix', data=csr_matrix(mat1)
    # mat1read = h5sparse.File("mat1"+".h5")


   
    mat_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round.npz'
    dic_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_all.txt'
    #dic_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_test1.txt'
    cws_clean = json.load(open(dic_file)) #this how you read json
    print "Finished reading content word dictionary its length is:", len(cws_clean)
    
    mat=ss.load_npz('final_mat.npz')
    (nrows, ncols) = mat.get_shape() # gets the original shape in this case (169836, 169836)
        
    colTotals =  mat.sum(axis=0)  # sum the columns
    N = np.sum(colTotals)
    rowTotals = mat.sum(axis=1)  # sum the rows
 
    nonzeroRows=mat.nonzero()[0]
    nonzeroCols=mat.nonzero()[1]
    
    matres = lil_matrix((len(cws_clean), len(cws_clean)) )
    
    for ind, row in enumerate(nonzeroRows):
        if ind % 100000 == 0: #n_lines divides in 10000 without remainder
            print  str(row)+'\r'
            sys.stdout.flush()
        col=nonzeroCols[ind]
        val = np.log((mat[row,col] * N) / float(rowTotals[row,0] * colTotals[0,col])) #its actually ln
        #matres[row,col] = max(0, val) #consider round
        matres[row,col] = max(0, round(val,2))
   
    ss.save_npz(mat_file, csr_matrix(matres))
    print " i'm here"
    
#################################################################     
########################

if __name__ == '__main__':
    main()
