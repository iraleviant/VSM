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
from scipy.sparse import csr_matrix, lil_matrix
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
import scipy.sparse as ss
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.preprocessing import  normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy  import sparse
import sklearn.preprocessing as pp

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0) #if zero then cosine between columns
    return col_normed_mat.T * col_normed_mat

def main():
    
    ####  Example ###########
    #===========================================================================
    # A =  csc_matrix(np.array([[1, 1, 1], [1, 1, 0]]))
    # col_normed_A = pp.normalize(A.tocsc(), axis=0)
    # cols=[0,2]
    # nmat=col_normed_A[:,cols]
    # res= nmat.T * nmat
    # rows, cols = res.nonzero()
    #===========================================================================

    #######################################################################################################################
    #dic_file_order='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_allpats_python_order_200.dat'
    dic_file_order='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_allpats_python_order.dat'
    fread=codecs.open(dic_file_order)
    cws_clean={}
    
    lines_f = fread.readlines()[1:]
    for line_g in lines_f:
        line_f=line_g.strip()
        line=line_f.split(" ")
        cws_clean[line[0]]=line[1]
    print "Finished reading content word dictionary its length is:", len(cws_clean)
        
    
    fread_human=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    count=0
    cols_to_cos=[]
    pair_list_human=[]
    new_dic={}
    lines_f = fread_human.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    for line_f in lines_f:
        tokens = line_f.split(",")
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        if word_i in cws_clean and word_j in cws_clean:
            pair_list_human.append(((word_i, word_j), score))
            if word_i not in new_dic:
                cols_to_cos.append(int(cws_clean[word_i]))
                new_dic[word_i]=int(cws_clean[word_i])
                count+=1
            if word_j not in new_dic:
                cols_to_cos.append(int(cws_clean[word_j]))
                new_dic[word_j]=int(cws_clean[word_j])
                count+=1
        else:
            pass
    cols_cos=list(sorted(set(cols_to_cos))) 
    list_dic=sorted(new_dic.items(), key=lambda x: x[1]) #sort dictionary by increasing values
    
    new_dic_s={}
    index=0
    for (e,ind) in list_dic:
        new_dic_s[e]=index
        index+=1
    
    
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_ppmi_all_pats_mat_200.npz')
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round_allpats_200.npz')
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/all_pats_python_mat_200.npz')
    
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_mat_simlex_allpats.npz')
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round_allpats.npz') #best
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/all_pats_python_mat.npz')
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/svd_reduced_mat_300_allpats.npz')
    
    nmat=mat.T #important for svd, for regular doesn't matter since the matrix is symmetric
    col_normed_mat = pp.normalize(nmat.tocsc(), axis=0) 
    del mat
        
    new_mat=col_normed_mat[:,cols_cos]
    simL= new_mat.T * new_mat # cosine values matrix (count, count)
    rowsL, colsL = simL.nonzero()
    
    pair_list_human.sort(key=lambda x: - x[1])  ###sorts the list according to the human scores in descreasing order
    coverage = len(pair_list_human)

    model_scores = {} #{key=(word1,word2), value=cosine_sim_betwen_vectors}    
    cnt=0
    for (x, y) in pair_list_human: #pair_list_human:((vanish,disappear),9.8)
        (word_i, word_j) = x
        cnt+=1
        if cnt % 10 == 0: #n_lines divides in 10000 without remainder
            #print  str(cnt)+'\r'
            sys.stdout.flush()
        r=new_dic_s[word_i]
        c=new_dic_s[word_j]
        cosval=simL[r,c]
        model_scores[(word_i, word_j)] = round(cosval,2)
     
    spearman_human_scores=[]
    spearman_model_scores=[]
         
    for position_1, (word_pair, score_1) in enumerate(pair_list_human):
        score_2 = model_scores[word_pair]
        spearman_human_scores.append(score_1)
        spearman_model_scores.append(score_2)  
     
    spearman_rho = spearmanr(spearman_human_scores, spearman_model_scores)
    #spearman_rho_test = spearmanr([1,2,3,4,5], [5,6,7,8,7])  # (corr=0.82, pval=0.088)
     
    print "The spearman corr is: ",  round(spearman_rho[0], 3)
    print "The coverage is: ", coverage
    
    
    print "i'm here"
#################################################################     
########################

if __name__ == '__main__':
    main()
