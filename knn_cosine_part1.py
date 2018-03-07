import codecs
import sys
import re
import numpy as np
import json
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from ast import literal_eval
import re
import collections
import scipy.sparse as ss
import scipy
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import sklearn.preprocessing as pp
from scipy.stats import spearmanr
from sklearn.neighbors import LSHForest
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from operator import itemgetter 
from itertools import combinations
import sklearn.metrics.pairwise
from scipy.sparse import vstack


def main():

    #===========================================================================
    # X = csc_matrix(np.array([[1, 1, 1], [1, 1, 0], [2,2,1]]))
    # A = pp.normalize(X.tocsc(), axis=1) #normalizes by columns [0.577, 0.577, 0.577], [0.707, 0.707, 0]    
    # row_v=A[1,:]
    # sim_test= A*row_v.T 
    # cos1=sklearn.metrics.pairwise.cosine_similarity(A[1,:], A[0,:])
    # cos2=sklearn.metrics.pairwise.cosine_similarity(A[1,:], A[1,:])
    # cos3=sklearn.metrics.pairwise.cosine_similarity(A[1,:], A[2,:])
    #===========================================================================
    #############################################################################################################

    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round_allpats.npz') #0.433
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/all_pats_python_mat.npz') #0.391
    
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
    
    pair_list_human = [] #((word1,word2), human_score), only words that have vecs
    
    fread_simlex=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    
    new_dic={}
    rows_to_smooth=[]
    lines_f = fread_simlex.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    count_words=0
    for line_f in lines_f:
        tokens = line_f.split(",")
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        if word_i in cws_clean and word_j in cws_clean:
            pair_list_human.append(((word_i, word_j), score))
            if word_i not in new_dic:
                rows_to_smooth.append(int(cws_clean[word_i]))
                new_dic[word_i]=(int(cws_clean[word_i]), count_words)
                count_words+=1
            if word_j not in new_dic:
                rows_to_smooth.append(int(cws_clean[word_j]))
                new_dic[word_j]=(int(cws_clean[word_j]), count_words)
                count_words+=1
                
  
    #nmat=mat.T #important for svd, for regular doesn't matter since the matrix is symmetric
    col_normed_mat = pp.normalize(mat.tocsc(), axis=1) #(800, 700K)
    all_cos_mat = lil_matrix((len(new_dic), mat.shape[1]) )
    #del mat
    cnt=0
    for w in new_dic:
        print cnt
        cnt+=1
        v=col_normed_mat[new_dic[w][0]]
        simL=col_normed_mat*v.T  #simL[new_dic[w][0]]=1
        all_cos_mat[new_dic[w][1]]=np.round(simL,6).T
   
    ss.save_npz("all_cos_mat.npz", csr_matrix(all_cos_mat))   
    
    
    ######################################################################################################
    print "i'm' here "   
    #ss.save_npz(mat_file_res, csr_matrix(matres))    
        
    print "i'm here"
#################################################################     
########################

if __name__ == '__main__':
    main()
