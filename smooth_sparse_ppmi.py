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
from sklearn.neighbors import NearestNeighbors


def main():
    
    alpha=7
    mat_file_res='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_mat_simlex.npz'

    pair_list = [] #((word1,word2), human_score), only words that have vecs
    
    fread_simlex=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill

    lines_f = fread_simlex.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    for line_f in lines_f:
        tokens = line_f.split()
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        pair_list.append(word_i)
        pair_list.append(word_j)
    
    upair_list=set(pair_list)
    
    dic_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_all.txt'
    cws_clean = json.load(open(dic_file)) #this how you read json
    
    matres = lil_matrix((len(upair_list), len(cws_clean)) )
    
    missing_words=[]
    rows_to_smooth=[]
    for w in upair_list:
        if w in cws_clean:
            rows_to_smooth.append(cws_clean[w])
        else:
            missing_words.append(w)
            
           
    mat=ss.load_npz('mat_ppmi_round.npz')
       
    nonzeroRows=mat.nonzero()[0]
    unonserosRows=sorted(set(nonzeroRows))
    
    rows_to_knn=sorted(set(unonserosRows).intersection(rows_to_smooth))
    
    matres = lil_matrix((len(cws_clean), len(cws_clean)) )

    mask = np.ones(mat.shape[0], dtype=bool)
    #The ranking your would get with cosine similarity is equivalent to the rank order of the euclidean distance when you normalize all the data points first. 
    nbrs=NearestNeighbors(n_neighbors=250, algorithm='auto', metric='cosine', n_jobs=-1) # uses all cpus
    for ind, r in enumerate(rows_to_knn):
        #print  str(ind)+'\r'
        if ind % 10 == 0: #n_lines divides in 10000 without remainder
            print  str(ind)+'\r'
            sys.stdout.flush()
        row_v=mat[r,:]
        #samples=scipy.sparse.vstack([mat[:r, :], mat[r+1:, :]])
        mask[r] = False
        samples=mat[mask]
        nbrs.fit(samples)
        k_nnn=nbrs.kneighbors(row_v) #returns two arrays, the first is the distance array, and the second is the array of indexes closets to row_v
        sum_knn=(samples[k_nnn[1][0],:] ).sum(axis=0)
        matres[r,:]=mat[r,:]+alpha*sum_knn
    
    
    ss.save_npz(mat_file_res, csr_matrix(matres))
            
    
    print "Missing words from corpus for simlex: ", missing_words
    
    
    
    #print ("I'm here")   
        
    #################################################################     
########################

if __name__ == '__main__':
    main()
