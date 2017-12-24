import ConfigParser
import numpy
import codecs
import sys
import time
import random
import math
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
import scipy.sparse as ss
from sklearn.metrics.pairwise import pairwise_distances

def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def main():
    
    dic_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_all.txt'
    cws_clean = json.load(open(dic_file)) #this how you read json
    
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_mat_simlex.npz') #vectors mat
    fread_human=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    
    
    pair_list_human=[]
    lines_f = fread_human.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    for line_f in lines_f:
        tokens = line_f.split()
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        
        if word_i in cws_clean and word_j in cws_clean:
            pair_list_human.append(((word_i, word_j), score))
        else:
            pass
        
    pair_list_human.sort(key=lambda x: - x[1])  ###sorts the list according to the human scores in descreasing order
    coverage = len(pair_list_human)

    model_list = [] #list:[index, ((word1,word2), model_score) ]
    model_scores = {} #{key=(word1,word2), value=cosine_sim_betwen_vectors}
    
    for (x, y) in pair_list_human:
        (word_i, word_j) = x
        
        #current_distance = distance(mat[cws_clean[word_i]], mat[cws_clean[word_j]]) #computes the cosine similarity
        current_distance=pairwise_distances(mat[cws_clean[word_i]], mat[cws_clean[word_j]], metric = 'cosine')
        model_scores[(word_i, word_j)] = current_distance
        model_list.append(((word_i, word_j), current_distance))

    model_list.sort(key=lambda x: x[1]) ###sorts the list according to the model scores in increasing order

    spearman_original_list = []  #human indexes
    spearman_target_list = []    # model indexes corresponding to the human indexes
    spearman_human_scores=[]
    spearman_model_scores=[]
    
    for position_1, (word_pair, score_1) in enumerate(pair_list_human):
        score_2 = model_scores[word_pair][0][0]
        position_2 = model_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)
        spearman_human_scores.append(score_1)
        spearman_model_scores.append(score_2)  
    
    spearman_rho = spearmanr(spearman_human_scores, spearman_model_scores)
    #spearman_rho_test = spearmanr([1,2,3,4,5], [5,6,7,8,7])  # (corr=0.82, pval=0.088)
    
    print "The spearman corr is: ",  round(spearman_rho[0], 3)
    print "The coverage is: ", coverage
 
    
    print ("I'm here")   
        
#################################################################     
########################

if __name__ == '__main__':
    main()
