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


def main():

    #===========================================================================
    # X = csc_matrix(np.array([[1, 1, 1], [1, 1, 0], [2,2,1]]))
    # A = pp.normalize(X.tocsc(), axis=1) #normalizes by columns [0.577, 0.577, 0.577], [0.707, 0.707, 0]
    # #nbrs=NearestNeighbors(n_neighbors=2, algorithm='auto', metric='cosine', n_jobs=-1)
    # nbrs=NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=-1) # uses all cpus
    # nbrs.fit(A)
    # row_v=A[1,:]
    # k_nnn=nbrs.kneighbors(row_v)
    #===========================================================================
    ###############################################################################################################
    alpha=7
    n=250
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/mat_ppmi_round_allpats.npz') #0.433
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/all_pats_python_mat.npz') #0.391
    #nmat=mat.T #important for svd, for regular doesn't matter since the matrix is symmetric
    #normed_mat = pp.normalize(mat.tocsc(), axis=0) #(800, 700K)
    
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
    for line_f in lines_f:
        tokens = line_f.split(",")
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        if word_i in cws_clean and word_j in cws_clean:
            pair_list_human.append(((word_i, word_j), score))
            if word_i not in new_dic:
                rows_to_smooth.append(int(cws_clean[word_i]))
                new_dic[word_i]=int(cws_clean[word_i])
            if word_j not in new_dic:
                rows_to_smooth.append(int(cws_clean[word_j]))
                new_dic[word_j]=int(cws_clean[word_j])
        
    pair_list_human=pair_list_human[0:400] # for test
    #cols_cos=sorted(set(rows_to_smooth)) #693,1596, 1675, 2117, 2290, 2843
    #cols_cos=sorted(set([361945,248945,357095,593200,533805,627531]))
    
    for n in range(100,200, 10):
        #nbrs=NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree', n_jobs=-1) # uses all cpus
        nbrs=NearestNeighbors(n_neighbors=n, algorithm='auto', metric='cosine', n_jobs=-1)
        nbrs.fit(mat)
        for alpha in range(5,20):    
            spr_list=[]
            matres = lil_matrix((len(cws_clean), mat.shape[1]) )
            test_size=0.25
            spr_list=[]
            rs = model_selection.ShuffleSplit(n_splits=3, test_size=test_size, random_state=0)
            print "Alpapha and n are:", alpha, n
            for val_index, test_index in rs.split(pair_list_human):
            #===========================================================================
                new_pair_list=[]
                new_pair_list=list(itemgetter(*test_index)(pair_list_human))
                cols_to_cos=[]
                for (x, y) in new_pair_list: #pair_list_human:((vanish,disappear),9.8)
                    (word_i, word_j) = x
                    cols_to_cos.append(new_dic[word_i])
                    cols_to_cos.append(new_dic[word_j])  
                cols_cos=sorted(set(cols_to_cos))
                # #The ranking your would get with cosine similarity is equivalent to the rank order of the euclidean distance when you normalize all the data points first. 
                #nbrs=NearestNeighbors(n_neighbors=250, algorithm='auto', metric='cosine', n_jobs=-1) # uses all cpus
                for ind, r in enumerate(cols_cos):
                    #print  str(ind)+'\r'
                    if ind % 30 == 0: #n_lines divides in 10000 without remainder
                        print  str(ind)+'\r'
                        sys.stdout.flush()
                    row_v=mat[r,:]
                    k_nnn=nbrs.kneighbors(row_v) #returns two arrays, the first is the distance array, and the second is the array of indexes closets to row_v
                    sum_knn=(mat[k_nnn[1][0][1:],:] ).sum(axis=0)
                    matres[r,:]=mat[r,:]+alpha*sum_knn
                    
                #########################################################################
                #compute cosine with matres #############################################
                #########################################################################
                
                nmat=matres.T #important for svd, for regular doesn't matter since the matrix is symmetric
                col_normed_mat = pp.normalize(nmat.tocsc(), axis=0) 
                
                
                new_mat=col_normed_mat[:,cols_cos]
                simL= new_mat.T * new_mat # cosine values matrix (count, count)
                #rowsL, colsL = simL.nonzero()
                
                pair_list_human.sort(key=lambda x: - x[1])  ###sorts the list according to the human scores in descreasing order
                coverage = len(pair_list_human)
            
                model_scores = {} #{key=(word1,word2), value=cosine_sim_betwen_vectors}    
                cnt=0
                for (x, y) in pair_list_human: #pair_list_human:((vanish,disappear),9.8)
                    (word_i, word_j) = x
                    cnt+=1
                    if new_dic[word_i] not in cols_cos or new_dic[word_j] not in cols_cos:
                        continue
                    #r=new_dic_s[word_i]
                    #c=new_dic_s[word_j]
                    r=list(cols_cos).index(new_dic[word_i])
                    c=list(cols_cos).index(new_dic[word_j])
                    cosval=simL[r,c]
                    model_scores[(word_i, word_j)] = round(cosval,2)
                 
                spearman_human_scores=[]
                spearman_model_scores=[]
                     
                for position_1, (word_pair, score_1) in enumerate(pair_list_human):
                    if word_pair not in model_scores:
                        continue
                    score_2 = model_scores[word_pair]
                    spearman_human_scores.append(score_1)
                    spearman_model_scores.append(score_2)  
                 
                spearman_rho = spearmanr(spearman_human_scores, spearman_model_scores)
                spr_list.append(round(spearman_rho[0], 3))
                #spearman_rho_test = spearmanr([1,2,3,4,5], [5,6,7,8,7])  # (corr=0.82, pval=0.088)
                 
                print "The spearman corr is: ",  round(spearman_rho[0], 3)
            
            #print "Alpapha and n are:", alpha, n
            print "The avg spearman corr is: ", round( sum(spr_list) / float(len(spr_list))  , 3) 
    #print "The coverage is: ", coverage

    #mat_file_res='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_ppmi_all_pats_mat_10fold_knn.npz'
    #ss.save_npz(mat_file_res, csr_matrix(matres))    
        
    print "i'm here"
#################################################################     
########################

if __name__ == '__main__':
    main()
