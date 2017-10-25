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

lp_map = {}
lp_map["english"] = u"en_"
lp_map["german"] = u"de_"
lp_map["italian"] = u"it_"
lp_map["russian"] = u"ru_"


###Usage

###python evaluate_vsm.py file_word_vector_location language human_word_pairs_scores_file


# def normalise_word_vectors(word_vectors, norm=1.0):
#     """
#     This method normalises the collection of word vectors provided in the word_vectors dictionary.
#     """
#     for word in word_vectors:
#         word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
#         word_vectors[word] = word_vectors[word] * norm
#     return word_vectors


def load_word_vectors(file_destination_vectors, language):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print "Loading word vectors from", file_destination_vectors
    word_dictionary_vecs = {}

    try:
        f = codecs.open(file_destination_vectors, 'r', 'utf-8')
        lines = f.readlines()[1:]  #skips the first line of "number of vecs, dimension"


        for line in lines:
            line = line.split(" ", 1)
            key = line[0].lower()
            try:
                transformed_key = unicode(key) #?
            except:
                print "CANT LOAD", transformed_key
            word_dictionary_vecs[transformed_key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
    except:
        print "Word vectors could not be loaded from:", file_destination_vectors
        return {}

    print len(word_dictionary_vecs), "vectors loaded from", file_destination_vectors

    #return normalise_word_vectors(word_dictionary_vecs)
    return word_dictionary_vecs

def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def simlex_analysis(word_dictionary_vecs, words_human_scores_file, language="german", source="simlex"):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt,
    and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt
    """
    pair_list_human = []    #((word1,word2), human_score), only words that have vecs
    


    fread_simlex=codecs.open(words_human_scores_file, 'r', 'utf-8')  #scores from felix hill

    lines_f = fread_simlex.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"

    for line_f in lines_f:
        tokens = line_f.split()
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        #word_i = lp_map[language] + word_i
        #word_j = lp_map[language] + word_j

        if word_i in word_dictionary_vecs and word_j in word_dictionary_vecs:
            pair_list_human.append(((word_i, word_j), score))
        else:
            pass


    pair_list_human.sort(key=lambda x: - x[1])  ###sorts the list according to the human scores in descreasing order
    coverage = len(pair_list_human)

    model_list = [] #list:[index, ((word1,word2), model_score) ]
    model_scores = {} #{key=(word1,word2), value=cosine_sim_betwen_vectors}

    for (x, y) in pair_list_human:
        (word_i, word_j) = x
        current_distance = distance(word_dictionary_vecs[word_i], word_dictionary_vecs[word_j]) #computes the cosine similarity
        model_scores[(word_i, word_j)] = current_distance
        model_list.append(((word_i, word_j), current_distance))

    model_list.sort(key=lambda x: x[1]) ###sorts the list according to the model scores in increasing order

    spearman_original_list = []  #human indexes
    spearman_target_list = []    # model indexes corresponding to the human indexes
    spearman_human_scores=[]
    spearman_model_scores=[]
    
    
    for position_1, (word_pair, score_1) in enumerate(pair_list_human):
        score_2 = model_scores[word_pair]
        position_2 = model_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)
        spearman_human_scores.append(score_1)
        spearman_model_scores.append(score_2)        
    
    spearman_rho = spearmanr(spearman_human_scores, spearman_model_scores)
    #spearman_rho_test = spearmanr([1,2,3,4,5], [5,6,7,8,7])  # (corr=0.82, pval=0.088)
    
    #print "done"
    return round(spearman_rho[0], 3), coverage


def main():
    """
    The user can provide the location of the config file as an argument.
    If no location is specified, the default config file (experiment_parameters.cfg) is used.
    """
    try:
        file_word_vector_location = sys.argv[1]   #file_destination_vectors
        language = sys.argv[2]
        human_word_pairs_scores_file= sys.argv[3]
        word_vectors = load_word_vectors(file_word_vector_location, language)
    except:
        print "USAGE: python evaluate_vsm.py file_word_vector_location language human_word_pairs_scores_file"
        return

    print "\n============= Evaluating word vectors for language:", language, " =============\n"

    
    
    simlex_score, simlex_coverage = simlex_analysis(word_vectors,human_word_pairs_scores_file, language)
    print "SimLex-999 score and coverage:", simlex_score, simlex_coverage



if __name__ == '__main__':
    main()
