import codecs
import sys
import re
import numpy as np
import json
import h5py
from scipy.sparse import csr_matrix, lil_matrix
import h5sparse
from scipy import sparse
import scipy.sparse as ss
#import tables as tb

PATT_STR="PATT"
CW_SYMBOL="CW"
PATT_ELEMENTS_SEPERATOR="-"
HIGH_FREQUENCY_THR = 0.8 #use constant HIGH_FREQUENCY_THR => 0.002; orig, mine_test=0.8
MIN_FREQ =3 #orig=100, mine_test=3

class Trie(object):
    """ Trie implementation in python 
    """
    
    def __init__(self, ):
        """ So we use a dictionary at each level to represent a level in the hashmap
        the key of the hashmap is the string and the value is another hashmap 
        
        """
        self.node = {}
        self={}
        
    def build(self, pattern):
        """
        Takes a pattern and splits it to its components  
        Arguments:
        - `pattern`: The pattern to be stored in the trie
        """
        node = self.node
        cw_indices=[]
        for cnt, patt in enumerate(pattern.split(PATT_ELEMENTS_SEPERATOR)):
            if patt==CW_SYMBOL:
                cw_indices.append(cnt)
            if node.has_key(patt) is True:
                node = node[patt]
            else:
                node[patt] = {}
                node = node[patt]
        node[PATT_STR]=[pattern, cw_indices]
        node = node[PATT_STR]

    def sub_trie(self, patt): #returns the sub trie with 'root'=patt
        node = self.node
        if node.has_key(patt) is True:
            subtr=Trie()
            subtr.node=node[patt]
            return subtr
    


def read_patterns_trie(patterns_input_file):
    """ 
    implementing # Read patterns into a Trie data structure.
    my $patterns_trie = read_patterns_trie($patterns_input_file);   
    """

    #patterns_input_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/selected_patterns.dat'
    print "Reading patterns from: ", patterns_input_file
    
    try:
        ifh = codecs.open(patterns_input_file, 'r', 'utf-8')
        #lines = ifh.readlines()
    except:
        print "Patterns could not be loaded from:", patterns_input_file
        return {}
    
    tr=Trie()
    n_patts=0
    for patt_line in ifh:
        n_patts+=1
        patt_line=patt_line.rstrip()
        tr.build(patt_line)
    
    print "Read", n_patts,  " patterns"
    ifh.close()
    return tr


def add_patt_instance(words, start, patt_index, patterns_trie, cws_clean, co_mat, pat_words):
    
    # Pattern found.
    #global num_words
    if patterns_trie.node.has_key(PATT_STR):
        orig_patt_str=patterns_trie.node[PATT_STR][0] #for example: 'CW-and-CW'
        cw_indices=np.asarray(patterns_trie.node[PATT_STR][1]) #for example: [0 2]
        beg_index=start-patt_index
        pattern_words=words[beg_index:beg_index+cw_indices[1]+1] #for example: big and small
        elements=np.array(pattern_words)[cw_indices] #elements[0]=big, elements[1]=small
      
        
        #update the co-occorence mat
        row=cws_clean[elements[0]]
        col=cws_clean[elements[1]]
        
       
        co_mat[row,col]+=1
        co_mat[col,row]+=1
        
        if row not in pat_words:
            pat_words[row]=1
        if col not in pat_words:
            pat_words[col]=1
        
        ### next in the original code: (I don't understand why extra incrementing)
        #$word_vocab->{$elements[1]}++;
        #$elements[0] .= "_r";
        #$context_vocab->{$elements[0]}++;
        
    # Recursion break condition.
    if start== len(words):
        return
    # Return if word is empty or punctuation, same as in function get_cws, check it
    #if len(word)==0 or not re.match(r'^[a-zA-Z]+$', word):  #r'\b[a-zA-Z]+\b', \b-# Assert position at a word boundary
    elif not(len(words[start])) or bool(re.match(r'^(?!.*[a-z]+).*$', words[start])):  #if not(len(word)) or bool(re.match(r'^(?![a-z])+$', word)):
        return
    
    
    # Next word could either be one of the words the continues a pattern, or a wildcard.
    if patterns_trie.node.has_key(words[start]):
        substr=patterns_trie.sub_trie(words[start])
        add_patt_instance(words, start+1,patt_index+1, substr, cws_clean, co_mat, pat_words)
    elif (not bool(re.match(r'^(?!.*[a-z]+).*$', words[start])) ) and ( cws_clean.has_key(words[start]) )  and ( patterns_trie.node.has_key(CW_SYMBOL) ) :
        substr=patterns_trie.sub_trie(CW_SYMBOL)
        add_patt_instance(words, start+1, patt_index+1, substr, cws_clean, co_mat, pat_words)
        

    
def main():
#===============================================================================
#     try:
#         input_files = sys.argv[1]   #file_destination_vectors
#         patterns_input_file = sys.argv[2]
#         context_pairs_output_file= sys.argv[3]
#         word_vocabularty_output_file = sys.argv[4]
#         context_vocabularty_output_file= sys.argv[5]
#     except:
#         print "Usage: $0 <input files (comma separated)> <patterns input file (one pattern per line)> <context pairs output file> <word vocab output file> <context vocab output file> <word count -- optional>\n"
# 
#     if len(sys.argv) > 5:
#          word_count_file=sys.argv[6]
#===============================================================================
    
    hfw_thr = 0.0001

    #input_files="/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus/english_test.txt" #for test
    #input_files="/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news-commentary-v6.en,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/europarl-v6.en,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2007.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2008.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2009.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2010.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2011.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2012.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2013.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2014.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2015.en.shuffled,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/English_Corpus_P/news.2016.en.shuffled"
    input_files="/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/webbase_phrase2.txt"
    patterns_input_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/selected_patterns.dat'
    mat_file='webbase_mat.npz'
    dic_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_a.txt'
    dic_file1='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_aclean.txt'
    pat_words_file='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/pat_words.txt'
    # Read patterns into a Trie data structure.
    patterns_trie = read_patterns_trie(patterns_input_file)
    
    ifs = input_files.split(",")  #ifs=input files, spolits in case there are several input files
      
    cws = json.load(open(dic_file)) #this how you read json
    print "Finished reading content word dictionary its length is:", len(cws)
    
    cws_clean=json.load(open(dic_file1))
    print "Finished reading content word dictionary clean its length is:", len(cws_clean)
    
    count=0
    #cws_num={}
    for w in cws:
        if bool(re.match(r'^[a-z_]+$', w )) :
            cws_clean[w]=count
            count+=1
    
    
    co_mat = lil_matrix((len(cws_clean), len(cws_clean)) )
    #co_mat=np.zeros( shape=(len(cws), len(cws) )   ) #doesn't work for huge size
    
    n_lines = 0
    pat_words={}
    
    for corpus_file in ifs:
        n_lines = 0
        print "Reading ", corpus_file
        try:
            ifh = codecs.open(corpus_file, 'r', 'latin-1')
        except:
            print "Can't open ", corpus_file, "for reading"
            continue
         
        for line in ifh:
            n_lines=n_lines+1
            if n_lines % 10000 == 0: #n_lines divides in 10000 without remainder
                print  str(round(float(n_lines)/1000, 0))+'K'+'\r', 
                sys.stdout.flush()
                
            line=line.strip()
            line=line.lower() #lower case
            #words=re.split("\W+", line) #\W non word, here 18 and in perl 20, re.split("\\W+", line) leaves only words
            words=re.findall(r'\w+|[^\w\s]+', line) #this version the same as perl, \w+ - 1 or more word chars (letters, digits or underscores), | - or, [^\w\s] - 1 char other than word / whitespace
            
            # Search for patterns starting at each word in the sentence.
            end_loop=len(words)-2
            for start in range(0,end_loop):
                add_patt_instance(words, start, 0, patterns_trie, cws_clean, co_mat, pat_words)
        
        ifh.close()
        
    #===========================================================================
    
    
    print "Preparing to write co-occurence mat to file"
    
    ss.save_npz(mat_file, csr_matrix(co_mat))

    print "I'm here"
  
    with open(pat_words_file, 'w') as fl:
        fl.write(json.dumps(pat_words))
#################################################################     
########################

if __name__ == '__main__':
    main()
