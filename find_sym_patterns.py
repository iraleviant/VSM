# -*- coding: utf-8 -*-
import re, math, collections, itertools, os
import nltk
import operator
#from nltk.classify import NaiveBayesClassifier
#from nltk.metrics import BigramAssocMeasures
import codecs
import sys
##########################################################
######################### constants  #####################
##########################################################
# A regular expression of a word.
#WORD_RE = qr/^[a-z\_\$]++$ #convert to python, ???#next if ($w =~ /^[^a-z]++$/ or not length($w));  ##if bool(re.match(r'^(?!.*[a-z]+).*$', word)):
#re.match(r'^(?!.*[a-z]+).*$', w)) python previous version
# Split token (to separate words and punctuation)
SPLIT_TOKEN = "[ \t]++|\b"  #convert to python

# A (fake) enum of word types:
HFW = 1
CW = 0
NO_WORD = -1
CW_SYMBOL = "CW"
MIN_PATT_LENGTH = 3


class Dict:
    def __init__(self, dict,pattern_edges, cws_list):
        self.dict = dict
        self.pattern_edges=pattern_edges
        self.cws_list = cws_list

dicti={}
patt_edg={}   
cws_list=[]
p1 = Dict(dicti, patt_edg, cws_list) #this is the object i want to pass to functions
#p1.dict={}
#p1.pattern_edges={}

def main():

    infile ="/home/ira/Google Drive/IraTechnion/PhD/patterns/english_test"  # Input file of plain text to compute set of SPs.
    n_hfws=100 # Number of high frequency words (HFWs) that can serve as pattern elements.
    n_cws=10000   # Number of content words (CWs) for computing the M measure (for efficiency)
    outfile="output_python.txt"  # Output file for selected SPs.
    m_thr = 0.05   # Threshold of M measure for selecting SPs, The only real important parameter is m_thr
    top_m_thr = 0.05
    max_pattern_length = 7   # Maximum pattern length to consider.
    min_num_of_edges_per_pattern = 3   # Minimum of edge types for a pattern to be considered a candidate. (5000)
    n_pattern_candidates = 5000   # Number of patterns to considered (the N most frequent patterns)
    top_n_lines = 1000000 # Use only the top N lines for computing vocabulary and list of pattern candidates.
    min_edge_frequency = 1 # 3   # Minimal frequency for edge to be considered in the 
                                    # graph construction.
    merge_sps= True  # Optional: merge patterns that are a longer version of 
                                    #             another selected SP. 
    lc = False   # Convert text to lower case

    # First, generate a HFW/CW dictionary.  high frequency words (HFWs)/content words (CWs)
    print ("Generating HFW/CW dictionary from ", infile) 
    hfw_dict={}
    cws={}
    (hfw_dict, cws) = gen_HFW_dict(infile, n_hfws, n_cws, lc, top_n_lines)  #implement this function
    
    pattern_candidates=[] 
    # Now, get list of symmetric pattern candidates.
    print ("Getting symmetric pattern candidates.") #error in this func
    
    pattern_candidates= get_pattern_candidates(infile, hfw_dict, cws, max_pattern_length, n_pattern_candidates, lc, top_n_lines )
    #$pattern_candidates = get_pattern_candidates($if, $hfw_dict, $cws, $max_pattern_length, $n_pattern_candidates, $lc, $top_n_lines);

    write_vocab(pattern_candidates, "patterns_python.txt")
    
    # Third, collect edges for all pattern candidates.
    print ("Getting pattern edges")
    pattern_edges = read_pattern_edges(infile, hfw_dict, cws, pattern_candidates, min_num_of_edges_per_pattern, max_pattern_length, lc)
        
    # Fourth, select symmetric patterns.
    print ("Selecting symmetric patterns.")
    selected_patterns = select_sps(pattern_edges, min_edge_frequency, m_thr)
        
    # Merge SPs that contain other SPs (e.g., "between CW and CW" contains "CW and CW", so we omit it.
    if (merge_sps):
        print ("Merging patterns" )
        selected_patterns = merge_sptrs(selected_patterns)
        

    # Last, write selected patterns to output file.
    print ("Writing selected patterns to ",outfile )
    write_sps(outfile, selected_patterns)
    
    print("I'm here")

    
    return



def merge_sptrs(sps):
    
    sps_list= list(reversed(list(sps.keys()) ))  #or try selected_sps[sps_list[end -i-1]]
    
    selected_sps={}
    
    end= len(sps_list)
    for i in range(0,end ):
        good = 1
        for j in range(0, end):
            if i==j: continue
            if re.search(sps_list[j], sps_list[i]): #searches the right in the left side
                print (sps_list[i]," contains ", sps_list[j], ". Skipping it.\n")
                good=0
        if good:
            selected_sps[sps_list[i]] = sps[sps_list[i]]
        
        
    return selected_sps


# Traverse input file and select high frequency words.
def  gen_HFW_dict(infile, n_hfws, n_cws, lc, top_n_lines):
    
    #n_hfws = shift; # Number of high frequency words (HFWs) that can serve as pattern elements.
    #n_cws = shift;  # Number of content words (CWs) for computing the M measure (for efficiency)
    
    vocab={} #ditionary    
    line_ctr = 0
    word_counter = 0
    
    #####################stopped here               ##########################
    #First, generate a unigram dictionary.
    try:
        ifh = codecs.open(infile, 'r', 'utf-8')
    except:
        print ("Could not be loaded from:", infile)
        return {}
    #my $ifh = new IO::File($if) or die "Cannot open $if for reading";
    
    for line in ifh:
        if top_n_lines!=True or line_ctr < top_n_lines:
            line_ctr+=1
            if line_ctr % 10000 == 0: #n_sent divides in 10000 without remainder
                print ( str(round(float(line_ctr)/1000, 0))+'K'+'\r', )
                sys.stdout.flush()
            line=line.rstrip()
            if lc:
                line=lc(line) #lower case line
 
            #words=line.split(" ")# re.split(SPLIT_TOKEN, line) # change split token to be a python regex , my @words = split(SPLIT_TOKEN, $line);
            #words=re.split(r'[\s\t]+|[a-zA-Z]+', line)
            #words = re.sub("[^\w ]", "", line).split()
            
            #words=re.findall(r'[^\W]+|[^\W\s]+', line)
            words=re.findall(r'\w+|[^\w\s]+', line) #this is the closest to perl
            #words=re.split(r'(\W+)', line ) #this is the second best
            #===================================================================
            
            for w in words:
                if not bool(re.match(r'^[a-z_$]+$', w)): #only words that include a-z or _ or $ included special word dont go through like confédération
                #if not bool(re.match(r'[^\W0-9]+$', w)): #catches all the foreign words besides $
                    continue
                else:
                    word_counter+=1
                    if w not in vocab:
                        vocab[w]=1
                    else:
                        vocab[w]+=1
                    
    write_vocab(vocab, "vocab.txt")
    
    compare_dictionaries("vocab.txt", "/home/ira/Google Drive/IraTechnion/PhD/perl/perl_patts/vocab_perl.txt")
    
    print("Read ", line_ctr,"lines and ", len(vocab), "words" )
    hfws={}                
    n_words=0
    cws={}
    # Now, leave only high frequency words.
    sorted_words=sorted(vocab.items(), key=lambda x:- x[1]) #- for decreasing from big to small
    for k in sorted_words: # k is a tuple ("the", 464)
        n_words+=1
        if  n_words < n_hfws:
            hfws[k[0]]=vocab[k[0]]/word_counter
        elif n_words<n_cws:
            cws[k[0]]=1
        else:
            break   #last
 

    return (hfws,cws)


# Traverse input file and select high frequency pattern candidates with exactly 2 context words.
def get_pattern_candidates(infile, hfw_dict, cws, max_pattern_length, n_pattern_candidates, lc, top_n_lines):
    
    line_ctr=0
    # First, generate a pattern dictionary.
    try:
        ifh = codecs.open(infile, 'r', 'utf-8')
    except:
        print ("Could not be loaded from:", infile)
        return {}
    p1.dict ={}
    
    for line in ifh:
        if top_n_lines!=True or line_ctr < top_n_lines:
            line_ctr+=1
            if line_ctr % 10000 == 0: #n_sent divides in 10000 without remainder
                print ( str(round(float(line_ctr)/1000, 0))+'K'+'\r', )
                sys.stdout.flush()
            #add_pattern_instance_func(dic, st, extra_param)  add_pattern_instance_func(dict, st, cws)
            
            
            extract_patterns(line, lc, max_pattern_length, hfw_dict, cws,  1 , p1.dict)
            #extract_patterns($line, $lc, $max_pattern_length, $hfw_dict, $cws, \&add_pattern_instance_func, \%dict);

    # Now, leave only patterns with high enough frequency.
    patterns={}
    print("Found ", len(p1.dict.keys()), " patterns."  )
    n_patterns = 0
    
    sorted_dict=sorted(p1.dict.items(), key=lambda x:-x[1]) #sorts the words in stats based to their fequency(values) from top to bottom 
    for k in sorted_dict:  #k should be CW the CW"
        patterns[k[0]] = k[1]/float(line_ctr)
        n_patterns+=1
        if n_patterns == n_pattern_candidates:
            break


    return patterns

def extract_patterns(line, lc, max_pattern_length, hfw_dict, cws, func, dict):
    line=line.rstrip()
    if lc:
        line=lc(line) #lower case line
        
    #words=re.split(SPLIT_TOKEN, line)  
    words=re.findall(r'\w+|[^\w\s]+', line) #this is the closest to perl
    types=[]
    
    # A (fake) enum of word types:
    #use constant HFW => 1;
    #use constant CW => 0;
    #use constant NO_WORD => -1;
    for w in words:
        #print(w)
        if w in hfw_dict.keys():
            types.append(HFW) #append to the begining of the list
        #if not bool(re.match(r'^[a-z_$]+$', w)): 
        elif not bool(re.match(r'^[a-z_$]+$', w)) or  (w not in cws.keys()) :  #elsif ($w !~ WORD_RE or not exists $cws->{$w}) errpr
            types.append(NO_WORD)
        else:
            types.append(CW)   #never gets here
    
    if len(words)<MIN_PATT_LENGTH: #MIN_PATT_LENGTH => 3
        return

    end= len(words)- MIN_PATT_LENGTH + 1

    #for i in range(0, end):append
    i=0
    while i< end:
        has_hfw=0
        stop=0
        p1.cws_list=[] #my @cws;  #what here
        patt_words=[]
    # First, take the first MIN_PATT_LENGTH-1 words that must appear in each pattern. #the first two words 
        for k in range (i, i+MIN_PATT_LENGTH-1): #IndexError: list index out of range
            if types[k]==NO_WORD:
                # No point trying longer patterns, as we reached a non-word.
                stop = 1
                i=k+1 #changed
                break
            elif types[k]==CW:
                # If both CWs are identical, stopping.
                if (len(p1.cws_list) == 1 and words[k]==p1.cws_list[0]) :
                #if (@cws == 1 and $words[$k] eq $cws[0]) {
                    stop=1
                    i=i+1
                    break
                p1.cws_list.append(words[k])
                patt_words.append(CW_SYMBOL)
            else:
                patt_words.append(words[k])
                has_hfw=1
         
                
        if stop:
            continue
        
        end2=min(i+max_pattern_length,len(words) ) #changed here increased by one
        
        for k in range(i+MIN_PATT_LENGTH-1, end2):
            #print("i is:", i)
            #print("k is:",k )
            if types[k]==NO_WORD:  # No point trying longer patterns, as we reached a non-word.
                break
            elif types[k]==CW: # Each pattern candidate must have exactly 2 CWs.
                if len(p1.cws_list)==2: # No point trying longer patterns, as we already have 3 cws here.
                    stop=1
                    break
                elif len(p1.cws_list)==1 and words[k]==p1.cws_list[0] :# If both CWs are identical, stopping.       
                    break
                p1.cws_list.append(words[k])
                patt_words.append(CW_SYMBOL)
            else:
                patt_words.append(words[k])
                has_hfw=1
             
            # This pattern has 2 CWs: check if it as a candidate.
            if len(p1.cws_list)==2 and has_hfw:
                st= " ".join(patt_words) 
                #func_name=add_pattern_instance_func   #### missimg
                #add_pattern_instance_func(dict, st, cws)  ##not sure it works
                if func==1:
                    add_pattern_instance_func(dict, st, p1.cws_list)
                else:
                    add_edges_func(dict, st, p1.cws_list)   
                #next line here  $func->($dict, $str, \@cws);
                
        i=i+1


def add_pattern_instance_func(dict, st, extra_param):
    if st not in dict:
        dict[st]=1
    else:
        dict[st]+=1  
    
def add_edges_func(dict, st, extra_param):
    if st in dict.keys():
        if not extra_param[0] in dict[st]: ### errorrr# unless (exists $dict->{$str}->{$extra_param->[0]}) {
            dict[st][extra_param[0]]={}
            
        if extra_param[1] not in dict[st][extra_param[0]]:
            dict[st][extra_param[0]][extra_param[1]]=1  ###????
        else:
            dict[st][extra_param[0]][extra_param[1]]+=1
    
# Traverse input file and extract edges for each pattern candidate.        
def read_pattern_edges(infile, hfw_dict, cws, pattern_candidates, min_num_of_edges_per_pattern, max_pattern_length, lc):
    line_ctr = 0
    
    p1.pattern_edges={}
    #p1.pattern_edges.fromkeys(list(pattern_candidates.keys()) ,None )
    for k in pattern_candidates.keys():
        p1.pattern_edges[k]={}
    
    
    try:
        ifh = codecs.open(infile, 'r', 'utf-8')
    except:
        print ("Could not be loaded from:", infile)
        return {}
    
    for line in ifh:
        line=line.rstrip()
        line_ctr+=1
        if line_ctr % 10000 == 0: #n_sent divides in 10000 without remainder
            print ( str(round(float(line_ctr)/1000, 0))+'K'+'\r', )
            sys.stdout.flush()
                
        extract_patterns(line, lc, max_pattern_length, hfw_dict, cws,  0 , p1.pattern_edges)  
        #extract_patterns($line, $lc, $max_pattern_length, $hfw_dict, $cws, \&add_edges_func, \%pattern_edges);
    for k in p1.pattern_edges.copy().keys(): #can't chnage the original while iterating that's why copy
        if compute_n_edges(p1.pattern_edges[k])< min_num_of_edges_per_pattern: #changed the min to 3
            del(p1.pattern_edges[k])
    
    return p1.pattern_edges

def compute_n_edges(patt):
    n=0
    for k in patt.keys(): # foreach my $k (keys %$patt) 
        #n=n+int(patt[k])  #  $n += scalar (keys %{$patt->{$k}});  ####error
        n=n+len(patt[k].keys())
    return n

# Compute m measure for each pattern candidate.
def select_sps(pattern_edges, min_edge_frequency, m_thr ):
    
    m=[]
    m_measures={}
    for patt in pattern_edges.keys():
        m= compute_m(pattern_edges[patt], min_edge_frequency)
        #m_thr = 0.05   # Threshold of M measure for selecting SPs, The only real important parameter is m_thr
        if m>=m_thr:
            m_measures[patt] = m
            
    return m_measures


# Compute m measure for a single pattern candidate.
def compute_m(edges, min_edge_frequency ):  #errrorrr
    n_symmetric_edges = 0
    n_asymmetric_edges = 0
    
    for w1 in edges.keys(): 
        for w2 in  edges[w1].keys():  
            if edges[w1][w2] <   min_edge_frequency:  #next if $edges->{$w1}->{$w2} < $min_edge_frequency;
                continue           
            
            if w2 in edges.keys() and w1 in edges[w2].keys() and edges[w2][w1] >= min_edge_frequency:
                n_symmetric_edges+=1
            
            else:
                n_asymmetric_edges+=1
            
    if n_symmetric_edges == 0 :
        return 0
    
    # Each symmetric edge is counted twice, so dividing number of these edges by half.
    n_symmetric_edges /= float(2)
    
    m = float(n_symmetric_edges)/(n_symmetric_edges+n_asymmetric_edges)
    
    return m

def write_sps(of, select_sps):
    
    try:
        ifh = codecs.open(of, 'w', 'utf-8')
    except:
        print ("Cannot open for writing", of)
        return {}
    # 
    
    #sotred_words=sorted(vocab.keys()(), key=lambda x:-x[1]) \ from top to bottom, sort {$vocab{$b} <=> $vocab{$a}} keys %vocab) 
    sotred_sps=sorted(select_sps.items(), key=lambda x:-x[1])
    for k in  sotred_sps:
        ifh.write(str(k)+'\n')
    
    ifh.close()
    
def usage(options, mandatory, help):
    for k in mandatory:
        if k=="ARRAY":
            for k2 in k:
                if not (k2 in options.keys()):
                    print("Mandatory option ", k2 ,"is not optional")
                    raise Exception('Mandatory option k2 is not optional')
                elif k not in options.keys():
                    print("Mandatory option ", k ,"is not optional")
                    raise Exception('Mandatory option k is not optional')
    
    options["h+"]=help #$options->{"h+"} = \$help;
    #result = GetOptions(options) #my $result = GetOptions(%$options); import argparse
    result =" "  #chnage that
    
    if (not result or help):
        getopt_gen_message(options, mandatory)
    
    for k in mandatory:
        if k =="ARRAY":
            found=0
            for k2 in k:
                if k2 in options.keys():
                    found=1
                    break
            if not found:
                getopt_gen_message(options, mandatory)
        elif k not in options.keys():
            getopt_gen_message(options, mandatory)
        
def getopt_gen_message(options, mandatory):
    
    print("0. \n")
    sorted_options=sorted(options.keys(), key=lambda x:-x[1])
    for k in sorted_options:
        print("-",k," ")
        if k in options.keys():
            print (options[k])
        for field in mandatory:
            if field ==k and k not in options.keys():
                print(" [MANDATORY]")
        print("\n")
        
    # die "\n";
    
def write_vocab(dict, output_file):
    
    sorted_vocab=sorted(dict.items(), key=lambda x:-x[1]) #sorted_vocab is a tuple ('cute', 4)
    
    try:
        ofh = codecs.open(output_file, 'w', 'utf-8')
    except:
        print ("Can't open ", output_file, "for writing")
        return
    
    # Add dummy </s> node.
    ofh.write('<\/s> 0\n')
    for k in sorted_vocab:
        ofh.write(k[0])
        ofh.write(' ')
        ofh.write(str(k[1]))
        ofh.write('\n')
    
    ofh.close()
                
#####################################################################################################
def compare_dictionaries(file1, file2):
    #file1='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_allpats_python_order.dat'
    fread=codecs.open(file1) #python_file
    python_dic={} #python
    
    lines_f = fread.readlines()[1:]
    for line_g in lines_f:
        line_f=line_g.strip()
        line=line_f.split(" ")
        python_dic[line[0]]=line[1]
    print ("Finished reading content word dictionary its length is:", len(python_dic))
    
    fread=codecs.open(file2)
    perl_dic={}
    
    lines_f = fread.readlines()[1:]
    for line_g in lines_f:
        line_f=line_g.strip()
        line=line_f.split(" ")
        perl_dic[line[0]]=line[1]
    
    print ("length of dic1 is: ", len(python_dic))   
    print ("length of dic1 is: ", len(perl_dic))
    commons = set(python_dic).intersection(set(perl_dic))
    print ("length of commons is: ", len(commons))
    diff_list = list(set(perl_dic.keys()) - commons) #elements that exist in p erl but not in python
    print ("The number of keys in perl that do not exist in python is:", len(diff_list))
    
    
    cnt=0
    for k in commons:
        if python_dic[k]!= perl_dic[k]:
            cnt+=1
            print("for the key: ", k, "the value is different in: ",int(perl_dic[k])-int(python_dic[k]) )
    print ("The number of differnt values for the same keys is:",cnt)       
    
    cnt=0
    for k in perl_dic:
        if k not in python_dic:
            cnt+=1
            print("This key is in perl but not in python: ", k)
    
    for k in python_dic:
        if k not in perl_dic:
            cnt+=1
            print("This key is in python but not in perl: ", k)
    print ("The number of differnt keys is:",cnt)
    print()


#################################################################################################
#################################################################################################
if __name__ == '__main__':
    main()
