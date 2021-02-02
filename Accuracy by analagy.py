# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:41:01 2018

@author: AZROUMAHLI Chaimae
"""
# =============================================================================
# importing Libraries 
# =============================================================================
import os
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
import regex
import re
import io

# =============================================================================
# reading and layering the pairs from the OMA project ArEmbed
# =============================================================================

#the list of pairs should be normalized before hand 
#normalizing function from the pre-proceesing steps
def remove_diacritics(text):
    arabic_diacritics = re.compile(" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ", re.VERBOSE)
    text = re.sub(arabic_diacritics,'',text)
    return text 

def normalizing(string):
    a='ا'
    b='ء'
    c='ه'
    d='ي'
    string=regex.sub('[آ]|[أ]|[إ]',a,string)
    string=regex.sub('[ؤ]|[ئ]',b,string)
    string=regex.sub('[ة]',c,string)
    string=regex.sub('[ي]|[ى]',d,string)
    return remove_diacritics(string)

def get_relation_questions(questions_directory):
    questions=[]
    #detecting the files
    testing_files=[f for f in listdir(questions_directory) if isfile(join(questions_directory,f))]
    #Openin,g the directory
    os.chdir(questions_directory)
    #reading and listing the files' content
    for file in testing_files:
        with open(file,'r', encoding='utf-8-sig') as f:
            reader=csv.reader(row.replace('\0','') for row in f)
            paires=list(reader)
            for row in paires:
                questions.append([normalizing(r) for r in row])
            print('%s has been appended'%(file))
    return questions

# =============================================================================
# Importing the dictionnary or the model to test
# =============================================================================
def get_dictionnary(dictionnary_file):
    return np.load(dictionnary_file).item()

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        try:
            data[tokens[0]] = list(map(float,tokens[1:]))
        except:
            pass
    return data

# =============================================================================
# Calculating the accuracy
# =============================================================================

def accuracies_in_a_file(corpus_directory,results_file):
    #getting the dictionnaries to test
    os.chdir(corpus_directory)
    os.chdir(os.path.join('./dictionnaries'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    print('The dictionnary files are read and waiting to be tested')
    os.chdir(corpus_directory)
    #Calculating the accuracy for each file
    with open(results_file,'a',encoding='utf-8-sig') as results:
        writer=csv.writer(results)
        writer.writerow(["dictionnary's name","The list of semantic accuracies","The list of syntactic accuracies"])
    print('The file: %s is created'%(results_file))
    for file in dict_files:
        os.chdir(os.path.join('./dictionnaries'))
        dictionnary=load_vectors(file)
        Semantic_accuracy=get_accuracy(Semantic_questions,dictionnary)
        print("%s 's semantic accuracy is computed "%(file))
        Syntactic_accuracy=get_accuracy(Syntactic_questions,dictionnary)
        print("%s 's syntactic accuracy is computed "%(file))
        os.chdir(corpus_directory)
        #the accuries are list that contains the statiqtics of each relation with the accurucy obtained
        with open(results_file,'a',encoding='utf-8-sig') as results:
            writer=csv.writer(results)
            writer.writerow([file,Semantic_accuracy,Syntactic_accuracy])
        print("%s 's accuracy is computed and noted in the accuracies file")
    return


def get_accuracy(testing_questions,dictionnary_to_test):
    words=[keys for keys in dictionnary_to_test.keys()]
    vectors=dictionnary_to_test
    vocab_size=len(words)
    vocab={w:idx for idx,w in enumerate(words)}
    ivocab={idx:w for idx,w in enumerate(words)}
    vector_dim=len(vectors[ivocab[0]])
    W=np.zeros((vocab_size,vector_dim))
    for word,v in vectors.items():
        W[vocab[word],:]=v
    #normalization des vecteur à des (unit variances or zero means)
    W_norm=np.zeros(W.shape)
    d=(np.sum(W ** 2,1)**(0.5))
    W_norm=(W.T/d).T
    split_size=100 #Memory overflow
    correct_que=0 #number of correct questions
    count_que_an=0 # questions answered
    count_que=0 #count all the questions
    count_que+=len(testing_questions)
    data=[x for x in testing_questions if all(word in vocab for word in x)]
    indices=np.array([[vocab[word] for word in row] for row in data])
    print(indices)
    ind1, ind2, ind3, ind4=indices.T
    predictions=np.zeros((len(indices),))
    num_iter=int(np.ceil(len(indices)/float(split_size))) 
    for j in range(num_iter):
        subset=np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))
        pred_vec=(W_norm[ind2[subset],:] - W_norm[ind1[subset],:] + W_norm[ind3[subset],:])
        #cosine similarity if input W has been normalized
        dist=np.dot(W_norm,pred_vec.T)
        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf
        #predected word index
        predictions[subset]=np.argmax(dist,0).flatten()
    val=(ind4==predictions) 
    count_que_an=len(ind1)
    correct_que=sum(val)
    print('Total accuracy: %.2f%%  (%i/%i)' % (100*correct_que/float(count_que_an), correct_que, count_que))
    return 100*correct_que/float(count_que_an)

# =============================================================================
# Semantic & syntactic questions files
# =============================================================================
print('Loading the semantic and the syntactic files')
Syntactic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Morphosyntacitc analogies"
Semantic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Semantic analogies"
Semantic_questions=get_relation_questions(Semantic_path)
Syntactic_questions=get_relation_questions(Syntactic_path)


# =============================================================================
# CBOW Twitter accuracies in a file
# =============================================================================
print('Starting the accuracy computing for all the dictionnary files')
FastText_directory="/home/ubuntu/embeddings_analysis/FastText"
accuracies_in_a_file(FastText_directory,'Fasttext_accuracy_by_analogy.csv')