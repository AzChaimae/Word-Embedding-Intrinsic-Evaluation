#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:54:33 2019

@author: AZROUMAHLI chaimae
"""
# =============================================================================
# Libraries & main function
# =============================================================================
import csv
import os
import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score
import nltk
from nltk.cluster import KMeansClusterer
from sklearn import cluster
import regex
import re
from glove import Glove 

# =============================================================================
# Reading and listing each category and its content
# =============================================================================
#we have to  normalize the categories
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

def get_categories(categories_directory,categories_file):
    os.chdir(categories_directory)
    with open(categories_file,'r',encoding='utf-8-sig') as file:
        reader=csv.reader(row.replace('/0','') for row in file)
        row_categories=list(reader)
    categories=[]
    for row1 in row_categories:
        test=[categories[i][0] for i in range(len(categories))]
        if (row1[0] not in test):
            list_of_content=[]
            list_of_content.append(normalizing(row1[1]))
            for row2 in row_categories:
                if row_categories.index(row1)!=row_categories.index(row2) and row1[0]==row2[0]:
                    list_of_content.append(normalizing(row2[1]))
            categories.append([row1[0],list_of_content])
    return categories

#test
#file_directory="/home/khaosdev-6/AZROUMAHLI Chaimae/Embeddings analysis/Accuracy/My Arabic word-embeddings benchmarks"
#categories=get_categories(file_directory,'word categories Arabe.csv')

# =============================================================================
# Reading and listing the dictionnay
#looking for the vector representation of each content of each category
# =============================================================================


def reading_dictionnary(dictionnary_directory,model_file):
    os.path(dictionnary_directory)
    model=Glove.load(model_file)
    Glove_dict={}
    for key,val in model.dictionary.items():
        Glove_dict.update({key:model.word_vectors[val]})
    return Glove_dict

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

def get_vectors_benchmarks(dictionnary_file,categories_list):
    #os.path(dictionnary_directory)
    vector_representations=load_vectors(dictionnary_file)
    vector_categories=[]
    print(1)
    for row in categories_list:
        content_words=[]
        content_vectors=[]
        for i in range(len(row[1])):
            for key,val in vector_representations.items():
                print(2)
                if key==row[1][i]:
                    print(3)
                    content_vectors.append(val)
                    content_words.append(key)
                print(4)
        vector_categories.append([row[0],content_words,content_vectors])
        print(5)
    return vector_categories

def get_dictionnary_benchmarks(dictionnary_file,dictionnary_directory,categories_list):
    #os.path(dictionnary_directory)
    os.chdir(dictionnary_directory)
    vector_representations=load_vectors(dictionnary_file)
    vector_dictionnary={}
    for row in categories_list:
        for i in range(len(row[1])):
            for key,val in vector_representations.items():
                if key==row[1][i]:
                    vector_dictionnary.update({key:val})
    print('categories representation is generated')
    return vector_dictionnary
#test
#file_directory="/home/khaosdev-6/AZROUMAHLI Chaimae/Embeddings analysis/CBOW training/wiki/CBOW dictionaries"
#dict_representations=get_dictionnary_benchmarks('CBOW_HS_dict_200_3.npy',file_directory,categories)

# =============================================================================
# TRUE CLUSTERS
# =============================================================================
def get_true_clusters(word_representations,word_categories):
    true_word_clusters=[]
    i=0
    for row in word_categories:
        for word in word_representations.keys():
            if word in row[1] and word not in [true_word_clusters[i][1] for i in range(len(true_word_clusters))]: 
                true_word_clusters.append([row[0],word,i])
        i+=1
    return [true_word_clusters[i][2] for i in range(len(true_word_clusters))]

#test
#true_clusters=get_true_clusters(dict_representations,categories)                       

# =============================================================================
# K-mean algorithm to find the predicted clusters
# =============================================================================

#The purity score of the clusters
def purity_score(y_true_clusters,y_predicted_clusters):
    y_voted_labels=np.zeros(y_true_clusters.shape)
    labels=np.unique(y_true_clusters)
    ordered_labels=np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true_clusters[y_true_clusters==labels[k]]=ordered_labels[k]
    labels=np.unique(y_true_clusters)
    bins=np.concatenate((labels,[np.max(labels)+1]),axis=0)
    for cluster_ in np.unique(y_predicted_clusters):
        hist, _ =np.histogram(y_true_clusters[y_predicted_clusters==cluster_],bins=bins)
        winner=np.argmax(hist)
        y_voted_labels[y_predicted_clusters==cluster_]=winner
    return accuracy_score(y_true_clusters,y_voted_labels)

def get_kmeans_predicted_clusters(word_representions,Num_clusters):
    #from dictionnary type to transposed dataframe
    Y=pd.DataFrame(data=word_representions).T
    X=Y.values
    #Clustering the data using sklearn library
    kclusterer = KMeansClusterer(Num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25, avoid_empty_clusters=False)
    predicted_clusters= kclusterer.cluster(X, assign_clusters=True, )
    return predicted_clusters

#test
#predicted_clusters=get_kmeans_predicted_clusters(dict_representations,22)
#purity_score(np.asarray(true_clusters),np.asarray(predicted_clusters))

# =============================================================================
# Ploting the scatter of the clusters
# =============================================================================

def plot_scatter(word_representations):
    plt.rcParams['figure.figsize']=(14,6)
    X=pd.DataFrame(data=word_representations).T.values
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    return

    
def plot_kmeans_predicted_clusters(word_representations,Num_clusters):
    plt.rcParams['figure.figsize']=(14,6)
    #creating the data set
    X=pd.DataFrame(data=word_representations).T.values
    #X=X.values
    Y=np.asarray(get_kmeans_predicted_clusters(word_representations,Num_clusters))
    kmeans = cluster.KMeans(n_clusters=Num_clusters)
    kmeans.fit(X)
    #labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='#050505', s=250)
    return
        
#test
   #purity_score(np.asarray(true_clusters),np.asarray(predicted_clusters))

#plot_kmeans_predicted_clusters(dict_representations,22)           
# =============================================================================
# Calculating the accuracies for all the dictionnaries generated
# =============================================================================
def corpus_category_accuracies(dictionnaries_directory,categories_directory,categories_file,Num_clusters,results_directory,results_file):
    os.chdir(dictionnaries_directory)
    #detecting the creted dictionnaries
    categories=get_categories(categories_directory,categories_file)
    dictionnary_files=[f for f in listdir(dictionnaries_directory) if isfile(join(dictionnaries_directory,f))]
    print('dictionnary files detected')
    with open(results_file,'a',encoding='utf-8-sig') as results:
        writer=csv.writer(results)
        writer.writerow(["dictionnary","accuracy"])
    for dictionnary in dictionnary_files:
        os.chdir(dictionnaries_directory)
        categories_representation=get_dictionnary_benchmarks(dictionnary,dictionnaries_directory,categories)
        true_clusters=get_true_clusters(categories_representation,categories)
        print('true clusters generated')
        predicted_clusters=get_kmeans_predicted_clusters(categories_representation,Num_clusters)
        print('predicted clusters generated')
        accuracy=purity_score(np.asarray(true_clusters),np.asarray(predicted_clusters))
        os.chdir(results_directory)
        with open(results_file,'a',encoding='utf-8-sig') as results:
            writer=csv.writer(results)
            writer.writerow([dictionnary,accuracy])
        print('the purity score of: %.2f is saved in the file'%(accuracy))
    return 

#Change the function parameters
dictionnaries_directory="/home/ubuntu/embeddings_analysis/FastText/dictionnaries"
categories_directory="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks"
categories_file="word categories Arabe.csv"
Num_clusters=22
results_directory="/home/ubuntu/embeddings_analysis/FastText"
results_file="Fasttext_Accuracy_by_categorization.csv"
corpus_category_accuracies(dictionnaries_directory,categories_directory,categories_file,Num_clusters,results_directory,results_file)
