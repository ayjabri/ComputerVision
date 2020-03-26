#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:06:38 2020

@author: aymanjabri
"""
#%% Pre_Run
import pandas as pd
import os
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.sparse import save_npz,load_npz,csr_matrix
from joblib import dump,load


#train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
#test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

train_path = "/Users/aymanjabri/notebooks/Columbia/Week_11_project/aclImdb/train/"
test_path = "/Users/aymanjabri/notebooks/Columbia/Week_11_project/aclImdb/test/"
test_file ='/Users/aymanjabri/notebooks/Columbia/Week_11_project/imdb_te.csv'

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    df = []
    pos = glob(inpath + "pos/*.txt")
    neg = glob(inpath + "neg/*.txt")
    for file in pos:
        with open(file,mode='r',encoding = "ISO-8859-1") as r:
            for line in r:
                txt = [line,1]#.lower(),1]
                df.append(txt)
    for file in neg:
        with open(file,mode='r',encoding = "ISO-8859-1") as r:
            for line in r:
                txt = [line,0]#.lower(),0]
                df.append(txt)
    imdb = pd.DataFrame(df,columns=['text','polarity'])
    if mix:
        imdb = imdb.sample(frac=1).reset_index(drop=True)
    imdb.to_csv(outpath+name,index_label='row_number')
    return imdb
  
def Unigram(path=train_path,in_file='imdb_train.csv',out_file='X_train_unigram.npz',save=True):
    df = pd.read_csv(path+in_file)
    if save:
        unigram_victor = CountVectorizer(ngram_range=(1,1),
                        encoding="ISO-8859-1",strip_accents="ascii",stop_words={'english'})
        unigram_victor.fit(df.text.values)
        x = unigram_victor.transform(df.text.values)
        dump(unigram_victor,'unigram_victor.joblib')
        save_npz(out_file,x)
    else:
        x = load_npz(out_file)
        return x

def Bigram(path=train_path,in_file='imdb_train.csv',out_file='X_train_bigram.npz',save=True):
    df = pd.read_csv(path+in_file)
    if save:
        bigram_victor = CountVectorizer(ngram_range=(1,2),
                        encoding="ISO-8859-1",strip_accents="ascii",stop_words={'english'})
        bigram_victor.fit(df.text.values)
        x = bigram_victor.transform(df.text.values)
        dump(bigram_victor,'bigram_victor.joblib')
        save_npz(out_file,x)
    else:
        x = load_npz(out_file)
        return x


def Tf_Unigram(path=train_path,in_file='imdb_train.csv',out_file='X_train_tf_uni.npz',save=True):
    df = pd.read_csv(path+in_file)
    if save:
        tf_unigram_victor = TfidfVectorizer(ngram_range=(1,1),
                        encoding="ISO-8859-1",strip_accents="ascii",stop_words={'english'})
        tf_unigram_victor.fit(df.text.values)
        x = tf_unigram_victor.transform(df.text.values)
        dump(tf_unigram_victor,'tf_unigram_victor.joblib')
        save_npz(out_file,x)
    else:
        x = load_npz(out_file)
        return x


def Tf_Bigram(path=train_path,in_file='imdb_train.csv',out_file='X_train_tf_bi.npz',save=True):
    df = pd.read_csv(path+in_file)
    if save:
        tf_bigram_victor = TfidfVectorizer(ngram_range=(1,2),
                        encoding="ISO-8859-1",strip_accents="ascii",stop_words={'english'})
        tf_bigram_victor.fit(df.text.values)
        x = tf_unigram_victor.transform(df.text.values)
        dump(tf_unigram_victor,'tf_bigram_victor.joblib')
        save_npz(out_file,x)
    else:
        x = load_npz(out_file)
        return x


#%% Models  

df = pd.read_csv('imdb_train.csv')
y_train = df.polarity.values

''' Unigram Classifier '''
X_train_uni = load_npz('X_train_unigram.npz')
unigram_victorizer = load('unigram_victor.joblib')
sgd_unigram = SGDClassifier(loss="hinge",penalty='l1')
sgd_unigram.max_iter = 10000
sgd_unigram.fit(X_train_uni,y_train)
dump(sgd_unigram,'model1.joblib')


''' Bigram Classifier '''
X_train_bi = load_npz('X_train_bigram.npz')
bigram_victorizer = load('bigram_victor.joblib')
sgd_bigram = SGDClassifier(loss="hinge",penalty='l1')
sgd_bigram.max_iter = 10000
sgd_bigram.fit(X_train_bi,y_train)
dump(sgd_bigram,'model2.joblib')
  

''' Tf-idf Unigram Classifier '''
X_train_tf_uni = load_npz('X_train_tf_uni.npz')
tf_unigram_victor = load('tf_unigram_victor.joblib')
clf_tfd_uni = SGDClassifier(loss='hinge',penalty='l1')
clf_tfd_uni.max_iter = 10000
clf_tfd_uni.fit(X_train_tf_uni,y_train)
dump(clf_tfd_uni,'model3.joblib')


''' Tf-idf Bigram Classifier '''


#%% Predictions
if __name__ == "__main__":
    
    dft= pd.read_csv(test_file,encoding = "ISO-8859-1")    
    
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    
    # X_train_uni = load_npz('X_train_unigram.npz')
    unigram_victorizer = load('unigram_victor.joblib')
    x_test_uni = unigram_victorizer.transform(dft.text.values)
    # sgd_unigram = SGDClassifier(loss="hinge",penalty='l1')
    # sgd_unigram.fit(X_train_uni,y_train)
    sgd_unigram = load('model1.joblib')
    output_unigram = sgd_unigram.predict(x_test_uni)
    output_unigram.tofile('unigram.output.txt',sep='\n')
    

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    
    # X_train_bi = load_npz('X_train_bigram.npz')
    bigram_victorizer = load('bigram_victor.joblib')
    x_test_bi = bigram_victorizer.transform(dft.text.values)
    # sgd_bigram = SGDClassifier(loss="hinge",penalty='l1')
    # sgd_bigram.fit(X_train_bi,y_train)
    sgd_bigram = load('model2.joblib')
    output_bigram = sgd_bigram.predict(x_test_bi)
    output_bigram.tofile('bigram.output.txt',sep='\n')
    
    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigramtfidf.output.txt'''
    
    # X_train_tf_uni = load_npz('X_train_tf_uni.npz')
    tf_unigram_victor = load('tf_unigram_victor.joblib')
    x_test_tf_uni = tf_unigram_victor.transform(dft.text.values)
    # clf_tfd_uni = SGDClassifier(loss='hinge',penalty='l1')
    # clf_tfd_uni.fit(X_train_tf_uni,y_train)
    clf_tfd_uni = load('model3.joblib')
    output_tfd_uni = clf_tfd_uni.predict(x_test_tf_uni)
    output_tfd_uni.tofile('unigramtfidf.output.txt',sep='\n')
    
    # unigramtfidf.output.txt
    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to bigramtfidf.output.txt'''
    
   