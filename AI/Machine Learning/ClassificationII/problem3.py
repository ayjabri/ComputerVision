#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 06:23:09 2020

@author: aymanjabri
"""

import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report,plot_confusion_matrix
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('default')

# =============================================================================
# One class to solve them all
# =============================================================================

class Classifier(object):
    def __init__(self,input_file,output_file=None):
        self.data = pd.read_csv(input_file)
        self.x = self.data.drop(['label'],axis=1).values
        self.y = self.data.label.values
        '''
        stratify using label "y" to ensure equal 
        distribution of samples across training and test sets
        '''
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            self.x,self.y,test_size=0.4,stratify=self.y)

    def grid(self,estimator,parameters):
        grid = GridSearchCV(estimator,parameters,cv=5)
        grid.fit(self.x_train,self.y_train)
        best_score = grid.best_score_
        test_score = grid.score(self.x_test,self.y_test)
        return best_score,test_score

    def SVM_Linear(self,C=[0.1,0.5,1,5,10,50,100]):
        clf = svm.SVC(kernel='linear')
        parameters = {'C': C,'gamma':('scale','auto')}
        return self.grid(clf,parameters)

    def SVM_Polynomial(self,C=[0.1,1,3],degree=[4,5,6],gamma=[0.1,0.5]):
        clf = svm.SVC(kernel='poly')
        parameters={'C':C,'gamma':gamma,'degree':degree}
        return self.grid(clf,parameters)

    def SVM_RBF(self,C=[0.1,0.5,1,5,10,50,100],gamma = [0.1,0.5,1,3,6,10]):
        clf = svm.SVC(kernel='rbf')
        parameters = {'C': C,'gamma':gamma}
        return self.grid(clf,parameters)

    def Logistic(self,C=[0.1,0.5,1,5,10,50,100]):
        lrc = LogisticRegression()
        parameters = {'C': C}
        return self.grid(lrc,parameters)

    def KNN(self,n_neighbors=np.arange(1,51),leaf_size=np.arange(1,61)):
        knc = KNeighborsClassifier()
        parameters ={'n_neighbors':n_neighbors,'leaf_size':leaf_size}
        return self.grid(knc,parameters)
    
    def DecisionTree(self,max_depth=np.arange(1,51),min_samples_split=np.arange(2,11)):
        dtc = DecisionTreeClassifier()
        parameters ={'max_depth':max_depth,'min_samples_split':min_samples_split}
        return self.grid(dtc,parameters)
    
    def RandomForest(self,max_depth=np.arange(1,51),min_samples_split=np.arange(2,11)):
        rfc = RandomForestClassifier()
        parameters ={'max_depth':max_depth,'min_samples_split':min_samples_split}
        return self.grid(rfc,parameters)


def main():
    
    in_file = sys.argv[1].lower()

    out_file = sys.argv[2].lower()
    
    f = Classifier(in_file)
    
    svml_score,svml_test = f.SVM_Linear()
    
    svmp_score,svmp_test = f.SVM_Polynomial()
    
    svmr_score,svmr_test = f.SVM_RBF()
    
    lrc_score,lrc_test = f.Logistic()
    
    knc_score,knc_test = f.KNN()
    
    dtc_score,dtc_test = f.DecisionTree()
    
    rfc_score,rfc_test = f.RandomForest()
    
    scores = [['svm_linear',svml_score,svml_test],
          ['svm_polynomial',svmp_score,svmp_test],
          ['svm_rbf',svmr_score,svmr_test],
          ['logistic',lrc_score,lrc_test],
          ['knn',knc_score,knc_test],
          ['decision_tree',dtc_score,dtc_test],
          ['random_forest',rfc_score,rfc_test]]
    
    # scores = ['svm_linear',svml_score,svml_test]
    
    scores = pd.DataFrame(scores)
    
    scores.to_csv(out_file,index=False,header=False)
    
    return 


if __name__=='__main__':
    
    main()