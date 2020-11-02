import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace
from sklearn.tree import DecisionTreeClassifier


params = {
            'RandomForestClassifier':{'n_estimators':[50,100,150],
                                    'criterion':['gini'],
                                    'max_depth':np.arange(1,50),
                                    'min_samples_split':np.arange(2,7),
                                    'min_samples_leaf':np.arange(1,5),
                                    },
            'KNeighborsClassifier': {
                                    'n_neighbors':np.arange(1,10),
                                    'weights':['uniform','distance'],
                                    'algorithm':['auto'],
                                    'leaf_size':np.arange(1,51),
                                    },
            'DecisionTreeClassifier':{
                                    'criterion':['gini'],
                                    'max_depth':np.arange(1,51),
                                    'min_samples_split':np.arange(2,11),
                                    },
            'LogisticRegression':   {
                                    'penalty':['l1','l2','elasticnet','none'],
                                    'C': [0.1,0.5,1,5,10,50,100],
                                    },
            'SVC':                  {
                                    'C': [0.1,0.5,1,5,10,50,100],
                                    'cache_size': [200],
                                    'gamma': [0.1,0.5,1,3,6,10],
                                    'kernel': ['rbf']
                                    }
            }

class BestModel(object):
    def __init__(self, cv = None, params=None):
        self.__c = []
        if params is not None:
            for clf in params:
                setattr(self, clf+'_Grid', GridSearchCV(eval(clf)( ), param_grid=params[clf], cv=cv))
                self.__c.append(getattr(self, clf+'_Grid'))
        self.params = params
        self.best_classifier = None

    def fit(self,x,y):
        self.results = {}
        r = 0
        for i,classifier in enumerate(self.__c):
            classifier.fit(x,y)
            score = classifier.best_score_
            print(f'classifier {i} has a top score of {score}')
            self.results[list(params.keys())[i]] = [score, classifier.best_params_]
            if score > r:
                self.best_classifier = classifier.best_estimator_
            r = score

