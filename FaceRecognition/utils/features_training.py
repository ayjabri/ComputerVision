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


classifiers = {
            'RandomForestClassifier':{'n_estimators':[100],
                                    'criterion':['gini'],
                                    'max_depth':np.arange(1,10),
                                    'min_samples_split':np.arange(1,7),
                                    'min_samples_leaf':np.arange(1,5),
                                    },
            'KNeighborsClassifier': {
                                    'n_neighbors':np.arange(1,10),
                                    'weights':['uniform'],
                                    'algorithm':['auto','KDTree','BallTree'],
                                    'leaf_size':np.arange(1,6),
                                    },
            'DecisionTreeClassifier':{
                                    'criterion':['gini'],
                                    'max_depth':np.arange(1,10),
                                    'min_samples_split':np.arange(2,11),
                                    },
            'LogisticRegression':   {
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
    def __init__(self, classifiers=None):
        self.__c = []
        if classifiers is not None:
            for clf in classifiers:
                setattr(self, clf+'_Grid', GridSearchCV(eval(clf)( ), param_grid=classifiers[clf]))
                self.__c.append(getattr(self, clf+'_Grid'))
        self.classifiers = classifiers
        self.best_classifier = None
        # self.grid = GridSearchCV()

    def fit(self,x,y):
        results = {}
        for i,classifier in enumerate(self.__c):
            classifier.fit(x,y)
            print(f'classifier {i} has a top score of {classifier.best_score}')


b = BestModel(classifiers)
b.fit(x,y)