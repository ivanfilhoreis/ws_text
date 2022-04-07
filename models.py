#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.spatial.distance import cosine
import numpy as np

def cosseno(x,y):
    
    dist = cosine(x,y)
    if np.isnan(dist):
        return 1
    return dist

def get_models():
    
    models = {"MLP - Relu-camadas (1) e neurônios (50)"       : MLPClassifier(activation='relu', hidden_layer_sizes=(50,)),
            "MLP - Logistic-camadas (1) e neurônios(50)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(50,)),
            "MLP - TanHip-camadas (1) e neurônios(50)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(50,)),
            "MLP - Relu-camadas(3) e neurônios (50)"       : MLPClassifier(activation='relu', hidden_layer_sizes=(50,50,50)),
            "MLP - Logistic-camadas (3) e neurônios(50)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(50,50,50)),
            "MLP - TanHip-camadas (3) e neurônios(50)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(50,50,50)),
            "MLP - Relu-camadas(6) e neurônios (50)"       : MLPClassifier(activation='relu', hidden_layer_sizes=(50,50,50,50,50,50)),
            "MLP - Logistic-camadas (6) e neurônios(50)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(50,50,50,50,50,50)),
            "MLP - TanHip-camadas (6) e neurônios(50)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(50,50,50,50,50,50)),
            "MLP - Logistic-camadas (1) e neurônios(150)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(150,)),
            "MLP - TanHip-camadas (1) e neurônios(150)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(150,)),
            "MLP - Relu-camadas(3) e neurônios (150)"       : MLPClassifier(activation='relu', hidden_layer_sizes=(150,150,150)),
            "MLP - Logistic-camadas (3) e neurônios(150)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(150,150,150)),
            "MLP - TanHip-camadas (3) e neurônios(150)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(150,150,150)),
            "MLP - Relu-camadas(6) e neurônios (150)"       : MLPClassifier(activation='relu', hidden_layer_sizes=(150,150,150,150,150,150)),
            "MLP - Logistic-camadas (6) e neurônios(150)"  : MLPClassifier(activation='logistic', hidden_layer_sizes=(150,150,150,150,150,150)),
            "MLP - TanHip-camadas (6) e neurônios(150)"     : MLPClassifier(activation='tanh', hidden_layer_sizes=(150,150,150,150,150,150)),
            "SVM - kernel linear e random 42" : SVC(kernel = 'linear', random_state=42),
            "SVM - kernel rbf scale e random 42" : SVC(kernel = 'rbf', gamma='scale', random_state=42),
            "SVM - kernel rbf auto e random 42" : SVC(kernel = 'rbf', gamma='auto', random_state=42),
            "SVM - kernel sigmoid auto e random 42" : SVC(kernel = 'sigmoid',gamma='auto', random_state=42),
            "SVM - kernel sigmoid scale e random 42" : SVC(kernel = 'sigmoid', gamma='scale', random_state=42),
            "SVM - kernel poly g2 auto e random 42" : SVC(kernel = 'poly', degree=2, gamma='auto', random_state=42),
            "SVM - kernel poly g2 scale e random 42" : SVC(kernel = 'poly', degree=2, gamma='scale', random_state=42),
            "SVM - kernel poly g3 auto e random 42" : SVC(kernel = 'poly', degree=3, gamma='auto', random_state=42),
            "SVM - kernel poly g3 scale e random 42" : SVC(kernel = 'poly', degree=3, gamma='scale', random_state=42),
            "SVM - kernel poly g4 auto e random 42" : SVC(kernel = 'poly', degree=4, gamma='auto', random_state=42),
            "SVM - kernel poly g4 scale e random 42" : SVC(kernel = 'poly', degree=4, gamma='scale', random_state=42),
            "SVM - kernel poly g5 scale e random 42" : SVC(kernel = 'poly', degree=5, gamma='scale', random_state=42),
            "SVM - kernel poly g5 auto e random 42" : SVC(kernel = 'poly', degree=5, gamma='auto', random_state=42),
            "SVM - kernel poly g6 auto e random 42" : SVC(kernel = 'poly', degree=6, gamma='auto', random_state=42),
            "KNN com k=3" : KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree', metric=cosseno),
            "KNN com k=7" : KNeighborsClassifier(n_neighbors=7,algorithm='ball_tree', metric=cosseno),
            "KNN com k=11" : KNeighborsClassifier(n_neighbors=11,algorithm='ball_tree', metric=cosseno),
            "KNN com k=15" : KNeighborsClassifier(n_neighbors=15,algorithm='ball_tree', metric=cosseno),
            "GNB - smoothing 1e-01 " : GaussianNB(var_smoothing=1e-01),
            "GNB - smoothing 1e-02 " : GaussianNB(var_smoothing=1e-03),
            "GNB - smoothing 1e-03 " : GaussianNB(var_smoothing=1e-05),
            "GNB - smoothing 1e-10 " : GaussianNB(var_smoothing=1e-10),
            "GNB - smoothing 1e-11 " : GaussianNB(var_smoothing=1e-11),
            "GNB - smoothing 1e-12 " : GaussianNB(var_smoothing=1e-12),
            "GNB - smoothing 1e-20 " : GaussianNB(var_smoothing=1e-20),
            "GNB - smoothing 1e-21 " : GaussianNB(var_smoothing=1e-21),
            "GNB - smoothing 1e-22 " : GaussianNB(var_smoothing=1e-22),
            "GNB - smoothing 1e-23 " : GaussianNB(var_smoothing=1e-23),
            "MNB - alpha 1.0 e fit_prior" : MultinomialNB(alpha=1.0,fit_prior=True),
            "MNB - alpha 1.0" : MultinomialNB(alpha=1.0, fit_prior=False),
            "MNB - alpha 0.9 e fit_prior" : MultinomialNB(alpha=0.9, fit_prior=True),
            "MNB - alpha 0.9" : MultinomialNB(alpha=0.9, fit_prior=False),
            "MNB - alpha 0.5 e fit_prior" : MultinomialNB(alpha=0.5, fit_prior=True),
            "MNB - alpha 0.5" : MultinomialNB(alpha=0.5, fit_prior=False),
            "MNB - alpha 0.4 e fit_prior" : MultinomialNB(alpha=0.4, fit_prior=True),
            "MNB - alpha 0.4" : MultinomialNB(alpha=0.4, fit_prior=False),
            "MNB - alpha 0.1 e fit_prior" : MultinomialNB(alpha=0.1, fit_prior=True),
            "MNB - alpha 0.1" : MultinomialNB(alpha=0.1, fit_prior=False),
            "MNB - alpha 0.0 e fit_prior" : MultinomialNB(alpha=0.0, fit_prior=True),
            "MNB - alpha 0.0" : MultinomialNB(alpha=0.0, fit_prior=False)
            }
    
    return models

