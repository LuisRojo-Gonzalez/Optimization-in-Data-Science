#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:09:40 2019

@author: luisrojo
"""

######### Support vector machines

#Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

#---------------------- Datos propios --------------

#Lectura de datos
data_propia = pd.read_csv('Social_Network_Ads.csv')
X_propio = data_propia.iloc[:, [2, 3]].values
y_propio = data_propia.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_propio, X_test_propio, y_train_propio, y_test_propio = train_test_split(X_propio, y_propio, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_propio = sc.fit_transform(X_train_propio)
X_test_propio = sc.transform(X_test_propio)

######## Formulacion dual
t0 = time.clock();
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_propio, y_train_propio)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000],
               'kernel': ['linear'], 'gamma': [0.001, 0.005, 0.01, 0.1, 0.5]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           #cv = 1,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_propio, y_train_propio)

best_accuracy_dual_propio = grid_search.best_score_
best_parameters_dual_propio = grid_search.best_params_

t_dual_propio = time.clock()-t0

classifier = SVC(C = best_parameters_dual_propio['C'], kernel = 'linear', gamma = best_parameters_dual_propio['gamma'], random_state = 0)
classifier.fit(X_train_propio, y_train_propio)

# Predicting the Test set results
y_pred_propio = classifier.predict(X_test_propio)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dual_propio = confusion_matrix(y_test_propio, y_pred_propio)

beta_dual_propia = classifier.coef_
beta0_dual_propia = classifier.intercept_

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_propio, y_train_propio
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title(", ".join(["Training set dual", "C = %f" % best_parameters_dual_propio['C'], "kernel = linear", "gamma = %f" % best_parameters_dual_propio['gamma']]))
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Training_dual_propio.png')
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_propio, y_test_propio
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title(", ".join(["Test set dual","C = %f" % best_parameters_dual_propio['C'], "kernel = linear", "gamma = %f" % best_parameters_dual_propio['gamma']]))
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Test_dual_propio.png')
plt.show()

######## Formulacion primal

t0 = time.clock();

from sklearn.svm import LinearSVC
classifier = LinearSVC(C = 1, max_iter = 10000, verbose = 1, dual = False)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           #cv = 1,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_propio, y_train_propio)

t_primal_propio = time.clock()-t0

best_accuracy_primal_propio = grid_search.best_score_
best_parameters_primal_propio = grid_search.best_params_

classifier = LinearSVC(C = best_parameters_primal_propio['C'], max_iter = 10000, verbose = 1, dual = False)
classifier.fit(X_train_propio, y_train_propio)

# Predicting the Test set results
y_pred_propio = classifier.predict(X_test_propio)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_primal_propio = confusion_matrix(y_test_propio, y_pred_propio)

beta_primal_propio = classifier.coef_
beta0_primal_propio = classifier.intercept_

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_propio, y_train_propio
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title(", ".join(["Training set primal", "C = %f" % best_parameters_primal_propio['C']]))
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Training_primal_propio.png')
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_propio, y_test_propio
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title(", ".join(["Test set primal", "C = %f" % best_parameters_primal_propio['C']]))
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('Test_primal_propio.png')
plt.show()

#---------------------- Datos generados --------------

#Lectura de datos
data_generada = pd.read_csv('Generados.csv')
X_generada = data_generada.iloc[:, 0:4].values
y_generada = data_generada.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_generada, X_test_generada, y_train_generada, y_test_generada = train_test_split(X_generada, y_generada, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_generada = sc.fit_transform(X_train_generada)
X_test_generada = sc.transform(X_test_generada)

######## Formulacion dual

t0 = time.clock();

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_generada, y_train_generada)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000],
               'kernel': ['linear'], 'gamma': [0.001, 0.005, 0.01, 0.1, 0.5]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           #cv = 1,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_generada, y_train_generada)

t_dual_generada = time.clock()-t0

best_accuracy_dual_generada = grid_search.best_score_
best_parameters_dual_generada = grid_search.best_params_

classifier = SVC(C = best_parameters_dual_generada['C'], kernel = 'linear', gamma = best_parameters_dual_generada['gamma'], random_state = 0)
classifier.fit(X_train_generada, y_train_generada)

# Predicting the Test set results
y_pred_dual_generada = classifier.predict(X_test_generada)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dual_generada = confusion_matrix(y_test_generada, y_pred_dual_generada)

beta_dual_generada = classifier.coef_
beta0_dual_generada = classifier.intercept_

######## Formulacion primal

t0 = time.clock();

from sklearn.svm import LinearSVC
classifier = LinearSVC(C = 1, max_iter = 10000, verbose = 1, dual = False)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           #cv = 1,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_generada, y_train_generada)

t_primal_generada = time.clock()-t0

best_accuracy_primal_generada = grid_search.best_score_
best_parameters_primal_generada = grid_search.best_params_

classifier = LinearSVC(C = best_parameters_primal_generada['C'], max_iter = 10000, verbose = 1, dual = False)
classifier.fit(X_train_generada, y_train_generada)

# Predicting the Test set results
y_pred_primal_generada = classifier.predict(X_test_generada)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_primal_generada = confusion_matrix(y_test_generada, y_pred_primal_generada)

beta_primal_generada = classifier.coef_
beta0_primal_generada = classifier.intercept_
