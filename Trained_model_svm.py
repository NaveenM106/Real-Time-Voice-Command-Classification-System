import joblib
from sklearn import datasets
from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


jorge = np.loadtxt('/Users/claudiogonzalezarriaga/Documents/Progra_Tec/CuartoSemestre/Modelacion del aprendizaje con IA/Voice Classification/archivo_concatenado.txt')
x = jorge[:, 1:]
y = jorge[:, 0]


## SVM LINEAL
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle = True)

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):
    
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf_cv = SVC(kernel = 'linear')
    clf_cv.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_cv.predict(x_test)

    # Concatenate results of evaluation
    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)   

# Model performance
print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

# Cargar el modelo entrenado desde un archivo
joblib.dump(clf_cv, '/Users/claudiogonzalezarriaga/Documents/Progra_Tec/CuartoSemestre/Modelacion del aprendizaje con IA/Voice Classification/trained_model_svm_linear.pkl')


