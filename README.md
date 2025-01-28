# SVM
SVM (Support Vector Machine) is a supervised machine learning algorithm used primarily for classification tasks, although it can also be applied to regression. It is one of the most popular and powerful methods for binary classification and is also useful for multi-class problems in some variations. 
Hyperparamter fine tuning:
 Define the hyperparameter grid to tune 
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    'degree': [3, 4, 5]  # Degree for 'poly' kernel, if applicable
