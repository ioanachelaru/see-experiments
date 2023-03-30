from sklearn.metrics import r2_score
from sklearn import ensemble
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from numpy import savetxt
import pickle

features = ['TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Transactions', 'Entities',
            'PointsNonAdjust', 'Adjustment', 'PointsAjust']
core_features = ['Length', 'Transactions', 'Entities', 'PointsNonAdjust', 'PointsAjust']

# Define the model architecture according to Grid Search results
params = {
    'criterion': 'squared_error',
    'learning_rate': 0.002,
    'loss': 'quantile',
    'max_depth': 4,
    'n_estimators': 2000,
    'subsample': 0.1
}


def gridSearchGradientRegressor():
    parameters = {
        'criterion': ['squared_error'],
        'learning_rate': [0.001, 0.002, 0.01, 0.02, 0.03],
        'loss': ['quantile'],
        'max_depth': [4],
        'n_estimators': [1000, 2000, 4000, 6000, 8000, 10000],
        'subsample': [0.1]
    }
    GBR = ensemble.GradientBoostingRegressor()
    grid_GBR = GridSearchCV(estimator=GBR, param_grid=parameters, cv=2, n_jobs=6)

    x_train = np.genfromtxt(f"dataset/k-folds-raw/x_train/fold_8.csv", delimiter=',')
    y_train = np.genfromtxt(f"dataset/k-folds-raw/y_train/fold_8.csv", delimiter=',')

    grid_GBR.fit(x_train, y_train)

    print(" Results from Grid Search ")
    print("\n The best estimator across ALL searched params:\n", grid_GBR.best_estimator_)
    print("\n The best score across ALL searched params:\n", grid_GBR.best_score_)
    print("\n The best parameters across ALL searched params:\n", grid_GBR.best_params_)


def kFoldGradientBoostingRegressor():
    # K-fold Cross Validation model evaluation
    mse_per_fold = []
    r2_per_fold = []

    reg = ensemble.GradientBoostingRegressor(**params)

    for fold_no in range(1, 11):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} of 10 ...')

        # read current fold distribution
        x_train_fold = np.genfromtxt(f"dataset/k-folds-raw/x_train/fold_{fold_no}.csv", delimiter=',')
        y_train_fold = np.genfromtxt(f"dataset/k-folds-raw/y_train/fold_{fold_no}.csv", delimiter=',')
        x_test_fold = np.genfromtxt(f"dataset/k-folds-raw/x_test/fold_{fold_no}.csv", delimiter=',')
        y_test_fold = np.genfromtxt(f"dataset/k-folds-raw/y_test/fold_{fold_no}.csv", delimiter=',')

        # Fit data to model
        reg.fit(x_train_fold, y_train_fold)

        # Compute R2 score
        score = reg.score(x_test_fold, y_test_fold)
        print("Score R^2 for fold %d: %f" % (fold_no, score))
        r2_per_fold.append(score)

        # Compute mse score
        mse = mean_squared_error(y_test_fold, reg.predict(x_test_fold))
        print("Score MSE for fold %d: %f" % (fold_no, mse))
        mse_per_fold.append(mse)

        # save model to file
        print(f"Saving model for fold {fold_no:} ...")
        pickle.dump(reg, open("models/10-fold-gbr/gbr_fold_%d_r2_%f.dat" % (fold_no, score), "wb"))

    # Save scores per fold to file
    savetxt(f"models/10-fold-gbr/R2_10-fold_gbr.csv", r2_per_fold, delimiter=',', fmt="%f")
    savetxt(f"models/10-fold-gbr/MSE_10-fold_gbr.csv", mse_per_fold, delimiter=',', fmt="%f")

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> R^2: {np.mean(r2_per_fold)}  MSE: {np.mean(mse_per_fold)}')
    print('------------------------------------------------------------------------')
