from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import tensorflow as tf
import pickle
import numpy as np
from numpy import savetxt

def kFoldCrossValidation(modelName):
    global model, directory, r2
    mse_per_fold = []
    r2_per_fold = []

    print('------------------------------------------------------------------------')
    print(f'Starting training for {modelName} model ...')

    match modelName:
        case "KNN":
            model = KNeighborsRegressor(n_neighbors=3, weights='uniform')
            directory = "k-fold-knn"
        case "LR":
            model = LinearRegression()
            directory = "k-fold-lr"
        case "SVM":
            model = SVR()
            directory = "k-fold-svm"
        case "RF":
            model = RandomForestRegressor(n_estimators=300, max_features=5)
            directory = "k-fold-rf"


    fold_no: int
    for fold_no in range(1, 11):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} of 10 ...')

        # read current fold distribution
        x_train_fold = np.genfromtxt(f"dataset/k-folds-normalized/x_train/fold_{fold_no}.csv", delimiter=',')
        y_train_fold = np.genfromtxt(f"dataset/k-folds-normalized/y_train/fold_{fold_no}.csv", delimiter=',')
        x_test_fold = np.genfromtxt(f"dataset/k-folds-normalized/x_test/fold_{fold_no}.csv", delimiter=',')
        y_test_fold = np.genfromtxt(f"dataset/k-folds-normalized/y_test/fold_{fold_no}.csv", delimiter=',')

        if modelName is "SVR":
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma': ('auto', 'scale')}

            LinearSVC = GridSearchCV(model, parameters, cv=3)
            LinearSVC.fit(x_train_fold, y_train_fold.ravel())
            print("Best params for SVM: {}".format(LinearSVC.best_params_))

            model = LinearSVC

        else:
            model.fit(x_train_fold, y_train_fold)

        # Save and print MSE and R^2 scores
        r2 = r2_score(y_test_fold, model.predict(x_test_fold))

        r2_per_fold.append(r2)
        print("Score R^2 for fold %d: %f" % (fold_no, r2))

        mse = tf.keras.metrics.MeanSquaredError()
        y_pred = model.predict(x_test_fold)
        mse.update_state(y_test_fold, y_pred.ravel())
        mse_per_fold.append(mse.result().numpy())
        print("Score MSE for fold %d: %f" % (fold_no, mse.result().numpy()))

        # save current model
        print("Saving model for fold %d ..." % fold_no)
        knnPickle = open(f"models/{directory}/fold_{fold_no}_mse_{mse.result().numpy()}.hdf5", 'wb')
        pickle.dump(model, knnPickle)

        print('------------------------------------------------------------------------')

    # Save scores per fold to file
    savetxt(f"models/{directory}/R2_10-fold.csv", r2_per_fold, delimiter=',', fmt="%f")
    savetxt(f"models/{directory}/MSE_10-fold.csv", mse_per_fold, delimiter=',', fmt="%f")

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> R^2: {np.mean(r2_per_fold)}  MSE: {np.mean(mse_per_fold)}')
    print('------------------------------------------------------------------------')