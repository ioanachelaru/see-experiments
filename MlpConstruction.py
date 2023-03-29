import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
from numpy.random import seed
from sklearn.metrics import r2_score
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import pandas as pd
import pickle

seed(42)

BATCH_SIZE = 48
NUMBER_OF_EPOCHS = 3000

verbosity = 1
learning_rate = 0.002
loss_function = tf.keras.losses.MeanSquaredError()


def testMlpConfiguration():
    x_train_val = np.genfromtxt('dataset/k-folds-normalized/x_train/fold_1.csv', delimiter=',')
    y_train_val = np.genfromtxt('dataset/k-folds-normalized/y_train/fold_1.csv', delimiter=',')

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.9)

    x_test = np.genfromtxt('dataset/k-folds-normalized/x_test/fold_1.csv', delimiter=',')
    y_test = np.genfromtxt('dataset/k-folds-normalized/y_test/fold_1.csv', delimiter=',')

    STEPS_PER_EPOCH = len(x_train) / BATCH_SIZE

    model = Sequential()
    model.add(Dense(16, input_shape=(5,), activation='relu'))
    # output layer
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=loss_function)

    # overview on the model architecture
    print(model.summary())

    # train the model
    history = model.fit(x_train, y_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUMBER_OF_EPOCHS,
                        validation_data=(x_val, y_val))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.savefig("mlp-teste/plots/loss_mlp_valid.png")

    # Save and print MSE and R^2 scores
    r2 = r2_score(y_test, model.predict(x_test))
    print("Score R^2: %f" % r2)

    mse = model.evaluate(x_test, y_test, verbose=0)
    print("Score MSE: %f" % mse)

    # save current model
    model.save("mlp-teste/mlp_validated_mse_%f.hdf5" % mse)


def buildMLP():
    # Define the model architecture
    model = Sequential()
    model.add(Dense(16, input_shape=(5,), activation='relu'))
    # output layer
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate), loss=loss_function)

    return model


def kFoldCrossValidation():
    model = buildMLP()

    mse_per_fold = []
    r2_per_fold = []

    fold_no: int
    for fold_no in range(1, 11):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} of 10 ...')

        # read current fold distribution
        x_train_fold = np.genfromtxt(f"dataset/k-folds-normalized/x_train/fold_{fold_no}.csv", delimiter=',')
        y_train_fold = np.genfromtxt(f"dataset/k-folds-normalized/y_train/fold_{fold_no}.csv", delimiter=',')
        x_test_fold = np.genfromtxt(f"dataset/k-folds-normalized/x_test/fold_{fold_no}.csv", delimiter=',')
        y_test_fold = np.genfromtxt(f"dataset/k-folds-normalized/y_test/fold_{fold_no}.csv", delimiter=',')

        STEPS_PER_EPOCH = len(x_train_fold) / BATCH_SIZE

        # Fit data to model
        history = model.fit(x_train_fold, y_train_fold, steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=NUMBER_OF_EPOCHS)
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.savefig("plots/k-fold-mlp/fold_%d.png" % fold_no)

        # Save and print MSE and R^2 scores
        r2 = r2_score(y_test_fold, model.predict(x_test_fold))
        r2_per_fold.append(r2)
        print("Score R^2 for fold %d: %f" % (fold_no, r2))

        mse = model.evaluate(x_test_fold, y_test_fold, verbose=0)
        mse_per_fold.append(mse)
        print("Score MSE for fold %d: %f" % (fold_no, mse))

        # save current model
        print("Saving model for fold %d ..." % fold_no)
        model.save("models/k-fold-mlp/fold_%d_mse_%f.hdf5" % (fold_no, mse))

        print('------------------------------------------------------------------------')

    # Save scores per fold to file
    savetxt(f"models/k-fold-mlp/R2_10-fold.csv", r2_per_fold, delimiter=',', fmt="%f")
    savetxt(f"models/k-fold-mlp/MSE_10-fold.csv", mse_per_fold, delimiter=',', fmt="%f")

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> R^2: {np.mean(r2_per_fold)}  MSE: {np.mean(mse_per_fold)}')
    print('------------------------------------------------------------------------')


def retModel():
    return models.load_model('models/k-fold-mlp/fold_8_mse_0.115527.hdf5')


def evaluateMLP():
    model = retModel()
    x_test = np.genfromtxt(f"dataset/k-folds-normalized/x_test/fold_8.csv", delimiter=',')
    y_test = np.genfromtxt(f"dataset/k-folds-normalized/y_test/fold_8.csv", delimiter=',')

    y_pred = model.predict(x_test)

    plt.plot(y_test, label='actual')
    plt.plot(y_pred, label='predicted')
    plt.legend()
    plt.savefig("models/k-fold-mlp/mlp_eval_fold_8.png")


def adaptiveBoostingMLP():
    x_train = np.genfromtxt(f"dataset/k-folds-normalized/x_train/fold_8.csv", delimiter=',')
    y_train = np.genfromtxt(f"dataset/k-folds-normalized/y_train/fold_8.csv", delimiter=',')
    x_test = np.genfromtxt(f"dataset/k-folds-normalized/x_test/fold_8.csv", delimiter=',')
    y_test = np.genfromtxt(f"dataset/k-folds-normalized/y_test/fold_8.csv", delimiter=',')

    preped_model = KerasRegressor(build_fn=retModel, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    boosted_model = AdaBoostRegressor(base_estimator=preped_model)
    boosted_model.fit(x_train, y_train.ravel())

    with open('models/k-fold-mlp/boosted_model_fold_8.pkl', 'wb') as f:
        pickle.dump(boosted_model, f)

    score = boosted_model.score(x_test, y_test)
    print("The score of the boosted model is %f" % score)

    y_pred = boosted_model.predict(x_test)

    plt.plot(y_test, label='actual')
    plt.plot(y_pred, label='predicted')
    plt.legend()
    plt.savefig("plots/k-fold-mlp/boosted_mlp_fold_8.png")

    # Save and print MSE and R^2 scores
    scores = [boosted_model.score(x_test, y_test)]
    print("Score R^2: %f" % scores[0])

    scores.append(mean_squared_error(y_test, boosted_model.predict(x_test)))
    print("Score MSE: %f" % scores[1])

    savetxt(f"models/k-fold-mlp/boosted_mlp_r2_mse.csv", scores, delimiter=',', fmt="%f")
