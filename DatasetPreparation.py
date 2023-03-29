from numpy import savetxt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


core_features = ['Length', 'Transactions', 'Entities', 'PointsNonAdjust', 'PointsAjust']


def prepareDataset():
    # read dataset
    data_frame = pd.read_csv('dataset/desharnais.csv', header=0)

    # drop entities with missing data
    data_frame = data_frame.drop([data_frame.index[37], data_frame.index[43],
                                  data_frame.index[65], data_frame.index[74]])

    # return X, Y
    return data_frame[core_features], data_frame['Effort']


def writeDistributionToFile(directory, fold_no, x_train, y_train, x_test, y_test):
    print(f"Saving raw split for fold {fold_no:} ...")

    savetxt(f"{directory}/x_train/fold_{fold_no}.csv", x_train, delimiter=',', fmt='%s')
    savetxt(f"{directory}/y_train/fold_{fold_no}.csv", y_train, delimiter=',', fmt='%s')
    savetxt(f"{directory}/x_test/fold_{fold_no}.csv", x_test, delimiter=',', fmt='%s')
    savetxt(f"{directory}/y_test/fold_{fold_no}.csv", y_test, delimiter=',', fmt='%s')


def kFoldSplit(k):
    x_raw, y_raw = prepareDataset()

    # Define the K-fold Cross Validator
    kFold = KFold(n_splits=k, shuffle=True)
    fold_no = 1

    for train_val, test in kFold.split(x_raw, y_raw):
        # extract validation set from training data
        x_train = x_raw.iloc[train_val]
        y_train = y_raw.iloc[train_val]
        x_test = x_raw.iloc[test]
        y_test = y_raw.iloc[test]
        print('------------------------------------------------------------------------')
        writeDistributionToFile('dataset/k-folds-raw', fold_no, x_train, y_train, x_test, y_test)

        # normalize the data
        scaler_x = StandardScaler()

        # fit only on training data
        scaler_x.fit(x_train)

        x_train_norm = scaler_x.transform(x_train)
        x_test_norm = scaler_x.transform(x_test)

        # reshape data
        y_train = y_train.to_numpy().reshape(-1, 1)
        y_test = y_test.to_numpy().reshape(-1, 1)

        scaler_y = StandardScaler()

        # fit only on training data
        scaler_y.fit(y_train)

        y_train_norm = scaler_y.transform(y_train)
        y_test_norm = scaler_y.transform(y_test)

        print(f"Saving normalized split for fold {fold_no:} ...")
        writeDistributionToFile('dataset/k-folds-normalized', fold_no, x_train_norm,
                                y_train_norm, x_test_norm, y_test_norm)

        fold_no = fold_no + 1
