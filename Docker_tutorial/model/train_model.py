import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

ROOT_FOLDER = os.getcwd()
PROJ_FOLDER = 'Docker_tutorial'
DATASET_FILE = 'data/heart_disease.csv'
X_TEST_FILE = 'data/X_test.csv'
Y_TEST_FILE = 'data/y_test.csv'
MODEL_FILE = 'model/heart_disease_v1.joblib'

def main():
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, DATASET_FILE)
    df = pd.read_csv(path)
    y = df.pop('condition')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    print('Saving X_test, y_test...')
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, X_TEST_FILE)
    X_test.to_csv(path, index = False)
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, Y_TEST_FILE)
    y_test.to_frame().to_csv(path, index = False)

    print ('Training model.. ')
    clf = RandomForestClassifier(n_estimators = 10, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    print ('Saving model..')
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, MODEL_FILE)
    dump(clf, path)

if __name__ == '__main__':
    main()