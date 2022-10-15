import os
from sklearn.metrics import classification_report
from joblib import load
import pandas as pd

ROOT_FOLDER = os.getcwd()
PROJ_FOLDER = 'Docker_tutorial'
X_TEST_FILE = 'data/X_test.csv'
Y_TEST_FILE = 'data/y_test.csv'
MODEL_FILE = 'model/heart_disease_v1.joblib'

def main():
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, X_TEST_FILE)
    X_test = pd.read_csv(path)
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, Y_TEST_FILE)
    y_test = pd.read_csv(path).to_numpy() # Convert into numpy
    
    print ('Load model.. ')
    path = os.path.join(ROOT_FOLDER, PROJ_FOLDER, MODEL_FILE)
    clf = load(path)

    print ('Evaluate results..')
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

if __name__ == '__main__':
    main()