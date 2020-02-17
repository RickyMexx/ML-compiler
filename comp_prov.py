# each JSON has: {instructions}, {opt}, {compiler}

# MODEL SETTINGS: please set these before running the main #
mode = "opt"  # Labels of the model: [opt] or [compiler]
samples = 3000  # Number of the blind set samples
fav_instrs_in = ["mov"]  # Set of instructions of which DEST register should be extracted [IN]
fav_instrs_eq = ["lea"]  # Set of instructions of which DEST register should be extracted [EQ]
# -------------- #

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import json
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scikitplot as skplt
import matplotlib.pyplot as plt


# Function that parses the input file
# Dataset can be 1 (train) or 2 (blind test)
def processFile(name, i, o, c, dataset):
    with open(name) as f:
        for jsonl in f:
            tmp = json.loads(jsonl)
            i.append(tmp['instructions'])
            if (dataset == 1):
                o.append(tmp['opt'])
                c.append(tmp['compiler'])

    for idx in range(len(i)):
        start = ""
        for word in i[idx]:
            tmp = ""
            arr = word.split()
            flag = True
            for ins1 in fav_instrs_in:
                if ins1 in arr[0]:
                    flag = False
                    tmp = arr[0] + " " + arr[1] + " "

            for ins2 in fav_instrs_eq:
                if ins2 == arr[0]:
                    flag = False
                    tmp = arr[0] + " " + arr[1] + " "

            if flag:
                tmp = arr[0] + " "

            start += tmp

        i[idx] = start


# Function that deals with the csv file
# Index can be: 1 (opt) or 2 (compiler)
def produceOutput(name, out, index):
    if index != 0 and index != 1:
        return

    lines = list()
    with open(name, "r") as fr:
        rd = csv.reader(fr)
        lines = list(rd)

    if not lines:
        lines = [None] * samples
        for i in range(samples):
            lines[i] = ["--", "--"]

    for i in range(samples):
        lines[i][index] = out[i]

    with open(name, "w") as fw:
        wr = csv.writer(fw)
        wr.writerows(lines)


if __name__ == "__main__":
    index = 1 if mode == "opt" else 0
    instrs = list()
    opt = list()
    comp = list()
    processFile("train_dataset.jsonl", instrs, opt, comp, 1)

    vectorizer = CountVectorizer(min_df=5)
    #vectorizer = TfidfVectorizer(min_df=5)

    X_all = vectorizer.fit_transform(instrs)
    y_all = opt if mode == "opt" else comp

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=15)

    #model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=7).fit(X_train, y_train)

    print("Outcomes on test set")
    pred = model.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    ll = log_loss(y_test, model.predict_proba(X_test))
    print("Log Loss: {}".format(ll))

    #skplt.metrics.plot_precision_recall_curve(y_test, model.predict_proba(X_test), title="MOGB")
    #skplt.metrics.plot_confusion_matrix(y_test, pred, normalize=True, title="MOGB")
    #plt.show()

    # Calculating the overfitting
    print("Outcomes on training set")
    pred2 = model.predict(X_train)
    print(confusion_matrix(y_train, pred2))
    print(classification_report(y_train, pred2))

    # Predicting the blind dataset
    b_instrs = list()
    processFile("test_dataset_blind.jsonl", b_instrs, list(), list(), 2)
    b_X_all = vectorizer.transform(b_instrs)
    b_pred = model.predict(b_X_all)

    produceOutput("1743168.csv", b_pred, index)



