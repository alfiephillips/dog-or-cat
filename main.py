import pandas as pd
import numpy as np
import sklearn
import pickle
import matplotlib.pyplot as pyplot

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

ATTRIBUTES = ["G1", "G2", "G3", "studytime", "failures", "absences"]
PREDICTION = "G3"
TEST = False
NUM_TESTS = 10000
TEST_SIZE = 0.1

data = (pd.read_csv("data.csv", sep=";"))[ATTRIBUTES]

x = np.array(data.drop(columns=PREDICTION, axis=1))
y = np.array(data[PREDICTION])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=TEST_SIZE)

if TEST:
    optimum = 0
    for _ in range(NUM_TESTS):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=TEST_SIZE)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        accuracy = linear.score(x_test, y_test)

        if accuracy > optimum:
            optimum = accuracy

            with open("mstudentmodel.pickle", "wb") as file:
                pickle.dump(linear, file)

    print(optimum)

input_model = open("mstudentmodel.pickle", "rb")
linear = pickle.load(input_model)


predictions = linear.predict(x_test)

for index in range(len(predictions)):
    accuracy = "0%"

    if y_test[index] != 0:
        calc = (min(predictions[index], y_test[index]) / max(predictions[index], y_test[index])) * 100
        accuracy = str(np.round(calc, 3)) + "%"

    print(predictions[index], x_test[index], y_test[index], accuracy)

