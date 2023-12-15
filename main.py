import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

attributes = ["G1", "G2", "G3", "studytime", "failures", "absences", "health"]

data = (pd.read_csv("data.csv", sep=";"))[attributes]
predict = "G3"

x = np.array(data.drop([predict], 1))
