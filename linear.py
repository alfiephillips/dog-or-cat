import pandas as pd
import numpy as np
import sklearn
import pickle
import matplotlib.pyplot as pyplot

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

from main import Model

# Constants for data attributes and prediction target
ATTRIBUTES = ["G1", "G2", "G3", "studytime", "failures", "absences"]
PREDICTION = "G3"
NUM_TESTS = 10000
TEST_SIZE = 0.1


class LinearRegressionModel(Model):
    """
    A linear regression model for predicting a target variable based on specified attributes.
    Inherits from a base Model class.
    """

    def __init__(self, data, filename, attributes, prediction, num_tests=10, test_size=0.1):
        """
        Initialize the LinearRegressionModel.

        :param data: DataFrame containing the dataset.
        :param filename: Name of the file to save the trained model.
        :param attributes: List of feature names used for prediction.
        :param prediction: Name of the target variable.
        :param num_tests: Number of iterations for training to find the best model.
        :param test_size: Fraction of data to be used as test set.
        """
        super().__init__(data, filename, num_tests, test_size)
        self.attributes = attributes
        self.prediction = prediction

        self.linear = None
        self.predictions = []

        self.x_test = []
        self.x_train = []
        self.y_test = []
        self.y_train = []

        self.data = self.data[attributes]

        self.x = np.array(self.data.drop(columns=self.prediction, axis=1))
        self.y = np.array(self.data[self.prediction])

        if len(self.errors) > 0:
            print(self.errors[:-1])

    def download(self):
        """
        Load a trained model from a file. If the file is not found, returns None.

        :return: Loaded model or None.
        """
        try:
            with open("data/" + self.filename, "rb") as input_model:
                self.linear = pickle.load(input_model)
                self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
                    self.x, self.y, test_size=self.test_size)

                return self.linear
        except FileNotFoundError as error:
            self.errors.append(error)
            return None

    def upload(self):
        """
        Save the trained model to a file. If the file is not found, returns None.

        :return: None.
        """
        try:
            with open("data/" + self.filename, "wb") as file:
                return pickle.dump(self.linear, file)

        except FileNotFoundError as error:
            self.errors.append(error)
            return None

    def create(self):
        """
        Train the model multiple times to find the best performing one based on accuracy.
        The best model is saved to a file.

        :return: The highest accuracy achieved among all iterations.
        """
        optimum = 0
        for _ in range(self.num_tests):
            self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
                self.x, self.y, test_size=self.test_size)

            self.linear = linear_model.LinearRegression()
            self.linear.fit(self.x_train, self.y_train)

            accuracy = self.linear.score(self.x_test, self.y_test)

            if accuracy > optimum:
                optimum = accuracy
                self.upload()

        return optimum

    def test(self):
        """
        Test the model by making predictions on the test set and printing the results.
        """
        print(self.x_test)
        self.predictions = self.linear.predict(self.x_test)

        for index in range(len(self.predictions)):
            accuracy = "0%"

            if self.y_test[index] != 0:
                calc = (min(self.predictions[index], self.y_test[index]) / max(self.predictions[index],
                                                                               self.y_test[index])) * 100
                accuracy = str(np.round(calc, 3)) + "%"

            print(self.predictions[index], self.x_test[index], self.y_test[index], accuracy)

    def show(self, attribute, y_label="Y axis"):
        """
        Show the collected dataset.

        :param attribute: The name of the attribute to plot on the X-axis.
        :param y_label: The label for the Y-axis. Defaults to 'Y axis'.
        :return: Matplotlib graph (ggplot style).
        """
        style.use("ggplot")
        scatter = pyplot.scatter(self.data[attribute], self.data[self.prediction], alpha=0.5)  # alpha for opacity

        pyplot.xlabel(attribute)
        pyplot.ylabel(y_label)
        pyplot.title(f"Scatter Plot of {attribute} vs. {self.prediction}")  # Add a title

        pyplot.grid(True)

        pyplot.show()