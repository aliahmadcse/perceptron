import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Perceptron:

    def relu_activation_function(self, x):
        if x < 0:
            return 0
        return x

    def step_activation_function(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def label_encoder(self, labels):
        unique_labels = {}
        encode_labels = [0] * len(labels)

        encode = 0
        for index, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels[label] = encode
                encode += 1

            encode_labels[index] = unique_labels[label]

        return encode_labels

    def update_weights(self, weights, change_in_weights):
        return np.add(weights, change_in_weights)

    def output_weights(self):
        weightsFile = open('weights.txt', 'w')
        for index, weight in enumerate(self.weights):
            weightsFile.write('{0}\n'.format(weight))

        weightsFile.close()

    def normalize_feature(self, feature):
        return np.divide(feature, np.amax(feature))

    def calculate_error(self, output, prediction):
        return output - prediction

    def initialize_weights(self, number_of_weights):
        weights = [0] * number_of_weights

        for index, weight in enumerate(weights):
            weights[index] = round(random.random() * 2, 4)

        return weights

    def get_confusion_matrix(self, predictions, outputs):
        matrix = [[0, 0], [0, 0]]
        for i in range(0, len(predictions)):
            matrix[predictions[i]][outputs[i]] += 1
        return matrix

    def output_metrics_results(self, predictions, outputs, file):
        cm = self.get_confusion_matrix(predictions, outputs)
        accuracy = ((cm[0][0] + cm[1][1])/(cm[0][0] +
                                           cm[1][1] + cm[1][0] + cm[0][1]))*100
        precision = ((cm[0][0]/(cm[0][0] + cm[0][1] + 1)))*100
        recall = ((cm[0][0])/(cm[0][0] + cm[1][0] + 1))*100
        file.write("Accuracy : {0}\n".format(accuracy))
        file.write("Precision : {0}\n".format(precision))
        file.write("Recall : {0}\n".format(recall))

    def iteration_results(self, weights, prediction, output, file):

        for i in range(len(weights)):
            file.write('Weights {0} : {1} \n'.format(i, weights[i]))

        file.write('Actual Value : {0} \n'.format(output))
        file.write('Prediction Value : {0} \n'.format(prediction))

    def fit(self, X, y, learning_rate=0.1, epochs=100):

        self.X_train = X
        self.y_train = y

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        self.weights = self.initialize_weights(X.shape[1] + 1)

        epochs_data = []

        resultsFile = open('train_results.txt', 'w')

        for epoch in tqdm(range(0, epochs)):

            learner_check = True
            #print("========================== Epoch : {0} Starts ==========================".format(epoch + 1))
            resultsFile.write(
                "========================== Epoch : {0} Starts ==========================\n\n".format(epoch + 1))

            predictions = []

            for index, instance in enumerate(X_train):

                resultsFile.write(
                    "========================== Iteration : {0} Starts ==========================\n".format(index + 1))
                error = 0

                instance = np.append(instance, -1)
                prediction = self.step_activation_function(
                    np.dot(instance, self.weights))

                self.iteration_results(
                    self.weights, prediction, y_train[index], resultsFile)

                if prediction != y_train[index]:
                    learner_check = False

                    error = self.calculate_error(y_train[index], prediction)
                    change_in_weight = np.multiply(
                        instance, learning_rate * error)
                    self.weights = self.update_weights(
                        self.weights, change_in_weight)

                val_predictions = self.predict(X_val)
                predictions.append(prediction)
                self.output_metrics_results(
                    val_predictions, y_val, resultsFile)
                resultsFile.write('Error : {0}\n\n'.format(error))

            epochs_data.append(predictions)

            #print("========================== Epoch : {0} Results ==========================")
            resultsFile.write(
                "========================== Epoch : {0} Results ==========================\n".format(epoch + 1))

            self.output_metrics_results(val_predictions, y_val, resultsFile)

            #print("========================== Epoch : {0} Ends ==========================".format(epoch))
            resultsFile.write(
                "========================== Epoch : {0} Ends ==========================\n".format(epoch + 1))

            if(learner_check):
                break

        resultsFile.close()
        self.output_weights()

    def predict(self, X):

        self.X_test = X

        predictions = []
        for instance in X:
            instance = np.append(instance, -1)
            prediction = self.step_activation_function(
                np.dot(instance, self.weights))
            predictions.append(prediction)
        return predictions


if '--Learning' in sys.argv:

    # Importing The Dataset

    dataset = pd.read_csv(r'{0}'.format(sys.argv[1]), header=None)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Making Perceptron Model And Preprocessing Data

    perceptron = Perceptron()
    X[:, 0] = perceptron.normalize_feature(X[:, 0])
    X[:, 1] = perceptron.normalize_feature(X[:, 1])
    X[:, 2] = perceptron.normalize_feature(X[:, 2])
    X[:, 3] = perceptron.normalize_feature(X[:, 3])
    X[:, 4] = perceptron.normalize_feature(X[:, 4])
    X[:, 5] = perceptron.normalize_feature(X[:, 5])
    X[:, 6] = perceptron.normalize_feature(X[:, 6])
    X[:, 7] = perceptron.normalize_feature(X[:, 7])

    perceptron.fit(X, y, epochs=500)

elif '--Test' in sys.argv:
    # Importing The Dataset

    dataset = pd.read_csv(r'{0}'.format(sys.argv[1]))
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Making Perceptron Model And Preprocessing Data

    perceptron = Perceptron()
    X[:, 0] = perceptron.normalize_feature(X[:, 0])
    X[:, 1] = perceptron.normalize_feature(X[:, 1])
    X[:, 2] = perceptron.normalize_feature(X[:, 2])
    X[:, 3] = perceptron.normalize_feature(X[:, 3])
    X[:, 4] = perceptron.normalize_feature(X[:, 4])
    X[:, 5] = perceptron.normalize_feature(X[:, 5])
    X[:, 6] = perceptron.normalize_feature(X[:, 6])
    X[:, 7] = perceptron.normalize_feature(X[:, 7])

    weights = []

    weightsFile = open('weights.txt', 'r')
    for line in weightsFile.readlines():
        weights.append(float(line))

    perceptron.weights = weights

    results_file = open('test_results.txt', 'w')

    predictions = perceptron.predict(X)
    perceptron.output_metrics_results(predictions, y, results_file)

    results_file.close()

else:
    print("Invalid Command Line Arugments. Please Try Again :)")
