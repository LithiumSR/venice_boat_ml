import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from DataLearner import BoatLearner


class DataValidator:

    def __init__(self, set, split, mode, vstype="Water", kernel="linear"):
        self.set = set
        self.split = split
        self.mode = mode
        self.vstype = vstype.strip()
        self.kernel = kernel

    def defaultvalidate(self):
        from DataLoader import Mode
        testing = list(filter(lambda x: x.partOfTestingSet == True, self.set))
        training = list(filter(lambda x: x.partOfTestingSet == False, self.set))
        print(testing)
        print(training)
        if self.mode == Mode.detection:
            self._defaultvalidatedetection(testing, training)
        else:
            self._defaultvalidateclassification(testing, training)

    def kcrossvalidate(self):
        from DataLoader import Mode
        if self.mode == Mode.detection:
            self._kcrossvalidatedetection()
        else:
            self._kcrossvalidateclassification()

    def _defaultvalidatedetection(self, testing, training):
        print("Validating using default testing set...")
        truepositive, truenegative, falsepositive, falsenegative = [0] * 4
        learner = BoatLearner(training, self.kernel)
        learner.learn()
        result = self._validatedetection(testing, learner)
        print(result)
        truepositive += result[0]
        truenegative += result[1]
        falsepositive += result[2]
        falsenegative += result[3]
        print("Precision: {}".format((truepositive / (truepositive + falsepositive))))
        print("Recall: {}".format((truepositive / (truepositive + falsenegative))))
        print("False positive rate: {}".format((falsepositive / (falsepositive + truenegative))))
        print("Accuracy: {}".format(
            (truenegative + truepositive) / (falsepositive + truenegative + truepositive + falsenegative)))

    def _defaultvalidateclassification(self, testing, training):
        print("Validating using default testing set...")
        order = set()
        for elem in self.set:
            order.add(elem.boatType)
        order = list(order)
        learner = BoatLearner(training, self.kernel)
        learner.learn()
        y_true, y_pred, matrix = self._validateclassification(testing, learner, order)
        print(classification_report(y_true, y_pred, labels=order))
        np.set_printoptions(suppress=True)
        df = pd.DataFrame(matrix, index=order, columns=order)
        plt.figure(figsize=(30, 14))
        sn.set(font_scale=1.5)
        sn.heatmap(df, annot=True, fmt='g')
        plt.show()

    def _kcrossvalidatedetection(self):
        kf = KFold(n_splits=self.split, shuffle=True)
        kf.get_n_splits(self.set)
        truepositive, truenegative, falsepositive, falsenegative = [0] * 4
        print("Validating using kcross...")
        for train_index, test_index in kf.split(self.set):
            X_train, X_test = [], []
            for index in train_index:
                X_train.append(self.set[index])
            for index in test_index:
                X_test.append(self.set[index])
            learner = BoatLearner(X_train, self.kernel)
            learner.learn()
            result = self._validatedetection(X_test, learner)
            print(result)
            truepositive += result[0]
            truenegative += result[1]
            falsepositive += result[2]
            falsenegative += result[3]
        truepositive /= self.split
        truenegative /= self.split
        falsepositive /= self.split
        falsenegative /= self.split
        print("Precision: {}".format((truepositive / (truepositive + falsepositive))))
        print("Recall: {}".format((truepositive / (truepositive + falsenegative))))
        print("False positive rate: {}".format((falsepositive / (falsepositive + truenegative))))
        print("Accuracy: {}".format(
            (truenegative + truepositive) / (falsepositive + truenegative + truepositive + falsenegative)))

    def _validatedetection(self, testing, learner):
        truepositive, truenegative, falsepositive, falsenegative = [0] * 4
        for elem in testing:
            tmp = [elem.features.flatten()]
            prediction = learner.classifier.predict(tmp)
            # print(prediction,"but its type is: ", elem.boatType)
            prediction = prediction[0].strip()
            if prediction == elem.boatType:
                if prediction == self.vstype:
                    truepositive += 1
                else:
                    truenegative += 1
            else:
                if prediction != self.vstype and elem.boatType != self.vstype:
                    truenegative += 1
                elif elem.boatType == self.vstype:
                    falsepositive += 1
                else:
                    falsenegative += 1
        return truepositive, truenegative, falsepositive, falsenegative

    def _validateclassification(self, testing, learner, order):
        y_true = []
        y_prediction = []
        for elem in testing:
            tmp = [elem.features.flatten()]
            prediction = learner.classifier.predict(tmp)
            y_true.append(elem.boatType)
            y_prediction.append(prediction)
        matrix = confusion_matrix(y_true, y_prediction, labels=order)
        return y_true, y_prediction, matrix

    def _kcrossvalidateclassification(self):
        kf = KFold(n_splits=self.split, shuffle=True)
        kf.get_n_splits(self.set)
        matrix = None
        order = set()
        for elem in self.set:
            order.add(elem.boatType)
        order = list(order)
        print("Validating using kcross...")
        all_true = []
        all_pred = []
        for train_index, test_index in kf.split(self.set):
            X_train, X_test = [], []
            for index in train_index:
                X_train.append(self.set[index])
            for index in test_index:
                X_test.append(self.set[index])
            learner = BoatLearner(X_train, self.kernel)
            learner.learn()
            tmp_true, tmp_pred, tmp_matrix = self._validateclassification(X_test, learner, order)
            all_true = all_true+tmp_true
            all_pred = all_pred+tmp_pred
            if matrix is None:
                matrix = tmp_matrix
            else:
                matrix += tmp_matrix
        print(classification_report(all_true, all_pred, labels=order))
        matrix = matrix / self.split
        matrix = matrix.round()
        np.set_printoptions(suppress=True)
        df = pd.DataFrame(matrix, index=order, columns=order)
        plt.figure(figsize=(30, 14))
        sn.set(font_scale=1.5)
        sn.heatmap(df, annot=True, fmt='g')
        plt.show()
