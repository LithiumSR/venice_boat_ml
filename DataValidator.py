from DataLearner import BoatLearner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tqdm
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

    def kcrossvalidate(self):
        from DataLoader import Mode
        if self.mode == Mode.detection:
            self._kcrossvalidatedetection()
        else:
            self._kcrossvalidateclassification()

    def _kcrossvalidatedetection(self):
        kf = KFold(n_splits=self.split, shuffle=True)
        kf.get_n_splits(self.set)
        truepositive, truenegative, falsepositive, falsenegative = [0] * 4
        print("Validating using kcross...")
        for train_index, test_index in tqdm.tqdm(kf.split(self.set)):
            X_train, X_test = [], []
            for index in train_index:
                X_train.append(self.set[index])
            for index in test_index:
                X_test.append(self.set[index])
            learner = BoatLearner(X_train, self.kernel)
            learner.learn()
            result = self._validatedetection(learner)
            print(result)
            truepositive += result[0]
            truenegative += result[1]
            falsepositive += result[2]
            falsenegative += result[3]
        truepositive /= self.split
        truenegative /= self.split
        falsepositive /= self.split
        falsenegative /= self.split
        print("Precision: {0:.2f}".format((truepositive / (truepositive + falsepositive))))
        print("Recall: {0:.2f}".format((truepositive / (truepositive + falsenegative))))
        print("False positive rate: {0:.2f}".format((falsepositive / (falsepositive + truenegative))))
        print("Accuracy: {0:.2f}".format(
            (truenegative + truepositive) / (falsepositive + truenegative + truepositive + falsenegative)))

    def _validatedetection(self, learner):
        truepositive, truenegative, falsepositive, falsenegative = [0] * 4
        for elem in self.set:
            tmp = [elem.features.flatten()]
            prediction = learner.classifier.predict(tmp)
            # print(prediction,"but its type is: ", elem.boatType)
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

    def _validateclassification(self, learner, order):
        y_true = []
        y_prediction = []
        for elem in self.set:
            tmp = [elem.features.flatten()]
            prediction = learner.classifier.predict(tmp)
            y_true.append(elem.boatType)
            y_prediction.append(prediction)
        matrix = confusion_matrix(y_true, y_prediction, labels=order)
        return matrix

    def _kcrossvalidateclassification(self):
        kf = KFold(n_splits=self.split)
        kf.get_n_splits(self.set)
        matrix = None
        order = set()
        for elem in self.set:
            order.add(elem.boatType)
        order = list(order)
        print("Validating using kcross...")
        for train_index, test_index in kf.split(self.set):
            X_train, X_test = [], []
            for index in train_index:
                X_train.append(self.set[index])
            for index in test_index:
                X_test.append(self.set[index])
            learner = BoatLearner(X_train, self.kernel)
            learner.learn()
            result = self._validateclassification(learner, order)
            if matrix is None:
                matrix = result
            else:
                matrix += result
            break
        matrix = matrix / self.split
        matrix = matrix.round()
        np.set_printoptions(suppress=True)
        df = pd.DataFrame(matrix, index=order, columns=order)
        plt.figure(figsize=(30, 14))
        sn.set(font_scale=1.5)
        sn.heatmap(df, annot=True, fmt='g')
        plt.show()
