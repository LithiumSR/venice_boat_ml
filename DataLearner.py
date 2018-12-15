from sklearn.svm import SVC
from sklearn.svm import LinearSVC


class BoatLearner:

    def __init__(self, dataset, kernel="linear"):
        if kernel == "linear":
            self.classifier = LinearSVC()
        else:
            self.classifier = SVC(kernel=kernel)
        self.dataset = dataset

    def learn(self):
        X = []
        y = []

        for element in self.dataset:
            X.append(element.features.flatten())
            y.append(element.boatType)
        self.classifier.fit(X, y)
