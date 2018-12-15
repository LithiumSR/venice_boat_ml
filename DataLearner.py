from sklearn.svm import SVC


class BoatLearner:

    def __init__(self, dataset, kernel="linear"):
        self.classifier = SVC(kernel=kernel)
        self.dataset = dataset

    def learn(self):
        X = []
        y = []

        for element in self.dataset:
            X.append(element.features.flatten())
            y.append(element.boatType)
        self.classifier.fit(X, y)
