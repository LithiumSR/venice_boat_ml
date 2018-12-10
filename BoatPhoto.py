import os
class BoatPhoto:

    def __init__(self, boatType, features, filename):
        self.boatType = boatType
        self.features = features
        self.filename = filename

    def __repr__(self):
        return "(Type:{},filename:{}, features:{})".format(self.boatType, os.path.basename(os.path.normpath(self.filename)), self.features.shape)
