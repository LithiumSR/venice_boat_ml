import glob
import os
from enum import Enum
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np


class Mode(Enum):
    detection = 0
    classification = 1


class BoatLoader:

    def __init__(self, mode, vstype="Water"):
        self.model = VGG16(weights='imagenet', include_top=False)
        self.mode = mode
        self.vstype = vstype.strip()

    def parseArgosTraining(self, directory):

        list = []
        folders = [x[0] for x in os.walk(os.getcwd() + "/data/" + directory)]
        for folder in folders:
            files = glob.glob(folder + "/*.jpg")
            for file in files:
                img = image.load_img(file, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = self.model.predict(img_data)
                from BoatPhoto import BoatPhoto

                if self.mode == Mode.classification:
                    elem = BoatPhoto(os.path.basename(os.path.normpath(folder)).strip(), vgg16_feature, file)
                    list.append(elem)
                elif self.mode == Mode.detection:
                    boatType = os.path.basename(os.path.normpath(folder)).strip()
                    if boatType != self.vstype:
                        if self.vstype == "Water":
                            boatType = "Boat"
                        else:
                            boatType = "Not" + self.vstype
                    elem = BoatPhoto(boatType, vgg16_feature, file)
                    list.append(elem)
        return list

    def parseArgosTesting(self, directory):
        list = []
        folders = [x[0] for x in os.walk(os.getcwd() + "/data/" + directory)]
        for folder in folders:
            files = glob.glob(folder + "/*.jpg")
            truth = open(folder + "/ground_truth.txt", "r")
            map_truth = {}
            line = truth.readline()
            while line:
                entry = line.split(";")
                line = truth.readline()
                if self.mode != Mode.detection:
                    if "Snapshot" in entry[1].strip():
                        continue
                    if entry[1].strip() == "Mototopo corto":
                        entry[1] = "Mototopo"
                elif self.vstype == "Water" and self.mode == Mode.detection:
                    if entry[1].strip() != "Water":
                        entry[1] = "Boat"
                elif self.vstype != "Water" and self.mode == Mode.detection:
                    if "Snapshot" in entry[1].strip():
                        continue
                    if entry[1].strip() != self.vstype:
                        entry[1] = "Not" + self.vstype
                map_truth[entry[0].strip()] = entry[1].strip().replace(" ", "").replace(":", "")
            truth.close()
            for file in files:
                img = image.load_img(file, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = self.model.predict(img_data)
                from BoatPhoto import BoatPhoto

                if os.path.basename(os.path.normpath(file)) in map_truth:
                    elem = BoatPhoto(map_truth[os.path.basename(os.path.normpath(file))], vgg16_feature, file)
                    list.append(elem)
        return list

    def loadset(self):
        list2 = self.parseArgosTesting("testing")
        list1 = self.parseArgosTraining("training")
        print(list2)
        print(list1)
        print("training size:{} testing size:{}".format(list1.__len__(), list2.__len__()))
        return list1, list2
