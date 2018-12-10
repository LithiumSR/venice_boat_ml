import sys
from DataLoader import BoatLoader, Mode, NetworkArchitecture
from DataValidator import DataValidator

pair = BoatLoader(Mode.classification, network=NetworkArchitecture.VGG19).loadset()
validator = DataValidator(pair[0]+pair[1], 2, Mode.classification)
validator.kcrossvalidate()
sys.exit(0)