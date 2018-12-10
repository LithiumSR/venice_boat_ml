import sys
from DataLoader import BoatLoader, Mode
from DataValidator import DataValidator

pair = BoatLoader(Mode.detection, vstype="VaporettoACTV").loadset()
validator = DataValidator(pair[0]+pair[1], 2, Mode.detection, vstype="VaporettoACTV")
validator.kcrossvalidate()
sys.exit(0)