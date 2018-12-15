import sys
import argparse
from DataLoader import BoatLoader, Mode, NetworkArchitecture
from DataValidator import DataValidator


def main(mode, inputclass, split, kernel):
    pair = BoatLoader(mode, network=NetworkArchitecture.VGG19, vstype=inputclass).loadset()
    validator = DataValidator(pair[0] + pair[1], split, mode, kernel)
    validator.kcrossvalidate()
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Let's do something with the ARGOS dataset!")
    parser.add_argument("mode", nargs='?', default="classification", help='Mode of operation that can be detection or '
                                                                          'classification')
    parser.add_argument("network", nargs='?', default="vgg16", help='Network architecture to do feature extraction')
    parser.add_argument("split", nargs='?', type=int, default=3, help='Number of split in kcross validation (default '
                                                                      'is 3)')
    parser.add_argument("inputclass", nargs='?', default="water", help='Class to analyze when doing detection ('
                                                                       'default is Water)')
    parser.add_argument("svmkernel", nargs='?', default="linear", help='Kernel to be used for SVM (default is the '
                                                                      'linear kernel)')
    args = parser.parse_args()

    modeoperation = args.mode
    networkarchitecture = args.network

    if networkarchitecture.lower().strip() == "vgg16":
        networkarchitecture = NetworkArchitecture.VGG16
    elif networkarchitecture.lower().strip() == "vgg19":
        networkarchitecture = NetworkArchitecture.VGG19
    else:
        raise SyntaxError
    if modeoperation.lower().strip() == "detection":
        modeoperation = Mode.detection
    elif modeoperation.lower().strip() == "classification":
        modeoperation = Mode.classification
    else:
        raise SyntaxError

    print("Using {} for a {} problem".format(args.network.upper(), args.mode.lower()))
    if modeoperation is not Mode.classification:
        print("Input class for detection is {}".format(args.inputclass))
    main(modeoperation, args.inputclass, args.split, args.svmkernel)
