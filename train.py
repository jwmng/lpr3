import cv2
import numpy as np
import sys
from pathlib import Path


def train(template_folder, outfile):
    """ Saves a numpy array containing training templates to `outfile`  """

    # Get all `.png` template files in `./templates`
    print("-- Beginning training")
    templates = list(Path(template_folder).glob('*.png'))
    print("Found %d templates" % len(templates))

    # Open each file and append its label and content to `data`
    data = {}
    for template in templates:
        label = template.stem
        data[label] = cv2.imread(str(template), 0)

    # Save data to the outfile path
    np.save(outfile, data)
    print("Done training")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: train.py <template_folder> <outfile>")
        sys.exit()

    train(sys.argv[1], sys.argv[2])
