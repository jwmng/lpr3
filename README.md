# `lpr3`
_Very basic license plate detector/recognizer_

At present, this works in very specific cases with very specific pictures.
No active development, I'm a bit too busy for this.

## Usage

### Recognize

`python app.py <image> <config file>`

Configuration options are documented in `config.ini`

### Training

`python train.py <template folder> <outfile>`

The template folder should contain black-on-white PNGs of each character  with 
the filenames matching the character name.

Example files for Dutch license plates are included in the `chars` folder.


## Analysis approach

_(may change)_

Steps:

1. Pre-warp
1. Binarization
1. Contour finding (find contour -> approach polynomial -> get bounding rectangle)
1. Plate ROI extraction (threshold size/shape -> rotate -> crop)
1. Plate character segmentation (binarization -> contour detection -> shape matching)
1. Character template matching (binary template matching)
