# `lpr3`
_Very basic license plate detector/recognizer_

## Usage

### Recognize

`python app.py <image> <config file>`

Configuration options are explained in `config.ini`

### Training

`python train.py <template folder> <outfile>`

The template folder should contain black-on-white PNGs of each character  with 
the filenames matching the character name.

Example files for Dutch license plates are included in `chars`.


## Analysis approach

_(may change)_

Steps:

0. Pre-warp
1. Contour finding
2. Plate ROI selection (shape/size matching)
3. Plate ROI extraction (rotate/crop)
4. Character template matching
