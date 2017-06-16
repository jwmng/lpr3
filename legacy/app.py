import numpy as np
import cv2
import time
from configparser import ConfigParser

import matplotlib.pyplot as plt

t0 = time.time()

## TODO:
# - Enfore rectangles before AR checking?
# - Prewarp
# - Regex post-processing / string template matching
# - Verbosity
# - Use the config

# def process_img(path, config_file="./config.ini"):
    # t0 = time.time()
    # conf = configparser.ConfigParser()
    # conf.read(config_file)

# Settings
candidates = []
target_aspect = 4.72
tol = 0.3
min_size = 200
inverted = False
letter_height = 27
letter_width = 17
letter_tol = 0.3
# inc ex
char_range = (6, 7)

# Load and treshold
print("Step 0")
img = cv2.imread('car.jpg', 0)
print("\t Loaded image (%dx%d)" % (img.shape[1], img.shape[0]))

# Get contours
print("Step 1")
ret, thresh = cv2.threshold(img, 127, 255, 0)
im2, cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("\t Got %d plate contours" % len(cnts))


# Get candidates
print("Step 2")
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w/h
    # print("Shape, edges: %d, aspect: %f size: %d" % (len(cnt), aspect, w*h))
    if ((aspect > (1-tol)*target_aspect) 
        and (aspect < (1+tol)*target_aspect)
        and w*h > min_size):
        candidate = img[y:y+h, x:x+w]
        candidates.append(candidate)

        # nf = cv2.drawContours(img, [cnt], -1, (255, 0, 0), 3)
        # plt.imshow(candidate)
        # plt.show()

print("\t Got %d plate candidates" % len(candidates))

# Threshold candidates
print("Step 4")
chars = []
cand2 = []
for idx, candidate in enumerate(candidates):
    cand_chars = []
    print("\t Candidate %d" % idx)
    ret, t2 = cv2.threshold(candidate, np.mean(candidate), 255, 0)
    # plt.imshow(t2)
    # plt.show()
    if not inverted:
        t2 = 255-t2

    im2, cnts, hier = cv2.findContours(t2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for d in cnts:
        x, y, w, h = cv2.boundingRect(d)
        if ((w > (1-letter_tol)*letter_width)
            and (w < (1+letter_tol)*letter_width)
            and (h > (1-letter_tol)*letter_height)
            and (h < (1+letter_tol)*letter_height)):
            char = candidate[y:y+h, x:x+w]
            fy = 30/char.shape[0]
            char = cv2.resize(char, (0, 0), fx=fy, fy=fy)
            ret, char = cv2.threshold(char, np.mean(char), 255, 0)
            cand_chars.append((char, x))

    cand_chars = [char for char, x in sorted(cand_chars, key=lambda x: x[1])]
    cand2.append((idx, candidate, cand_chars))
    print("\t\t Got %d characters contours" % len(cand_chars))
    if len(cand_chars) not in range(*char_range):
        candidates.pop(idx)
        print("\t\t Rejected candidate")

# OCR letters
print("Step 5")
templates = np.load('templates.npy')[()]
scores = []
for idx, candidate, chars in cand2:
    print("Matching %d chars" % len(chars))
    for char in chars:
        print("Character")
        char_scores = []
        for label, temp_l in templates.items():
            temp = np.array(temp_l)

            min_h = min(temp.shape[0], char.shape[0])
            min_w = min(temp.shape[1], char.shape[1])
            max_h = max(temp.shape[0], char.shape[0])
            max_w = max(temp.shape[1], char.shape[1])

            temp_slice = temp[:min_h, :min_w]
            char_slice = char[:min_h, :min_w]

            matches = np.sum(temp_slice == char_slice)
            score = matches/(max_h*max_w)
            char_scores.append((label, score))
            char_scores = sorted(char_scores, key=lambda x: x[1], reverse=True)[:3]
            # print("\t\t", label, score)
        scores.append(char_scores)
        print("\tThree best character scores:")
        for label, score in char_scores[:3]:
            print("\t\t", label, score)

# Postprocess result
print("Step 6")
print(''.join([char_score[0][0] for char_score in scores]))

print("Step 7")

t_elapsed = time.time() - t0

print("Finished in %s seconds" % t_elapsed)
