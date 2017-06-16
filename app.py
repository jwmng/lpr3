import configparser
import json
import re
import sys
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt

## TODO:
# - MinAreaRect
# - Approx-to-PolyLDP
# - Fix inversion

# loop over our contours
# for c in cnts:
	# # approximate the contour
	# peri = cv2.arcLength(c, True)
	# approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# # if our approximated contour has four points, then
	# # we can assume that we have found our screen
	# if len(approx) == 4:
		# screenCnt = approx
		# break

def _plot(img, **kwargs):
    plt.figure()
    kwargs.update({'cmap': 'Greys'})
    plt.imshow(img, **kwargs)
    plt.show()


def _check_tol(val, target, tol):
    """ Check if a value is within `tol` of `target` """
    return (val > target*(1-tol)) and (val < target*(1+tol))


def _match_single_character(char, templates):
    """ Template-match a character with each of the templates

    Returns:
        tuple: `(label, score)` with score 0..1, sorted by score descending
    """
    char_scores = []
    _, char = cv2.threshold(char, 0, 255, cv2.THRESH_OTSU)

    for label, temp_l in templates.items():
        template = np.array(temp_l)

        # We only compare the parts of the images that overlap, starting at the
        # left top.
        min_h = min(template.shape[0], char.shape[0])
        min_w = min(template.shape[1], char.shape[1])
        max_h = max(template.shape[0], char.shape[0])
        max_w = max(template.shape[1], char.shape[1])

        temp_slice = template[:min_h, :min_w]
        char_slice = char[:min_h, :min_w]

        # Template matching using mean absolute difference
        matches = np.sum(temp_slice == char_slice)

        # Divide over maximum dimensions to correct for size differences
        # e.g.: `I` and `M` could match perfectly, so we need to divide over
        # the full number of pixels in `M` to get an accurate score
        score = matches/(max_h*max_w)

        char_scores.append((label, score))
        char_scores = sorted(char_scores, key=lambda x: x[1], reverse=True)

    return char_scores


def _match_multi_characters(chars, templates, min_char_score=0):
    """
    Template-match an array of characters `chars` with `templates`
    Rejects characters that have no match better than `min_char_score`
    """
    plate_scores = []
    for char in chars:
        char_scores = _match_single_character(char, templates)

        # Character has no good enough matches
        if char_scores[0][1] < min_char_score:
            continue

        plate_scores.append(char_scores)
    return plate_scores


def read_image(path, conf):
    """ Load image in grayscale

    Returns:
        np.array: Image data in pixels (0...255, grayscale)
    """
    img = cv2.imread(path, 0)
    _plot(img)

    # Normalise, then back to uint8
    img = img.astype(np.float64)
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if conf['verbose']:
        print("Loaded image %s (%dx%d)" % (path, img.shape[1], img.shape[0]))

    if conf['debug']:
        _plot(img, cmap='Greys')

    return (img,)


def get_plate_contours(img, conf):
    """ Find contours in the image `img` """

    # Prewarp
    img = cv2.warpPerspective(img, np.matrix(conf['prewarp']), img.shape[::-1])

    # Contours on threshold image
    _, threshold_image = cv2.threshold(img, 0, 255,
                                       cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(threshold_image,
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    if conf['verbose']:
        print("Got %d contours" % len(contours))

    if conf['debug']:
        contour_img = cv2.drawContours(.3*threshold_image, contours, -1, [255])
        _plot(contour_img, cmap='Greys')

    return (img, contours)


def extract_plate_candidates(img, contours, conf):
    """
    Select the `countours` from `img` that satisfy the constraints from `conf`
    """

    plate_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        candidate = img[y:y+h, x:x+w]

        # Too small
        if cv2.contourArea(cnt) < int(conf['early_reject_min_area']):
            continue

        # Bad width
        if not _check_tol(w, int(conf['width.target']),
                          float(conf['width.tolerance'])):
            continue

        # Bad height
        if not _check_tol(h, int(conf['height.target']),
                          float(conf['height.tolerance'])):
            continue

        # Bad aspect ratio
        aspect_ratio = w/h
        if not _check_tol(aspect_ratio,
                          float(conf['aspect_ratio.target']),
                          float(conf['aspect_ratio.tolerance'])):
            continue

        if conf['debug']:
            _plot(candidate)

        plate_candidates.append(candidate)

    if conf['verbose']:
        print("Got %d plate candidates" % len(plate_candidates))

    return (plate_candidates,)


def segment_plate_candidates(candidates, conf):
    """
    Find contours in `candidate` and return those satisfying config constraints
    """
    candidates_segmented = []
    for candidate in candidates:
        cand_chars = []

        # Otsu binarization
        _, cand_threshold = cv2.threshold(candidate, 0,
                                          255, cv2.THRESH_OTSU)
        if conf['debug']:
            _plot(cand_threshold)

        # Invert if needed
        if not bool(int(conf['inverted'])):
            cand_threshold = 255-cand_threshold

        # Find character contours
        _, char_contours, _ = cv2.findContours(cand_threshold,
                                               cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        if conf['debug']:
            contour_img = cv2.drawContours(.3*cand_threshold, char_contours,
                                           -1, [255])
            _plot(contour_img)

        if conf['verbose']:
            print("Candidate")
            print("\tGot %d character contours" % len(char_contours))

        # ccnt: Character-contour
        for ccnt in char_contours:
            x, y, w, h = cv2.boundingRect(ccnt)

            # Bad character width
            if not _check_tol(w, int(conf['letter_width']),
                              float(conf['letter_size_tolerance'])):
                continue

            # Bad character height
            if not _check_tol(h, int(conf['letter_height']),
                              float(conf['letter_size_tolerance'])):
                continue

            # Possibly a character, resize, threshold, save for matching
            char = cand_threshold[y:y+h, x:x+w]

            scale = 30/char.shape[0]
            char = cv2.resize(char, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)

            if conf['debug']:
                _plot(char)

            # _, char = cv2.threshold(char, np.mean(char), 255, 0)
            cand_chars.append((char, x))

        if conf['verbose']:
            print("\tGot %d characters after processing" % len(cand_chars))

        # Too few candidate characters
        if len(cand_chars) not in range(int(conf['min_num_chars']),
                                        int(conf['max_num_chars'])+1):
            if conf['verbose']:
                print("\tRejected (bad number of characters)")

            continue

        # Sort candidate characters by x coordinate (left -> right)
        # also removes x coordinate
        chars = [char for char, x in sorted(cand_chars, key=lambda x: x[1])]
        candidates_segmented.append(chars)

    return (candidates_segmented,)


def match_plate_candidates(segmented_plates, conf):
    """
    Template match a plate candidate, reject if not enough characters are found
    """
    plate_re = conf['plate_re']

    # Load templates and filter those not matching `plate_re`
    templates = np.load(conf['templates_file'])[()]
    templates = {t: d for t, d in templates.items() if re.match(plate_re, t)}

    processed_plates = []
    for plate_chars in segmented_plates:
        plate_scores = _match_multi_characters(plate_chars, templates,
                                               float(conf['min_char_score']))

        if conf['verbose']:
            print("Got %d characters left" % len(plate_scores))
            print("Best guesses per character:")
            for char in plate_scores:
                print("\t\t".join(["%s: %.2f" % (l, c) for l, c in char[:3]]))

        # Too few characters left
        if len(plate_scores) < int(conf['min_num_chars']):
            if conf['verbose']:
                print("Rejected plate (too few characters)")

            continue
        processed_plates.append(plate_scores)

    return (processed_plates,)


def postprocess(processed_plates, conf):
    """ Convert plate letter scores to human/computer readable format """
    results = []
    json_output = bool(int(conf['json']))
    for idx, plate_scores in enumerate(processed_plates):
        mean_conf = np.mean([char_score[0][1] for char_score in plate_scores])
        best_plate = ''.join([char_score[0][0] for char_score in plate_scores])
        results.append({'plate': best_plate, 'conf': mean_conf})

        if not bool(int(conf['json'])):
            results.append("Plate %d: %s (c=%.2f) "
                           % (idx, best_plate, mean_conf))

    if json_output:
        if conf['openalpr_compat']:
            return json.dumps({'results': results})

        return json.dumps(results)
    return "\n".join(results)


def process_img(path, config_file="./config.ini"):
    """ Main processing function """
    time0 = time.time()
    confp = configparser.ConfigParser()
    confp.read(config_file)

    debug = confp.getboolean('default', 'debug')
    verbose = confp.getboolean('default', 'verbose')

    # Each step function returns a tuple of results, which are the positional
    # arguments to the next step. Each step should also take a `conf` keyword
    # argument which will contain the configuration for that step.
    steps = {'read_image': read_image,
             'get_plate_contours': get_plate_contours,
             'extract_plate_candidates': extract_plate_candidates,
             'segment_plate_candidates': segment_plate_candidates,
             'match_plate_candidates': match_plate_candidates,
             'postprocess': postprocess}

    out = [path]
    for step_name in steps:
        if verbose:
            print("--- %s" % step_name)

        step_conf = {'debug': debug, 'verbose': verbose}
        try:
            step_conf.update(dict(confp.items(step_name)))
        except configparser.NoSectionError:
            pass

        out = steps[step_name](*out, conf=step_conf)

    t_elapsed = time.time() - time0
    if verbose:
        print("Finished in %.3f seconds" % t_elapsed)

    return out


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python lpr.py <image_path> <config_path>")
        sys.exit()

    print(process_img(sys.argv[1], sys.argv[2]))
