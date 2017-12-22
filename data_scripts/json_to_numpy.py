import os.path
import argparse
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import data_utils


"""
Script to read handwriting data samples in json format. X,y position of each stroke and pen-up event are concatenated
to create one stroke step. A handwriting sample consists of sequence of strokes, and it is a numpy array with shape
(# strokes, 3). Each sample has various number of labels:
`char_labels` (# strokes): character label of each stroke of a handwriting sample. Takes values between 0-70 (see alphabet below).
`word_labels` (# strokes): word segmentation, showing which word a stroke belongs to in the sentence (i.e., first, second, etc.)
`eow_labels` (# strokes): end-of-word labels, 1 if stroke corresponds to end-of-word, 0 otherwise.
`bow_labels` (# strokes): beginning-of-word labels, 1 if stroke corresponds to beginning of a word, 0 otherwise.
`eoc_labels` (# strokes): end-of-character labels, 1 if stroke corresponds to end-of-character, 0 otherwise.
`boc_labels` (# strokes): beginning-of-character labels. 1 if stroke corresponds to beginning of a character, 0 otherwise.
`subject_labels` (1): id of subject.
`texts`: (string): text written by the handwriting sample.

`alphabet`: list of letters, numbers, symbols covered by `char_labels`.

Note that the dataset is kept as a dictionary where strokes and labels are kept in separate lists. There is one-to-one
correspondence among those lists.

You can run the following commands to save the data in numpy format:
> python json_to_numpy.py --input_dir <path-to-data>/ethDataSegmented --output_dir <path-to-output-dir> --output_file <eth-output-file> --scale --binarize_pen --block_save
--input_dir /home/eaksan/Warehouse/Datasets/Handwriting/temporal/dataset-full-final/ethDataSegmented --output_dir ../public_data --output_file eth_scaled --scale --binarize_pen --block_save --file_suffix segmented.json
> python json_to_numpy.py --input_dir <path-to-data>/iamondbDataSegmented --output_dir <path-to-output-dir> --output_file <iamondb-output-file> --file_suffix segmented-nonorm.json --scale --binarize_pen --block_save
--input_dir /home/eaksan/Warehouse/Datasets/Handwriting/temporal/dataset-full-final/iamondbDataSegmented --output_dir ../public_data --output_file iamondb_scaled --scale --binarize_pen --block_save --file_suffix segmented-nonorm.json

This should create two .npz files for eth and iamondb datasets.

More explanation for some of command-line arguments:
binarize_pen   : in *segmented.json data file, pen-event is stored by using three values (0,1 and 2). If `binarize_pen`
                 is set to True, pen-event will be in binary format. 1 for pen-up and 0 for pen-down.
rescale_canvas : in *segmented.json data files stroke positions (x,y) are not scaled. If `rescale_canvas` is set True,
                 then samples will be scaled by dividing with width and height of input canvas (not recommended).
"""
#TODO: Json files of ETH data contain missing fields.

# Encodes the alphabet numerically and converts ascii char_labels into integer labels. This function is useful to
# detect non alpha-numerical characters.
alphabet = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,-()/") # %:;&#
alphabet.insert(0, chr(0)) # '\x00' character, i.e., ord(0) to label concatenations.
char_label_encoder = LabelEncoder()
char_label_encoder.fit(alphabet)


def fetch_sample_from_dict(container, json_data, rescale_canvas=False, binarize_pen=False):
    """
    For ETH data.

    Args:
        container (dict): dictionary that collects samples.
        json_data (json): json object to be parsed.
        rescale_canvas (bool): whether to scale stroke x, y coordinates by using canvas width and height.
        binarize_pen (bool): whether to binarize pen event or not. pen event may take 3 different values in json files.

    TODO:
        * Handle words and characters with wrong recognition labels.

    Returns:
        (bool): status. False if the sample isn't inserted into container.
    """
    width_scale, height_scale = 1.0, 1.0
    if rescale_canvas:
        width_scale = float(json_data['image_width'])
        height_scale = float(json_data['image_heigth'])

    json_strokes = json_data['word_stroke']

    text = json_data["word_ascii"]
    strokes = np.zeros((len(json_strokes), 3), dtype=np.float32)
    char_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    word_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    eoc_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    boc_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    eow_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    bow_labels = np.zeros((len(json_strokes)), dtype=np.int32)

    for idx, pt in enumerate(json_strokes):
        x = float(pt['x'])
        y = float(pt['y'])
        ts = float(pt['ts'])
        pen = float(pt['ev'])
        strokes[idx] = [x/width_scale, y/height_scale, pen]

    if binarize_pen:
        strokes[strokes[:, 2] == 1, 2] = 0  # convert 1 into 0 (pen down)
        strokes[strokes[:, 2] == 2, 2] = 1  # convert 2 into 1 (pen up)

    num_mislabeled_words = 0
    num_mislabeled_chars= 0
    for p in range(len(json_data['wholeword_segments'])):
        word = json_data['wholeword_segments'][p]
        word_labels[word['ranges'][0]] = np.ones(len(word['ranges'][0]))*(p+1)
        eow_labels[word['ranges'][0][-1]] = 1
        bow_labels[word['ranges'][0][0]] = 1

        if word['recognition_is_correct'] is not True:
            num_mislabeled_words += 1

        for pc in range(len(word['chars'])):
            char = word['chars'][pc]
            try:
                char_labels[char['ranges'][0]] = char_label_encoder.transform([char['char']])[0]
                eoc_labels[char['ranges'][0][-1]] = 1
                boc_labels[char['ranges'][0][0]] = 1
            except (ValueError, IndexError):
                print(text)
                return False

            if ('recognition_is_correct' in char) and (char['recognition_is_correct'] is not True):
                num_mislabeled_chars += 1

    """
    # TODO Experimental: Align EOC with PenUp event. Pen up event is coupled with label 0. Replace label 0 with latter
    # label.
    c = np.where(char_labels == 0)[0]
    p = np.where(strokes[:,2] == 0)[0]
    common_indices = np.intersect1d(c, p, False)
    char_labels[common_indices] = char_labels[common_indices-1]
    """

    container['samples'].append(strokes)
    container['char_labels'].append(char_labels)
    container['word_labels'].append(word_labels)
    container['texts'].append(text)
    container['eow_labels'].append(eow_labels)
    container['bow_labels'].append(bow_labels)
    container['eoc_labels'].append(eoc_labels)
    container['boc_labels'].append(boc_labels)
    return True

def fetch_sample_from_string(container, json_data, rescale_canvas=False, binarize_pen=False):
    """
    For iamondb data.

    Args:
        container (dict): dictionary that collects samples.
        json_data (json): json object to be parsed.
        rescale_canvas (bool): whether to scale stroke x, y coordinates by using canvas width and height.
        binarize_pen (bool): whether to binarize pen event or not. pen event may take 3 different values in json files.

    TODO:
        * Handle words and characters with wrong recognition labels.

    Returns:
        (bool): status. False if the sample isn't inserted into container.
    """
    width_scale, height_scale = 1, 1
    if rescale_canvas:
        width_scale = float(json_data['image_width'])
        height_scale = float(json_data['image_heigth'])

    json_strokes = json_data['word_stroke']

    text = json_data["word_ascii"]
    strokes = np.zeros((len(json_strokes), 3), dtype=np.float32)
    char_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    word_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    eoc_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    boc_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    eow_labels = np.zeros((len(json_strokes)), dtype=np.int32)
    bow_labels = np.zeros((len(json_strokes)), dtype=np.int32)

    for idx, pt in enumerate(strokes):
        pt_parts = pt.split(",")
        if len(pt_parts) > 1 :
            x = float(pt_parts[0].split("=")[1])
            y = float(pt_parts[1].split("=")[1])
            ts = float(pt_parts[2].split("=")[1])
            pen = float(pt_parts[4].split("=")[1])
            strokes[idx] = [x/width_scale, y/height_scale, pen]

    if binarize_pen:
        strokes[strokes[:, 2] == 1, 2] = 0  # convert 1 into 0 (pen down)
        strokes[strokes[:, 2] == 2, 2] = 1  # convert 2 into 1 (pen up)

    num_mislabeled_words = 0
    num_mislabeled_chars = 0
    for p in range(len(json_data['wholeword_segments'])):
        word = json_data['wholeword_segments'][p]
        word_labels[word['ranges'][0]] = np.ones(len(word['ranges'][0]))*(p+1)
        eow_labels[word['ranges'][0][-1]] = 1
        bow_labels[word['ranges'][0][0]] = 1

        if word['recognition_is_correct'] is not True:
            num_mislabeled_words += 1

        for pc in range(len(word['chars'])):
            char = word['chars'][pc]
            try:
                char_labels[char['ranges'][0]] = char_label_encoder.transform([char['char']])[0]
                eoc_labels[char['ranges'][0][-1]] = 1
                boc_labels[char['ranges'][0][0]] = 1
            except ValueError:
                print(text)
                return False

            if ('recognition_is_correct' in char) and (char['recognition_is_correct'] is not True):
                num_mislabeled_chars += 1

    container['samples'].append(strokes)
    container['char_labels'].append(char_labels)
    container['word_labels'].append(word_labels)
    container['texts'].append(text)
    container['eow_labels'].append(eow_labels)
    container['bow_labels'].append(bow_labels)
    container['eoc_labels'].append(eoc_labels)
    container['boc_labels'].append(boc_labels)
    return True

def parse_json_file(json_file, rescale_canvas, binarize_pen):
    """
    Args:
        json_file:
        rescale_canvas:
        binarize_pen:

    Returns:

    """
    data_dict = {}
    data_dict['samples'] = [] # data matrix with shape (# strokes, 3) where the second dimension stands for x, y and pen.
    data_dict['char_labels'] = []
    data_dict['word_labels'] = []
    data_dict['subject_labels'] = []
    data_dict['texts'] = []
    data_dict['eow_labels'] = []
    data_dict['bow_labels'] = []
    data_dict['eoc_labels'] = []
    data_dict['boc_labels'] = []

    data_dict['alphabet'] = alphabet

    ids = fetch_ids(json_file)
    num_error_samples = 0

    with open(json_file) as data_file:
        json_data = json.load(data_file)
        for sample_key, sample_json in json_data.items():

            if 'word_stroke' in sample_json and 'wholeword_segments' in sample_json:
                if isinstance(sample_json['word_stroke'], str):
                    success = fetch_sample_from_string(data_dict, sample_json, rescale_canvas, binarize_pen)
                else:
                    success = fetch_sample_from_dict(data_dict, sample_json, rescale_canvas, binarize_pen)

                if success:
                    data_dict['subject_labels'].append(ids['subjectID'])
            else:
                num_error_samples += 1

    return data_dict


def fetch_ids(input_path):
    """

    Args:
        input_path:

    Returns:
        (dict): 'subjectID' and 'formID'
    """
    parts = input_path.split('/')
    ids = {}
    ids['subjectID'] = int(parts[-3])
    ids['formID'] =  parts[-2]
    return ids


def input_to_output_path(input_path):
    """
    args.output_dir/SubjectID-FormID

    Args:
        input_path:
    Returns:
        (str): output file path.
    """
    # TODO: *Assuming input_path isn't relative or contains subject and form information.
    parts = input_path.split('/')
    return parts[-3] + "-" + parts[-2] # SubjectID-FormID


def scale_zero_one(data_dict, threshold=None):
    """
    Scales the (x,y) stroke positions between 0 and 1 by calculating global maximum and minimum values in the whole
    dataset. Note that pen dimension (i.e., 2) is ignored during calculations. Data statistics are inserted into dataset
    dictionary.

    Args:
        data_dict (dict):
        threshold (scalar): clamps noisy values.

    Returns:
        (dict): Scaled data,
    """
    all = np.vstack(data_dict['samples'])
    training_min = all.min(axis=0)
    training_max = all.max(axis=0)

    for idx, sample in enumerate(data_dict['samples']):
        data_dict['samples'][idx][:, [0, 1]] = ((sample-training_min)/(training_max-training_min))[:, [0, 1]]
    data_dict['min'] = training_min
    data_dict['max'] = training_max
    if not('preprocessing' in data_dict):
        data_dict['preprocessing'] = []
    data_dict['preprocessing'].append('scale')

    return data_dict

def main(args):

    data_dict = {}
    data_dict['samples'] = []  # data matrix with shape (# strokes, 3) where the second dimension stands for x, y and pen.
    data_dict['char_labels'] = []
    data_dict['word_labels'] = []
    data_dict['subject_labels'] = []
    data_dict['texts'] = []
    data_dict['eow_labels'] = []
    data_dict['bow_labels'] = []
    data_dict['eoc_labels'] = []
    data_dict['boc_labels'] = []
    data_dict['alphabet'] = alphabet

    # Single file.
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise ValueError('Input file ' + args.input_file + ' does not exists.')

        single_file_data_dict = parse_json_file(args.input_file, args.rescale_canvas, args.binarize_pen)
        data_dict = data_utils.dictionary_merge([data_dict, single_file_data_dict], inplace_idx=0, keys_frozen=['alphabet'], verbose=0)

        if args.scale_zero_one:
            data_dict = scale_zero_one(data_dict)

        output_path = os.path.join(args.output_dir, input_to_output_path(args.input_file))
        np.savez_compressed(output_path, **data_dict)

        return data_dict

    # Directory: multiple files.
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            raise ValueError('Input folder ' + args.input_dir + ' does not exists.')

        json_file_list = [os.path.join(r, fn)
                        for r, ds, fs in os.walk(args.input_dir)
                        for fn in fs if fn.endswith(args.file_suffix)]
        if len(json_file_list) == 0:
            print('No file with extension <' + args.file_suffix + '> is found.')
            exit()


        for idx, json_file in enumerate(json_file_list):
            if (idx % 100) == 0:
                print("{}/{}".format(idx, len(json_file_list)))

            single_file_data_dict = parse_json_file(json_file, args.rescale_canvas, args.binarize_pen)
            if not args.block_save:
                if args.scale_zero_one:
                    data_dict = scale_zero_one(data_dict)
                output_path = os.path.join(args.output_dir, input_to_output_path(json_file))
                np.savez_compressed(output_path, **data_dict)
                return single_file_data_dict
            else:
                data_dict = data_utils.dictionary_merge([data_dict, single_file_data_dict], inplace_idx=0, keys_frozen=['alphabet'], verbose=0)

        if args.block_save:
            if args.scale_zero_one:
                data_dict = scale_zero_one(data_dict)
            output_path = os.path.join(args.output_dir, args.output_file)
            np.savez_compressed(output_path, **data_dict)

            return data_dict
    else:
        raise ValueError("Either --input_file or --input_dir must be provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument('-input_file', '--input_file', action="store", dest="input_file",
                        required=False, help='Single json file')
    parser.add_argument('-input_dir', '--input_dir', action="store", dest="input_dir",
                        required=False, help='Dataset folder containing multiple files.')
    parser.add_argument('-out_dir', '--output_dir', action="store", dest="output_dir",
                        required=True, help='Output directory.')
    parser.add_argument('-out_file', '--output_file', action="store", dest="output_file", default="data-all",
                        required=False, help='Output file.')
    parser.add_argument('-block', '--block_save', action="store_true", dest="block_save", default=False,
                        required=False, help='Save multiple json files into a single numpy file.')
    parser.add_argument('-suffix', '--file_suffix', action="store", dest="file_suffix", default="segmented.json",
                        required=False, help='Suffix of data files to be read from disk.')
    parser.add_argument('-rescale_canvas', '--rescale_canvas', action="store_true", dest="rescale_canvas",
                        required=False, help='Scale x,y by using canvas width and height.')
    parser.add_argument('-scale', '--scale_zero_one', action="store_true", dest="scale_zero_one",
                        required=False, help='Scale x,y by using global maximum and minimum values in the whole dataset.')
    parser.add_argument('-binarize_pen', '--binarize_pen', action="store_true", dest="binarize_pen",
                        required=False, help='Convert pen information into binary 0 and 1 format.')
    args = parser.parse_args()
    main(args)
