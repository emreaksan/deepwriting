import numpy as np
import data_utils
import utils_hw
import os
import argparse

"""
Preprocessing functions. Note that all functions take a *dataset* dictionary, which is prepared by json_to_numpy.py 
script and return the processed data in the same format.

You can run the following command to apply pre-processing:
-data_file <eth-output-file> <iamondb-output-file> -out_dir <path-to-output-dir> -out_file <output-file> -merge_first -translate -diff -standardize -semantic_chunks 300 -validation 0.01

Loads eth and iamondb datasets in dictionary format, merges them and applies: 
(1) [semantic_chunks] splits long sequences into shorter sequences by using character segmentation,
(2) [translate] subtracts offset so that all samples start from (0,0) coordinates,
(3) [diff] calculates delta (x,y) between consecutive steps,
(4) [validation] creates a validation split,
(5) [standardize] zero-mean, unit-variance normalization.
"""

def translate_to_origin(dataset):
    """
    Subtracts x- and y-offset such that the first stroke of every sample is (0,0).

    Args:
        dataset (dict): data dictionary.

    Returns:
        (dict): after process.
    """
    for sample in dataset['samples']:
        x_offset = sample[0,0]
        y_offset = sample[0,1]
        sample[:,0] -= x_offset
        sample[:,1] -= y_offset

    dataset['preprocessing'].append('origin_translation')

    return dataset

def convert_to_diff_representation(dataset):
    """
    Converts the stroke data into relative representation by calculating difference between consecutive steps.

    Args:
        dataset (dict): data dictionary.

    Returns:
        (dict): after process.

    """
    for idx, sample in enumerate(dataset['samples']):
        source = np.vstack((sample[0], sample))
        diff = np.diff(source, axis=0)
        diff[:,2] = sample[:,2] # Keep original pen information since it is already relative.
        dataset['samples'][idx] = diff

    dataset['preprocessing'].append('relative_representation')

    return dataset

def scale_dataset(training_data, validation_data, threshold=None):
    """
    Scales the (x,y) stroke positions between 0 and 1 by calculating global maximum and minimum values in the data.
    Note that pen dimension (i.e., 2) is ignored during calculations. Data statistics are inserted into dataset
    dictionaries.

    Args:
        training_data (dict):
        validation_data (dict) or (list): is validation dataset or list of multiple datasets.
        threshold (scalar): clamps noisy samples.

    Returns:
        (tuple): A tuple containing the followings:
            Normalized training_data,
            Normalized validation_data,
    """
    validation_data_list = validation_data
    all = np.vstack(training_data['samples'])
    training_min = all.min(axis=0)
    training_max = all.max(axis=0)

    for idx, sample in enumerate(training_data['samples']):
        training_data['samples'][idx][:, [0, 1]] = ((sample-training_min)/(training_max-training_min))[:, [0, 1]]
    training_data['min'] = training_min
    training_data['max'] = training_max
    training_data['preprocessing'].append('scale')

    if validation_data is not None:
        if type(validation_data_list) != type([]):
            validation_data_list = [validation_data]

        for v_data in validation_data_list:
            for idx, sample in enumerate(v_data['samples']):
                v_data['samples'][idx][:, [0, 1]] = ((sample-training_min)/(training_max-training_min))[:, [0, 1]]
                v_data['min'] = training_min
                v_data['max'] = training_max
            v_data['preprocessing'].append('scale')

        if type(validation_data) != type([]):
            validation_data_list = validation_data_list[0]

    return training_data, validation_data_list

def standardize_dataset(training_data, validation_data=None):
    """
    Calculates training data statistics (mean, std) and applies normalization. Note that pen dimension (the second dim.)
    is ignored during calculations. Data statistics are inserted into dataset dictionaries.

    Args:
        training_data (dict):
        validation_data (dict) or (list): is validation dataset or list of multiple datasets.

    Returns:
        (tuple): A tuple containing the followings:
            Normalized training_data,
            Normalized validation_data,
    """
    validation_data_list = validation_data
    all = np.vstack(training_data['samples'])
    training_mean = all.mean(axis=0)
    training_std = all.std(axis=0)
    training_std[np.where(training_std < 1e-10)] = 1.0 # suppress values where std = 0.0

    for idx, sample in enumerate(training_data['samples']):
        training_data['samples'][idx][:,[0,1]] = ((sample - training_mean) / training_std)[:,[0,1]]
    training_data['mean'] = training_mean
    training_data['std'] = training_std
    training_data['preprocessing'].append('normalization')

    if validation_data is not None:
        if type(validation_data_list) != type([]):
            validation_data_list = [validation_data]

        for v_data in validation_data_list:
            for idx, sample in enumerate(v_data ['samples']):
                v_data['samples'][idx][:,[0,1]] = ((sample - training_mean) / training_std)[:,[0,1]]
                v_data['mean'] = training_mean
                v_data['std'] = training_std

            v_data['preprocessing'].append('normalization')

        if type(validation_data) != type([]):
            validation_data_list = validation_data_list[0]

    return training_data, validation_data_list


def validation_split(dataset, amount_validation=1):
    """
    Creates a validation split by using the given `amount_validation` samples.
    Args:
        dataset (dict): data dictionary.
        amount_validation: amount of validation data. If between 0 and 1, then it is regarded as ratio. Otherwise,
            number of validation samples.

    Returns:
        (tuple): training and validation splits.
    """
    num_samples = len(dataset['samples'])
    indices = np.random.permutation(num_samples)
    if amount_validation <= 1 and amount_validation > 0:
        num_validation_samples = int(num_samples*amount_validation)
    else:
        assert(amount_validation > num_samples)
        num_validation_samples = amount_validation
    validation_indices = indices[:num_validation_samples]
    training_indices = indices[num_validation_samples:]

    return data_utils.dictionary_split(dataset, [training_indices, validation_indices])


def split_into_fixed_length_chunks(dataset, fixed_length, min_length=32, keep_residuals=False):
    """
    Splits the sequences into shorter sequences with the same length.
    
    Args:
        dataset (dict):
        length: 
        keep_residuals: 

    Returns:

    """
    new_data = {}

    # Detect <key, value> pairs having an entry per sample or per sequence.
    sequence_level_keys = [] # There is a list per sample, corresponding sequence labeling.
    sample_level_keys = [] # There is an entry per sample.
    dataset_level_keys = [] # Normalization statistics, etc.

    num_samples = len(dataset['samples'])
    random_sample_idx = 1
    for key, value in dataset.items():
        if (isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples):
            new_data[key] = []
            # Check length of sequence of random sample.
            random_sample = value[random_sample_idx]
            if (isinstance(random_sample, np.ndarray) or isinstance(random_sample, list)) and (len(value[random_sample_idx]) == len(dataset['samples'][random_sample_idx])):
                sequence_level_keys.append(key)
            else:
                sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            new_data[key] = value

    # Split each sample (sequence) into shorter chunks.
    for idx, sample in enumerate(dataset['samples']):
        if idx % 1000 == 0:
            print(str(idx) + "/" + str(len(dataset['samples'])))

        sample_len = sample.shape[0]
        num_chunks = int(sample_len/fixed_length)
        len_residual = sample_len%fixed_length

        # Split the sequence into fixed-length chunks for each sequence-level entry.
        for key in sequence_level_keys:
            chunks = []
            chunk_residual = []
            if num_chunks > 0:
                chunks = np.split(dataset[key][idx], np.cumsum([fixed_length]*num_chunks), axis=0)
                if len_residual >= min_length:
                    chunk_residual = chunks[-1]
                chunks = chunks[:-1]
            elif len_residual >= min_length:
                chunk_residual = dataset[key][idx]

            new_data[key].extend(chunks)
            if keep_residuals is True:
                new_data[key].append(chunk_residual)

        # Make copy of sample-level data for each new chunk.
        for key in sample_level_keys:
            new_data[key].extend([dataset[key][idx]]*num_chunks)
            if keep_residuals is True:
                new_data[key].append(dataset[key][idx])

    new_data['preprocessing'].append('fixed_length_chunks')

    assert len(new_data[sequence_level_keys[0]]) == len(new_data[sample_level_keys[0]]), "# of samples in new split doesn't match."
    if len(sequence_level_keys) > 1:
        assert len(new_data[sequence_level_keys[0]][0]) == len(new_data[sequence_level_keys[1]][0]), "# of samples in new split doesn't match."
    return new_data


def split_into_semantic_chunks(dataset, semantic_label_key='char_labels', max_length=400):
    """
    Splits sequences into chunks of max_length by using word/char information. A word is inserted into a chunk if chunk
    length doesn't exceed max_length. Otherwise, a new chunk is created.

    Quality check: if the sequences shorter than 32 steps or longer than 2*max_length or with total number of zero
    labels larger than 1/4 of the sequence length, then they are discarded.

    Args:
        dataset (dict):
        semantic_label_key (str): segmentation label key in the dataset, defining position of a split.
        max_length:

    Returns:

    """
    new_data = {}

    # Detect <key, value> pairs having an entry per sample or per sequence.
    sequence_level_keys = []  # There is a list per sample, corresponding sequence labeling.
    sample_level_keys = []  # There is an entry per sample.
    dataset_level_keys = []  # Normalization statistics, etc.

    num_samples = len(dataset['samples'])
    random_sample_idx = 1
    for key, value in dataset.items():
        if (isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples):
            new_data[key] = []
            # Check length of sequence of random sample.
            random_sample = value[random_sample_idx]
            if (isinstance(random_sample, np.ndarray) or isinstance(random_sample, list)) and (
                len(value[random_sample_idx]) == len(dataset['samples'][random_sample_idx])):
                sequence_level_keys.append(key)
            else:
                sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            new_data[key] = value

    for idx, sample in enumerate(dataset['samples']):
        if idx%1000 == 0:
            print(str(idx)+"/"+str(len(dataset['samples'])))

        split_labels = utils_hw.label_end_of_sub_sequences(dataset[semantic_label_key][idx], tolerate_zero=True)

        split_positions = np.where(split_labels == 1)[0]+1
        split_positions = np.insert(split_positions, 0, 0)
        # split_positions[-1] = len(split_labels)

        split_indices = []
        chunk_len = 0
        for start, end in zip(split_positions[:-1], split_positions[1:]):
            if chunk_len > max_length:
                split_indices.append(start)  # Add previous word.
                chunk_len = end-start
            else:
                chunk_len += end-start

        chunk_dict = {}
        for key in sequence_level_keys:
            chunk_dict[key] = np.split(dataset[key][idx], split_indices, axis=0)

        # Quality check
        total_chunk_size = 0
        for i, char_labels in enumerate(chunk_dict['char_labels']):
            total_chunk_size += char_labels.shape[0]
            if not (((char_labels == 0).sum() > len(char_labels)/4) or (char_labels.shape[0] < 32) or (char_labels.shape[0] > max_length*2)):

                for key in sequence_level_keys:
                    new_data[key].append(chunk_dict[key][i])
                for key in sample_level_keys:
                    new_data[key].append(dataset[key][idx])

    new_data['preprocessing'].append('semantic_chunks')

    assert len(new_data[sequence_level_keys[0]]) == len(new_data[sample_level_keys[0]]), "# of samples in new split doesn't match."
    if len(sequence_level_keys) > 1:
        assert len(new_data[sequence_level_keys[0]][0]) == len(new_data[sequence_level_keys[1]][0]), "# of samples in new split doesn't match."
    return new_data

def extract_eoc_labels(dataset):
    """
    Creates a label showing end of a character in a sequence.
    Args:
        dataset:

    Returns:

    """
    dataset['eoc_labels'] = []
    for idx, char_labels in enumerate(dataset['char_labels']):
        eoc_label = utils_hw.label_end_of_sub_sequences(char_labels)
        eoc_label = np.expand_dims(np.float32(eoc_label), axis=1) # Assuming the last stroke is always end-of-char
        dataset['eoc_labels'].append(eoc_label)

    return dataset


def process_dataset(args, dataset, out_file):
    if not('preprocessing' in dataset):
        dataset['preprocessing'] = []
    if isinstance(dataset['preprocessing'], np.ndarray):
        dataset['preprocessing'] = dataset['preprocessing'].tolist()

    if args.fixed_length_chunks is not None:
        if args.fixed_length_chunks[-1] == 'r':
            dataset = split_into_fixed_length_chunks(dataset, int(args.fixed_length_chunks[:-1]), keep_residuals=True)
        else:
            dataset = split_into_fixed_length_chunks(dataset, int(args.fixed_length_chunks), keep_residuals=False)
    elif args.semantic_chunks_max_len > 0:
        dataset = split_into_semantic_chunks(dataset, max_length=args.semantic_chunks_max_len)

    if args.translate_to_origin:
        dataset = translate_to_origin(dataset)

    if args.relative_representation:
        dataset = convert_to_diff_representation(dataset)

    if args.eoc_labels:
        dataset = extract_eoc_labels(dataset)

    if args.amount_validation_samples > 0:
        training_dataset, validation_dataset = validation_split(dataset, args.amount_validation_samples)
    else:
        training_dataset = dataset
        validation_dataset = None

    if args.standardize_data:
        training_dataset, validation_dataset = standardize_dataset(training_dataset, validation_dataset)

    if args.scale_data_zero_one:
        training_dataset, validation_data = scale_dataset(training_dataset, validation_dataset)

    if args.amount_validation_samples is not None:
        training_path = os.path.join(args.out_dir, out_file+"_training")
        validation_path = os.path.join(args.out_dir, out_file+"_validation")

        np.savez_compressed(training_path, **training_dataset)
        print("# training samples: "+str(len(training_dataset['samples'])))

        if validation_dataset:
            np.savez_compressed(validation_path, **validation_dataset)
            print("# validation samples: "+str(len(validation_dataset['samples'])))
    else:
        training_path = os.path.join(args.out_dir, out_file)
        np.savez_compressed(training_path, **training_dataset)
        print("# samples: "+str(len(training_dataset['samples'])))

if __name__ == "__main__":
    """    
-data_file
/home/eaksan/Warehouse/Projects/rlvnn/handwriting/public_data/eth_scaled.npz
/home/eaksan/Warehouse/Projects/rlvnn/handwriting/public_data/iamondb_scaled.npz
-out_dir
/home/eaksan/Warehouse/Projects/rlvnn/handwriting/public_data/
-out_file
data_preprocessed_semantic_300
-translate
-diff
-standardize
-merge_first
-semantic_chunks
300
--amount_validation_samples
0.01
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument('-merge_first', '--merge_input_dictionaries_first', action="store_true", dest="merge_input_dictionaries_first",
                        required=False, help='If -data_file addresses multiple data dictionary, then merge them first.')
    parser.add_argument('-data_file', '--data_npz_file', nargs='+', action="store", dest="data_file",
                        required=True, help='Data file saved in npz format.')
    parser.add_argument('-out_dir', '--output_directory', action="store", dest="out_dir",
                        required=True, help='Output directory to save in.')
    parser.add_argument('-out_file', '--output_file', nargs='+', action="store", dest="out_file",
                        required=False, help='Output file prefix to save in.')
    parser.add_argument('-translate', '--translate_to_origin', action="store_true", dest="translate_to_origin",
                        required=False, help='Translate to origin.')
    parser.add_argument('-diff', '--relative_representation', action="store_true", dest="relative_representation",
                        required=False, help='Relative representation.')
    parser.add_argument('-standardize', '--standardize', action="store_true", dest="standardize_data",
                        required=False, help='Standardize dataset.')
    parser.add_argument('-scale', '--scale_zero_one', action="store_true", dest="scale_data_zero_one",
                        required=False, help='Scale dataset between 0 and 1.')
    parser.add_argument('-validation', '--amount_validation_samples', action="store", dest="amount_validation_samples",
                        required=False, type=float, default=-1, help='Validation ratio in [0,1) or number of validation samples.')
    parser.add_argument('-fixed_length_chunks', '--fixed_length_chunks', action="store", dest="fixed_length_chunks",
                        required=False, type=str, default=None, help='Split sequence into fixed length chunks. A sequence length followed by "r" (<len>r) keeps residuals.')
    parser.add_argument('-semantic_chunks', '--semantic_chunks', action="store", dest="semantic_chunks_max_len",
                        required=False, type=int, default=0, help='Split sequence into chunks of words/characters with maximum length.')
    parser.add_argument('-eoc_labels', '--eoc_labels', action="store_true", dest="eoc_labels",
                        required=False, help='Calculate end-of-character labels and store.')
    args = parser.parse_args()

    # If output file name is not passed, then create a name from input file name.
    out_file_names = args.out_file
    if out_file_names is None or len(out_file_names) == 1:
        out_file_names = [out_file_names]*len(args.data_file)

    dataset_list = []
    for data_file, out_file in zip(args.data_file, out_file_names):
        dataset = dict(np.load(data_file))
        #dataset = data_utils.npz_to_dict(dataset)

        if out_file is None:
            out_file = data_file.split('/')[-1].split('.')[0]

        if args.merge_input_dictionaries_first is False:
            process_dataset(args, dataset, out_file)
        else:
            dataset_list.append(dataset)

    if args.merge_input_dictionaries_first:
        dataset = data_utils.dictionary_merge(dataset_list, keys_frozen=['alphabet', 'min', 'max', 'mean', 'std'])
        print(dataset.keys())
        process_dataset(args, dataset, out_file_names[0])


