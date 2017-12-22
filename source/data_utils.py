import numpy as np
import copy

def dictionary_split(dictionary, split_indices, keys_frozen=[], verbose=1):
    """
    Splits the data dictionary of lists into smaller chunks. All (key,value) pairs must have the same number of
    elements. If there is an index error, then the corresponding (key, value) pair is copied directly to the new
    dictionaries.

    Args:
        dictionary (dict): data dictionary.
        split_indices (list): Each element contains a list of indices for one split. Multiple splits are supported.
        keys_frozen (list): list of keys that are to be copied directly. The remaining (key,value) pairs will be
            used in splitting. If not provided, all keys will be used.
        verbose (int): status messages.

    Returns:
        (tuple): a tuple containing chunks of new dictionaries.
    """
    # Find <key, value> pairs having an entry per sample.
    sample_level_keys = []
    dataset_level_keys = []

    num_samples = sum([len(l) for l in split_indices])
    for key, value in dictionary.items():
        if not(key in keys_frozen) and ((isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples)):
            sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            print(str(key) + " is copied.")

    chunks = []
    for chunk_indices in split_indices:
        dict = {}

        for key in dataset_level_keys:
            dict[key] = dictionary[key]
        for key in sample_level_keys:
            dict[key] = [dictionary[key][i] for i in chunk_indices]
        chunks.append(dict)

    return tuple(chunks)

def dictionary_merge(dictionary_list, inplace_idx=-1, keys_frozen=[], verbose=1):
    """
    Merges given dictionaries by merging entries of list type and copies the remaining entries from the first dictionary
    in dictionary_list.

    Args:
        dictionary_list (list): list of data dictionaries.
        inplace_idx (int): makes the merge operations inplace on the dictionary item specified.
        keys_frozen (list): list of keys that are to be copied directly. The remaining (key,value) pairs will be
            used in splitting. If not provided, all keys will be used.
        verbose (int): status messages.

    Returns:
        (dict): final dictionary.

    """
    if inplace_idx > -1 and inplace_idx < len(dictionary_list):
        merged_dict = dictionary_list[inplace_idx]
    else:
        merged_dict = copy.deepcopy(dictionary_list[0])

    for key, value in dictionary_list[0].items():
        if not(key in keys_frozen) and (isinstance(value, np.ndarray) or isinstance(value, list)):
            for d_idx in range(1, len(dictionary_list)):
                if isinstance(merged_dict[key], list):
                    merged_dict[key].extend(dictionary_list[d_idx][key])
                elif isinstance(merged_dict[key], np.ndarray):
                    merged_dict[key] = np.concatenate((merged_dict[key], dictionary_list[d_idx][key]), axis=0)
                else:
                    raise Exception("Unidentified type.")

            if verbose > 0:
                print(str(key)+" is merged.")
        else:
            merged_dict[key] = value
            if verbose > 0:
                print(str(key)+" is copied.")

    return merged_dict

def npz_to_dict(npz_data):
    """
    Converts numpy compressed object into dictionary.

    Args:
        npz_obj (NpzFile):

    Returns:
        (dict)

    """
    out_dict = {}
    for key in npz_data.iterkeys():
        # Discard npz object key.
        if key != 'arr_0':
            if isinstance(npz_data[key], np.ndarray):
                out_dict[key] = npz_data[key].tolist()
            else:
                out_dict[key] = npz_data[key]

    return out_dict