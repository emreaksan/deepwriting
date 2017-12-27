import numpy as np

def label_end_of_sub_sequences(char_list, tolerate_zero=True):
    """
    Given a sequence of labels locates end-of-sub-sequences and assigns 1.

    For example:
    [0,0,10,10,10,10,10,10,0,10,10,12,12,12] to [0,0,0,0,0,0,0,0,0,0,1,0,0,1]

    Args:
        char_list:
        tolerate_zero: fills gaps of length one between two same labels.

    Returns:

    """
    prev_char = char_list[0]
    eoc_labels = np.zeros(len(char_list))
    for idx in range(len(char_list)):
        if char_list[idx] == 0 and tolerate_zero:
            if idx-1 >= 0 and idx+1 < len(char_list) and (char_list[idx-1] == char_list[idx+1]):
                char_list[idx] = char_list[idx+1]

        if (prev_char != char_list[idx]) and prev_char != 0:
            eoc_labels[idx-1] = 1
            #eoc_labels[idx] = 1 # Aligns eoc with pen event.

        prev_char = char_list[idx]

    if char_list[-1] != 0:
        eoc_labels[-1] = 1
    #eoc_labels[-1] = 1

    return eoc_labels


def smooth_int_labels(int_labels, horizon=3):
    """
    Given a sequence of integer labels, corrects individual differences by comparing a label at t with labels in
    t+horizon.

    For example:
    [0,5,10,10,10,10,10,0,3,12,12,12,12] to [0,10,10,10,10,10,10,0,12,12,12,12,12]

    Args:
        int_labels:
        horizon:

    Returns:

    """
    out = int_labels.copy()
    for idx in range(1, len(int_labels)-horizon):
        if (int_labels[idx] != 0) and not(int_labels[idx] in int_labels[idx+1:idx+4]) and not(int_labels[idx] == int_labels[idx-1]):
            out[idx] = int_labels[idx+1]

    return out

def simplify_int_labels(int_labels, threshold=5):
    """
    Given a sequence of integer labels, finds out unique labels in repetitions.

    For example:
    [0,10,10,10,10,10,10,0,12,12,12,12,12] to [10,12]

    Args:
        int_labels:
        threshold: number of consecutive occurrences before selecting a label.

    Returns:

    """
    out = []
    num_occur = 0
    for idx in range(len(int_labels)-1):
        if (int_labels[idx] == int_labels[idx+1]) and (int_labels[idx] != 0):
            num_occur += 1
        elif (int_labels[idx] != int_labels[idx+1]) and (num_occur >= threshold-1):
            out.append(int_labels[idx])
            num_occur = 0
        else:
            num_occur = 0

    return out

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def get_text(int_labels, eoc_labels, bow_labels, label_encoder, in_raw_threshold=2, eoc_threshold=0.5):
    """

    Args:
        int_labels:
        eoc:

    Returns:

    """
    eoc_indices = np.where((eoc_labels > eoc_threshold) == 1)[0]
    min_char_distance = np.diff(eoc_indices).min()

    bow_indices = np.where((bow_labels > eoc_threshold) == 1)[0].tolist()
    bow_indices.append(len(eoc_labels))

    chars = []
    char_int_labels = []
    indices = []
    next_space_index = 0
    num_occur = 0
    for idx in range(len(int_labels)-1):
        if (int_labels[idx] == int_labels[idx+1]) and (int_labels[idx] != 0):
            num_occur += 1
        elif (int_labels[idx] != int_labels[idx+1]) and (num_occur >= in_raw_threshold-1):
            num_occur = 0
            # Check for pen up event of a character to prevent duplicate entries.
            eoc_idx = find_nearest(eoc_indices, idx)
            #if (idx in eoc_indices):
            if abs(eoc_idx - idx) < min_char_distance/3.0:
                if idx > bow_indices[next_space_index]:
                    chars.append(" ")
                    next_space_index += 1

                char_int_labels.append(int_labels[idx])
                chars.append(label_encoder([int_labels[idx]])[0])
                indices.append(idx)

        else:
            num_occur = 0

    text = "".join(chars)

    return text, char_int_labels, indices


