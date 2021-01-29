import glob
import os

import mat73
import pandas as pd


def extract_elec_ids(conversation):
    """[summary]

    Args:
        conversation ([type]): [description]

    Returns:
        [type]: [description]
    """
    elec_files = glob.glob(os.path.join(conversation, 'preprocessed', '*.mat'))
    elec_ids_list = sorted(
        list(
            map(lambda x: int(os.path.splitext(x)[0].split('_')[-1]),
                elec_files)))

    return elec_ids_list


def get_common_electrodes(convs):
    """[summary]

    Args:
        convs ([type]): [description]

    Returns:
        [type]: [description]
    """
    all_elec_ids_list = [
        extract_elec_ids(conversation) for conversation in convs
    ]
    all_elec_labels_list = [
        extract_electrode_labels(conversation) for conversation in convs
    ]

    common_electrodes = list(set.intersection(*map(set, all_elec_ids_list)))
    common_labels = sorted(list(
        set.intersection(*map(set, all_elec_labels_list))),
                           key=lambda x: all_elec_labels_list[0].index(x))

    common_labels = [common_labels[elec - 1] for elec in common_electrodes]

    return common_electrodes, common_labels


def get_conversation_list(CONFIG):
    """Returns list of conversations

    Arguments:
        CONFIG {dict} -- Configuration information
        set_str {string} -- string indicating set type (train or valid)

    Returns:
        list -- List of tuples (directory, file, idx, common_electrode_list)
    """

    conversations = sorted(
        glob.glob(os.path.join(CONFIG["CONV_DIRS"], '*conversation*')))

    return conversations


def extract_conversation_contents(conversation, ex_words):
    """Return labels (lines) from conversation text

    Args:
        file ([type]): [description]
        ex_words ([type]): [description]

    Returns:
        list: list of lists with the following contents in that order
                ['word', 'onset', 'offset', 'accuracy', 'speaker']
    """
    df = pd.read_csv(conversation,
                     sep=' ',
                     header=None,
                     names=['word', 'onset', 'offset', 'accuracy', 'speaker'])
    df['word'] = df['word'].str.lower().str.strip()
    df = df[~df['word'].isin(ex_words)]

    return df.values.tolist()


def extract_electrode_labels(conversation_dir):
    """Read the header file electrode labels

    Args:
        conversation_dir (str): conversation folder name/path

    Returns:
        list: electrode labels
    """
    header_file = glob.glob(
        os.path.join(conversation_dir, 'misc', '*_header.mat'))[0]

    if not os.path.exists(header_file):
        return

    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels
