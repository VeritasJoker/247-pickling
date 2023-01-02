import pickle
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle

NONWORDS = {"hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"}
min_word_freq = 1
PCA_DIM = 50


def pca_by_svd(X, dim):

    X_mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - X_mean, full_matrices=False)

    return Vt.T[:, :dim], X_mean


def do_pca(embs, pca_eigen, pca_mean):
    return np.dot(embs - pca_mean, pca_eigen)


def process_pca(df):

    assert all([item in df.columns for item in ["adjusted_onset", "adjusted_offset"]])

    df = df[df.embeddings.apply(lambda x: not np.isnan(x).any())]
    k = np.array(df.embeddings.tolist())
    try:
        k = normalize(k, norm=False, axis=1)
    except ValueError:
        df["embeddings"] = k.tolist()

    start_n = len(df)
    common = df.in_glove
    nonword_mask = df.word.str.lower().apply(lambda x: x in NONWORDS)
    freq_mask = df.word_freq_overall >= min_word_freq
    ## filter only common words, in glove and not a non-word
    df = df[common & freq_mask & ~nonword_mask]
    end_n = len(df)
    print(f"Went from {start_n} words to {end_n} words")

    x = np.stack(df.embeddings).astype("float64")
    pca_eigen, pca_mean = pca_by_svd(x, PCA_DIM)
    df.embeddings = df.embeddings.apply(lambda x: do_pca(x, pca_eigen, pca_mean))

    return df


def get_csv_name(filename):

    assert ".pkl" in filename
    filename = f"{filename[:-4]}.csv"

    return filename


def get_new_name(filename, protocol):

    assert ".pkl" in filename
    filename = f"{filename[:-4]}_vv{protocol}.pkl"

    return filename


def save_pickle(item, file_name, protocol_v):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    file_name = get_new_name(file_name, protocol_v)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh, protocol=protocol_v)
    # item.to_pickle(file_name, protocol=protocol_v)
    return


def main():

    PRJCT_ID = "tfs"
    SID = "798"
    path = f"results/{PRJCT_ID}/{SID}/pickles/embeddings/"

    base_df_pattern = glob.glob(os.path.join(path, "*", "*", "*.pkl"))
    # emb_df_pattern = []
    # emb_folder_pattern = glob.glob(os.path.join(path, "*", "*", "cnxt_*"))
    # for emb_folder in emb_folder_pattern:
    #     emb_df = glob.glob(os.path.join(emb_folder, "*.pkl"))
    #     emb_df.sort()
    #     emb_df_pattern.append(emb_df[-1])  # take only last embedding

    for file in base_df_pattern:
        print(file)
        if "gpt2" not in file:
            continue
        emb_folder_pattern = glob.glob(os.path.join(os.path.dirname(file), "cnxt_*"))
        emb_folder_pattern.sort()
        emb_folder = emb_folder_pattern[-1]

        emb_df = glob.glob(os.path.join(emb_folder, "*.pkl"))
        emb_df.sort()
        emb_file = emb_df[-1]

        base_df = load_pickle(file)
        emb_df = load_pickle(emb_file)

        assert len(base_df) == len(emb_df)
        df = pd.concat([base_df, emb_df], axis=1)

        df = process_pca(df)
        filename = get_csv_name(emb_file)
        df.to_csv(filename)

        # for protocol in np.arange(0, 6):
        #     save_pickle(df, file, protocol)

    return


if __name__ == "__main__":
    main()
