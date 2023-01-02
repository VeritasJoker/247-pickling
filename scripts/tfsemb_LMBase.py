import gensim.downloader as api
import pandas as pd
import tfsemb_download as tfsemb_dwnld
from tfsemb_config import setup_environ
from tfsemb_main import tokenize_and_explode, get_utt_info
from tfsemb_parser import arg_parser
from utils import load_pickle, main_timer
from utils import save_pickle as svpkl


def add_vocab_columns(args, df, column=None):
    """Add columns to the dataframe indicating whether each word is in the
    vocabulary of the language models we're using.
    """

    # Add language models
    for model in [
        *tfsemb_dwnld.CAUSAL_MODELS,
        *tfsemb_dwnld.SEQ2SEQ_MODELS,
        *tfsemb_dwnld.MLM_MODELS,
    ]:
        try:
            tokenizer = tfsemb_dwnld.download_hf_tokenizer(
                model, local_files_only=True
            )
        except:
            tokenizer = tfsemb_dwnld.download_hf_tokenizer(
                model, local_files_only=False
            )

        key = model.split("/")[-1]
        print(f"Adding column: (token) in_{key}")

        try:
            curr_vocab = tokenizer.vocab
        except AttributeError:
            curr_vocab = tokenizer.get_vocab()

        def helper(x):
            try:
                if len(tokenizer.tokenize(x)) == 1:
                    return isinstance(
                        curr_vocab.get(tokenizer.tokenize(x)[0]), int
                    )
            except:
                return False

            return False

        df[f"in_{key}"] = df[column].apply(helper)

    return df


@main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    base_df = load_pickle(args.labels_pickle, "labels")

    glove = api.load("glove-wiki-gigaword-50")
    base_df["in_glove"] = base_df.word.str.lower().apply(
        lambda x: isinstance(glove.key_to_index.get(x), int)
    )

    if args.embedding_type == "glove50":
        base_df = base_df[base_df["in_glove"]]
        base_df = add_vocab_columns(args, base_df, column="word")
    else:
        # Add glove
        base_df = tokenize_and_explode(args, base_df)
        base_df = add_vocab_columns(args, base_df, column="token2word")

    svpkl(base_df, args.base_df_file)


if __name__ == "__main__":
    main()
