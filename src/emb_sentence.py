from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def main():
    """
    Main procedure for generating sentence embeddings for cleaned tweet texts.
    The target year is read from a configuration file, and embeddings are
    generated using a multilingual Sentence-BERT model (paraphrase-multilingual-MiniLM-L12-v2).
    """

    # ------------------------------------------------------------
    # Read the target year from the configuration file
    # The configuration file contains a line in the format: year:XXXX
    # ------------------------------------------------------------
    with open("../config/cfg.txt", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("year:"):
                year = line.strip().split(":")[1]

    # ------------------------------------------------------------
    # Load the cleaned tweet text data for the specified year
    # Each line corresponds to a preprocessed tweet
    # ------------------------------------------------------------
    with open(f'../data/clean_{year}.txt', 'r', encoding='utf-8') as file:
        docs = file.readlines()

    # ------------------------------------------------------------
    # Initialize the multilingual Sentence-BERT model
    # This model converts text into dense semantic embeddings
    # ------------------------------------------------------------
    emb_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ------------------------------------------------------------
    # Generate sentence embeddings for all tweets
    # show_progress_bar=True provides a progress indicator
    # for large-scale datasets
    # ------------------------------------------------------------
    embs = emb_model.encode(docs, show_progress_bar=True)

    # ------------------------------------------------------------
    # Save the embedding matrix as a NumPy binary file (.npy)
    # This format allows efficient storage and fast loading
    # for downstream tasks such as clustering or topic modeling
    # ------------------------------------------------------------
    save_file = f'../data/emb_sentence_{year}.npy'
    np.save(save_file, embs)


# ------------------------------------------------------------
# Script entry point
# Ensures that the main function runs only when the script is
# executed directly, not when imported as a module
# ------------------------------------------------------------
if __name__ == "__main__":
    main()