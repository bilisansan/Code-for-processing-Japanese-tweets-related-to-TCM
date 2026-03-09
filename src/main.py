import numpy as np
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """
    Main procedure for topic modeling using BERTopic.
    This script loads preprocessed tweet texts and precomputed
    sentence embeddings, performs dimensionality reduction
    and clustering, and outputs topic information.
    """

    # ------------------------------------------------------------
    # Read the target year from the configuration file
    # ------------------------------------------------------------
    with open("../config/cfg.txt", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("year:"):
                year = line.strip().split(":")[1]

    # ------------------------------------------------------------
    # Load tokenized tweet texts
    # Each line corresponds to a segmented tweet
    # ------------------------------------------------------------
    with open(f'../data/cut_{year}.txt', 'r', encoding='utf-8') as file:
        docs = file.readlines()

    print("Number of documents:", len(docs))

    # ------------------------------------------------------------
    # Load precomputed sentence embeddings
    # ------------------------------------------------------------
    embeddings = np.load(f'../data/emb_sentence_{year}.npy')
    print("Embedding matrix shape:", embeddings.shape)

    # ------------------------------------------------------------
    # Initialize embedding model (used internally by BERTopic)
    # ------------------------------------------------------------
    embedding_model = pipeline(
        "feature-extraction",
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ------------------------------------------------------------
    # Create the UMAP dimensionality reduction model
    # UMAP reduces embedding dimensionality before clustering
    # ------------------------------------------------------------
    umap_model = UMAP(
        n_neighbors=20,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42  # Ensures reproducibility
    )

    # ------------------------------------------------------------
    # Create the HDBSCAN clustering model
    # Adjust min_cluster_size and min_samples to control
    # the detection of noise points (outliers)
    # ------------------------------------------------------------
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=5,
        metric="euclidean"
    )

    # ------------------------------------------------------------
    # Create the CountVectorizer model
    # Used to extract topic keywords
    # ------------------------------------------------------------
    vectorizer_model = CountVectorizer(
        stop_words=["中国", "中華"]
    )

    # ------------------------------------------------------------
    # Initialize the BERTopic model
    # ------------------------------------------------------------
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        # nr_topics="auto"
    )

    # ------------------------------------------------------------
    # Fit the BERTopic model using the precomputed embeddings
    # ------------------------------------------------------------
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Display topic summary information
    topic_model.get_topic_info()

    # ------------------------------------------------------------
    # Optional: automatically reduce the number of topics
    # after clustering
    # ------------------------------------------------------------
    topic_model.reduce_topics(docs, nr_topics="auto")
    topic_model.get_topic_info()

    # ------------------------------------------------------------
    # Retrieve document-topic assignments
    # ------------------------------------------------------------
    topic_docs = topic_model.get_document_info(docs)

    # ------------------------------------------------------------
    # Save clustering results
    # ------------------------------------------------------------
    topic_docs.to_csv(f"../data/cluster_{year}.csv", index=False)


# ------------------------------------------------------------
# Script entry point
# Ensures the script runs only when executed directly
# ------------------------------------------------------------
if __name__ == "__main__":
    main()