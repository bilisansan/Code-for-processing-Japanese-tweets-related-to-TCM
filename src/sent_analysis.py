import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def main():
    """
    Main procedure for sentiment analysis of tweet texts.
    The script loads cleaned tweets, applies a multilingual
    sentiment analysis model, and outputs both sentiment labels
    and confidence scores.
    """

    # ------------------------------------------------------------
    # Read the target year from the configuration file
    # ------------------------------------------------------------
    with open("../config/cfg.txt", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("year:"):
                year = line.strip().split(":")[1]

    # ------------------------------------------------------------
    # Load pretrained multilingual sentiment analysis model
    # ------------------------------------------------------------
    model_name = "tabularisai/multilingual-sentiment-analysis"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # ------------------------------------------------------------
    # Load cleaned tweet texts
    # ------------------------------------------------------------
    with open(f"../data/clean_{year}.txt", "r", encoding="utf-8") as file:
        tw_text = file.readlines()

    sent = []
    score = []

    # ------------------------------------------------------------
    # Perform sentiment prediction for each tweet
    # ------------------------------------------------------------
    for text in tqdm(tw_text):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Sentiment label mapping
        sentiment_map = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive"
        }

        idx = torch.argmax(probabilities, dim=-1).item()
        prob = probabilities[0][idx].item()

        sent.append(sentiment_map[idx])
        score.append(prob)

    # ------------------------------------------------------------
    # Save sentiment labels to a text file
    # ------------------------------------------------------------
    with open(f"../data/sentiment_{year}.txt", "w", encoding="utf-8") as f:
        for line in sent:
            f.write(f"{line}\n")

    # ------------------------------------------------------------
    # Save sentiment results with probability scores to CSV
    # ------------------------------------------------------------
    df = pd.DataFrame({
        "sentiment": sent,
        "score": score
    })

    # Format probability scores to four decimal places
    df["score"] = df["score"].map(lambda x: f"{x:.4f}")

    df.to_csv(
        f"../data/sentiment_{year}.csv",
        index=False,
        encoding="utf-8"
    )


# ------------------------------------------------------------
# Script entry point
# Ensures execution only when the script is run directly
# ------------------------------------------------------------
if __name__ == "__main__":
    main()