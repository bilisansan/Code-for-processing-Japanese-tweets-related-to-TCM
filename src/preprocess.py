import re
import emoji
import pandas as pd
import MeCab
from bs4 import BeautifulSoup  # Used to process HTML entities


def remove_all_urls(text):
    """
    Remove HTML tags, URLs, emojis, and extra whitespace from text 
    using a comprehensive regular expression pattern.
    """

    # ------------------------------------------------------------
    # Comprehensive URL matching pattern
    # ------------------------------------------------------------
    url_pattern = re.compile(
        r'('
        r'(?:https?|ftp)://[^\s/$.?#][^\s]*'  # URLs with protocol
        r'|'
        r'www\.[^\s/$.?#][^\s]*'              # URLs starting with www
        r'|'
        r'(?<!\w@)'                           # Ensure it is not part of an email
        r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'   # Plain domain names
        r'(?::\d+)?'                          # Optional port number
        r'(?:/[^\s]*)?'                       # Optional path
        r'(?=\s|$|[,.!?])'                    # Boundary detection
        r'(?!\w+@)'                           # Exclude email suffix
        r')'
    )

    # Remove URLs
    text = url_pattern.sub('', text)

    # ------------------------------------------------------------
    # Post-processing cleanup
    # ------------------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()          # Remove redundant spaces
    text = re.sub(r'([,.!?])\1+', r'\1', text)        # Remove repeated punctuation

    return text


# ------------------------------------------------------------
# Basic text cleaning function
# This function performs multiple preprocessing operations
# tailored for social media text (e.g., tweets)
# ------------------------------------------------------------
def clean_text(text):

    if not isinstance(text, str):
        return ""

    # ------------------------------------------------------------
    # Remove HTML entities (e.g., &amp;)
    # ------------------------------------------------------------
    text = BeautifulSoup(text, 'html.parser').get_text()

    # ------------------------------------------------------------
    # Remove URLs
    # ------------------------------------------------------------
    text = remove_all_urls(text)

    # ------------------------------------------------------------
    # Remove user mentions (@username)
    # Japanese usernames may include Japanese characters
    # ------------------------------------------------------------
    text = re.sub(r'@[a-zA-Z0-9_ぁ-んァ-ン一-龥]+', '', text)

    # ------------------------------------------------------------
    # Remove retweet markers (RT)
    # ------------------------------------------------------------
    text = re.sub(r'^RT[\s:]?', '', text)
    text = re.sub(r'^RT\s*@?\w*:\s*|\s*RT\s*@?\w*:\s*', ' ', text).strip()
    text = re.sub(r'\bRT\b\s*(?:@\w+:\s*)?', '', text).strip()

    # ------------------------------------------------------------
    # Remove repeated symbols such as ^^
    # ------------------------------------------------------------
    text = re.sub(r'\^\^+', '', text)
    text = re.sub(r'\＾\＾+', '', text)

    # ------------------------------------------------------------
    # Remove specific punctuation marks
    # ------------------------------------------------------------
    text = text.replace(':', '')
    text = re.sub(r'；+', '', text)
    text = re.sub(r'＜+', '', text)
    text = re.sub(r'＞+', '', text)

    # ------------------------------------------------------------
    # Remove hashtag symbol while keeping the hashtag text
    # ------------------------------------------------------------
    text = re.sub(r'#([a-zA-Z0-9_ぁ-んァ-ン一-龥]+)', r'\1', text)

    # ------------------------------------------------------------
    # Remove special symbols while preserving common Japanese punctuation
    # ------------------------------------------------------------
    text = re.sub(r'[【】、♪★☆→↓←↑◆■□●○◎※〒△▽▼▲‼×･]', '', text)

    # ------------------------------------------------------------
    # Remove emojis
    # ------------------------------------------------------------
    text = emoji.replace_emoji(text, replace='')

    # ------------------------------------------------------------
    # Normalize full-width and half-width punctuation
    # ------------------------------------------------------------
    text = text.translate(str.maketrans('！？（）｛｝［］＃＠', '!?(){}[]#@'))

    # ------------------------------------------------------------
    # Remove redundant whitespace
    # ------------------------------------------------------------
    text = ' '.join(text.split())

    return text.strip()


def main():
    """
    Main procedure for cleaning tweet texts.
    The script loads tweet data, applies text preprocessing,
    and saves the cleaned corpus for downstream NLP tasks
    such as embedding generation or topic modeling.
    """

    # ------------------------------------------------------------
    # Read the target year from the configuration file
    # ------------------------------------------------------------
    with open("../config/cfg.txt", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("year:"):
                year = line.strip().split(":")[1]

    # ------------------------------------------------------------
    # Load tweet text data
    # ------------------------------------------------------------
    df = pd.read_csv(f'../data/tw_Text_{year}.csv', encoding='utf-8')

    text = df['Text']

    # ------------------------------------------------------------
    # Apply text preprocessing to each tweet
    # ------------------------------------------------------------
    df['cleaned_text'] = df['Text'].apply(clean_text)

    clean = df['cleaned_text']
    clean_list = clean.to_list()

    # ------------------------------------------------------------
    # Save cleaned tweets to a text file
    # Each line corresponds to one cleaned tweet
    # ------------------------------------------------------------
    with open(f'../data/clean_{year}.txt', 'w', encoding='UTF-8') as f:
        for line in clean_list:
            f.write(f"{line}\n")


# ------------------------------------------------------------
# Script entry point
# Ensures the script runs only when executed directly
# ------------------------------------------------------------
if __name__ == "__main__":
    main()