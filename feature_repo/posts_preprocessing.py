import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

# Регулярное выражение для удаления смайлов
emoji_pattern = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Регулярное выражение для удаления ссылок
url_pattern = re.compile(r"https?://\S+")


def text_preprocessing(text: str) -> str:
    """_summary_

    Args:
        text (str): input disription of vacansy

    Returns:
        str: output preprocessed discription of vacancy
    """
    text = emoji_pattern.sub(r"", text)
    text = url_pattern.sub(r"", text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    return " ".join(filtered_tokens)


def posts_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Posts preprocessing

    Args:
        df (pd.DataFrame): input_dataframe

    Returns:
        pd.DataFrame: output_dataframe
    """
    df["content"] = df["content"].apply(text_preprocessing)
    return df
