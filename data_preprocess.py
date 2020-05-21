import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

CLEAN_COL = 'clean_data'


def clean_col(col):
    """
    Cleans an input string using the following steps:
    """
    col = col.str.lower()
    col = col.str.replace(r"\s+", " ")
    col = col.str.encode("utf8", errors="ignore").str.decode("utf8")
    col = col.str.replace("-", "", regex=False)
    col = col.str.replace("`", "", regex=False)  # remove apostrophe
    col = col.str.replace("\'", "", regex=False)  # remove apostrophe
    col = col.str.replace(r"\S*@\S*", " ")
    col = col.str.replace(r"(http\S+|http\S+)", " ")
    col = col.str.replace("_", " ", regex=False)

    col = col.str.replace(r"(\d+\/\d+\/?\d*|\d+\.\d+\.?\d*|\d+\\\d+\\?\d*)", " ")
    col = col.str.replace(r"\d+:\d+:?\d*", " ")
    col = col.str.replace(r"(\d+\.\d+|\d+\,\d+|\+\d+|(?<=\s|\.|,)\d+|\d+(?=\s|\.|,)|\#\d+\b|\+\d+\b)", " ")
    col = col.str.replace(r"(?<=\w)([^\w\.,\s\:\/\\+])(?=\w+)", "")
    col = col.str.replace(r"\,(?=\w\D)", ", ")
    col = col.str.replace(r"\.(?=\w\D)", ". ")
    col = col.str.replace(r"([^\w\s])", " ")

    col = col.str.replace(r"((?<=\D)\d+(?=\D)|\d+(?=\D)|(?<=\D)\d+)", " ")
    col = col.str.replace(r"\b\d+\b", " ")
    col = col.str.replace(r" +", " ")  # remove extra spaces

    col = col.str.strip()
    # col = col.apply(lambda row: " ".join(row.split(" ")[:1000]))  # Limit text to 1000 words

    return col


def data_preprocess(df: pd.DataFrame, column: str, clean_data: bool = True, remove_stopwords: bool = False,
                    stem_lemma: bool = False, sentiment: bool = False):
    """

    :param df: pd.DataFrame, input data frame
    :param column: string, column name for the column which needs preprocessing
    :param clean_data:  bool, default True, control the clean data process
    :param remove_stopwords: bool, default False, control the remove stopwords process
    :param stem_lemma: bool, default False, control the lemmatization process
    :param sentiment: bool, default False, get the sentiment results from Textblob
    :return: pd.DataFrame with preprocessed text column
    """
    if clean_data:
        df[CLEAN_COL] = clean_col(df[column])
        df.drop(df.loc[df[CLEAN_COL].str.len() < 2, CLEAN_COL].index, inplace=True)
    else:
        df[CLEAN_COL] = df[column]

    if remove_stopwords:
        en_stopwords = stopwords.words('english')
        df[CLEAN_COL] = df[CLEAN_COL].apply(
            lambda x: ' '.join([w for w in word_tokenize(x) if w not in en_stopwords]))
    if stem_lemma:
        # stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        df[CLEAN_COL] = df[CLEAN_COL].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(x)]))

    if sentiment:
        df['polarity'] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df[column].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return df

def get_word_frequency(corpus, ngram=1, n=None) -> (pd.DataFrame, CountVectorizer, np.array):
    vec = CountVectorizer(ngram_range=(ngram, ngram), ).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_freq_df = pd.DataFrame(words_freq, columns=['words', 'counts'])
    return words_freq_df.iloc[:n], vec, bag_of_words


def wordcloud_plot(corpus, filename):
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", width=1000, height=1000, margin=2, mask=mask)
    word_freq_df, _, _ = get_word_frequency(corpus)
    word_freq_dict = {item['words']: item['counts'] for index, item in word_freq_df.iterrows()}
    wc.generate_from_frequencies(word_freq_dict)
    plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(figure_num, 1, index)
    plt.imshow(wc)
    wc.to_file(filename)
    # index += 1
    plt.show()
    plt.close()
