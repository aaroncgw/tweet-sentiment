import pandas as pd
from collections import Counter
import re

# https://github.com/explosion/spaCy/issues/2156 for hashtag with Spacy

REGEX_DICT = {'links': "http\S+",
              'pic_links': "pic.twitter.com\S+",
              'hashtags': "\B#\w*[a-zA-Z]+\w*",
              'emails': "[a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+",
              'adds': "\s([@][\w_-]+)"}


def read_raw_data(file='data/tweets.csv'):
    """
    Read csv file with raw tweet data

    Parameters:
        file: str, optional
            CSV file path
    Output:
        pandas DataFrame
    """

    print("Reading data")

    data = pd.read_csv(file, index_col=['timestamp'], parse_dates=True)
    # Drop NAs
    data.dropna(inplace=True)

    return data


def remove_timestamp_tweet_id_mismatch(input_file='data/tweets.csv', output_file='data/tweets.csv'):
    """
    Remove tweets whose tweet_id order doesn't match with the timestamp order

    Parameters:
        input_file: directory
            Directory where text files are located
        output_file: string
            File path where the file will be written to
    """

    tweets_df = pd.read_csv(input_file, index_col=['timestamp'], parse_dates=True)
    # Drop NAs
    tweets_df.dropna(inplace=True)
    # Remove tweets whose timestamp order doesn't match tweet_id order, original file is sorted by tweet_id
    tweets_df = tweets_df[tweets_df.index == tweets_df.index.sort_values()]

    tweets_df.to_csv(output_file, index=False)


def get_clean_data(file='data/tweets.csv'):
    """
    Reads csv file with raw tweet data and returns a DataFrame with tweets parsed and cleaned
    :param file: csv file path
    :return: pandas DataFrame
    """

    def clean(tweet):
        # Turn everything to lower case
        tweet = tweet.lower()
        # Remove symbols other than letters in the alphabet and numbers
        tweet = re.sub(r"\'", '', tweet)
        tweet = re.sub(r'[^a-zA-Z0-9]', ' ', tweet)

        return tweet

    tweet_data = read_raw_data(file)

    # Remove links, hashtags, emails, @s
    for key, value in REGEX_DICT.items():
        print('Filtering', key)
        tweet_data.tweet = tweet_data.tweet.apply(lambda tweet: re.sub(value, '', tweet))

    print("Cleaning tweets")
    tweet_data.tweet = tweet_data.tweet.apply(clean)

    return tweet_data


def get_links(file='data/tweets.csv'):

    tweet_data = read_raw_data(file)
    tweet_data['links'] = tweet_data.tweet.apply(lambda tweet: re.findall(REGEX_DICT['links'], tweet))

    return tweet_data.drop(columns=['tweet'])


def get_hashtags(file='data/tweets.csv'):

    tweet_data = read_raw_data(file)
    tweet_data['hashtags'] = tweet_data.tweet.apply(lambda tweet: re.findall(REGEX_DICT['hashtags'], tweet))

    return tweet_data.drop(columns=['tweet'])


def get_adds(file='data/tweets.csv'):

    tweet_data = read_raw_data(file)
    tweet_data['adds'] = tweet_data.tweet.apply(lambda tweet: re.findall(REGEX_DICT['adds'], tweet))

    return tweet_data.drop(columns=['tweet'])


def create_text(tweet_data):

    print("Creating text")

    return ' '.join(tweet_data.tweet)


def create_lookup_tables(text):

    print("Creating lookup tables")

    words = text.split()
    word_counts = Counter(words)
    # words sorted in descending frequency
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def create_int_words(text, vocab_to_int):

    words = text.split()

    return [vocab_to_int[word] for word in words]
