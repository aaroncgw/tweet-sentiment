import pandas as pd
from collections import Counter
import re
import os


REGEX_DICT = {'links': "http\S+",
              'hashtags': "\B#\w*[a-zA-Z]+\w*",
              'emails': "[a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+",
              'adds': "\s([@][\w_-]+)"}


def from_raw_txt_to_csv(input_directory='data', output_file='data/tweets.csv'):
    # Obtain a list of all text files with raw tweet data. One file per handle
    raw_handle_files = os.listdir(input_directory)
    # List of all handles
    raw_handles = [raw_handle_file.rstrip('.txt') for raw_handle_file in raw_handle_files]

    # List of all tweets in raw form
    raw_tweets = []
    for raw_handle_file in raw_handle_files:
        with open(input_directory + '/' + raw_handle_file, 'r') as f:
            raw_tweets += f.readlines()

    # Remove tweets that dont have an int as first element (tweet_id)
    raw_tweets = [raw_tweet for raw_tweet in raw_tweets if raw_tweet.split(' ')[0].isdigit()]

    # Split tweets in raw form into tweet_id, timestamp, handle, raw_tweet
    df_raw_tweets_input = [[int(raw_tweet.split(' ')[0]),
                            ' '.join(raw_tweet.split(' ')[1:3]),
                            raw_tweet.split(' ')[4].strip('<>'),
                            ' '.join(raw_tweet.split(' ')[5:]).rstrip('\n')] for raw_tweet in raw_tweets]

    # Create dataframe and sort by tweet_id
    tweets_df = pd.DataFrame(df_raw_tweets_input, columns=['tweet_id', 'timestamp', 'handle', 'tweet'])
    tweets_df.sort_values('tweet_id', inplace=True)

    # Remove tweets whose handle isnt in the raw_handle list
    tweets_df = tweets_df[tweets_df.handle.isin(raw_handles)].reset_index(drop=True)

    tweets_df.to_csv(output_file, index=False)

    print("From raw text files to csv tweet database successful")


def read_raw_data(file='data/tweets.csv'):

    print("Reading data")

    data = pd.read_csv(file, index_col=['timestamp'], parse_dates=True)
    # Drop NAs
    data.dropna(inplace=True)
    # Remove tweets whose timestamp order doesn't match tweet_id order, original file is sorted by tweet_id
    data = data[data.index == data.index.sort_values()]

    return data


def get_clean_data(file='data/tweets.csv'):

    tweet_data = read_raw_data(file)

    for key, value in REGEX_DICT.items():
        print('Filtering', key)
        tweet_data.tweet = tweet_data.tweet.apply(lambda tweet: re.sub(value, '', tweet))

    print("Cleaning tweets")
    tweet_data.tweet = tweet_data.tweet.apply(_clean)

    return tweet_data


def _clean(tweet):

    # Turn everything to lower case
    tweet = tweet.lower()
    # Remove symbols other than letters in the alphabet and numbers
    tweet = re.sub(r"\'", '', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9]', ' ', tweet)

    return tweet


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

    return ' '.join(tweet_data.tweets)


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
