import pandas as pd
from collections import Counter
import re
import os


class TweetData:

    def __init__(self):

        self.regex_dict = {'links': "http\S+",
                           'hashtags': "\B#\w*[a-zA-Z]+\w*",
                           'emails': "[a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+",
                           'adds': "\s([@][\w_-]+)"}

    def from_raw_txt_to_csv(self, input_directory='data', output_file='data/tweets.csv'):

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

    def read_raw_data(self, file='data/tweets.csv'):

        print("Reading data")

        tweet_data = pd.read_csv(file, index_col=['timestamp'], parse_dates=True)
        # Drop NAs
        tweet_data.dropna(inplace=True)
        # Remove tweets whose timestamp order doesn't match tweet_id order, original file is sorted by tweet_id
        tweet_data = tweet_data[tweet_data.index == tweet_data.index.sort_values()]

        return tweet_data

    def get_clean_data(self, file='data/tweets.csv'):

        tweet_data = self.read_raw_data(file=file)

        for key, value in self.regex_dict.items():
            print('Filtering', key)
            tweet_data.tweet = tweet_data.tweet.apply(lambda tweet: re.sub(value, '', tweet))

        print("Cleaning tweets")
        tweet_data.tweet = tweet_data.tweet.apply(self._clean)

        return tweet_data

    def get_links(self, file='data/tweets.csv'):

        tweet_data = self.read_raw_data(file=file)
        tweet_data['links'] = tweet_data.tweet.apply(lambda tweet: re.findall(self.regex_dict['links'], tweet))

        return tweet_data.drop(columns=['tweet'])

    def get_hashtags(self, file='data/tweets.csv'):

        tweet_data = self.read_raw_data(file=file)
        tweet_data['hashtags'] = tweet_data.tweet.apply(lambda tweet: re.findall(self.regex_dict['hashtags'], tweet))

        return tweet_data.drop(columns=['tweet'])

    def get_adds(self, file='data/tweets.csv'):

        tweet_data = self.read_raw_data(file=file)
        tweet_data['adds'] = tweet_data.tweet.apply(lambda tweet: re.findall(self.regex_dict['adds'], tweet))

        return tweet_data.drop(columns=['tweet'])

    def _clean(self, tweet):

        # Turn everything to lower case
        tweet = tweet.lower()
        # Remove symbols other than letters in the alphabet and numbers
        tweet = re.sub(r"\'", '', tweet)
        tweet = re.sub(r'[^a-zA-Z0-9]', ' ', tweet)

        return tweet

    def create_text(self, tweet_data):
        text = ' '.join(tweet_data.tweets)
        return text

    def tokenize_text(self, text):
        words = text.split()
        return words

    def create_lookup_tables(self, words):

        word_counts = Counter(words)
        # words sorted in descending frequency
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
        return vocab_to_int, int_to_vocab

    def create_int_words(self, vocab_to_int):
        return [vocab_to_int[word] for word in self.words]
