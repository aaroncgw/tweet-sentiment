import pandas as pd
import twint
import requests
import zipfile
import tempfile
import os


def scrape_twitter_handles(handles_path='data/handles.csv', handles_directory='data/handles_raw_data', start_from=0):
    """
    Scrape all tweets from a list of Twitter handles using the twint library

    Input:
    handles_path: CSV file path containing a column handles where its elements are Twitter handles
    handles_directory: directory where text files will be written to
    start_from: (int) column index from where to start scraping

    Output: A text file per Twitter handle with all their respective tweets
    """

    # Load file with all twitter handles to scrape
    names = pd.read_csv(handles_path)

    for name in names.handles[start_from:]:

        # Get all tweets for a twitter handle and write them to a txt file
        with open(handles_directory + '/{}.txt'.format(name), 'w+') as f:
            c = twint.Config()
            c.Username = name
            c.Output = handles_directory + '/{}.txt'.format(name)
            twint.run.Search(c)


# Adapted from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(id, destination):
    """
    :param id: string, Google Drive file ID
    :param destination: file path where the file will be downloaded to
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def download_txt_files_from_google_drive(id='1zZ02PQKig89mikY5Xhks0vtvzcsr3Dd1', destination='data/handles_raw_data'):
    """
    Download zip file with .txt files which each contain tweets per handle
    :param id: string, Google Drive file ID
    :param destination: file path where the text files will be unzipped to
    """
    # Download zip file to a temporary directory and unzip there,
    with tempfile.TemporaryDirectory() as tmpdir:
        download_file_from_google_drive(id, tmpdir + 'txt_files.zip')
        # Unzip into destination file
        with zipfile.ZipFile(tmpdir + 'txt_files.zip', "r") as zip_ref:
            zip_ref.extractall(destination)


def from_raw_txt_to_csv(input_directory='data/handles_raw_data', output_file='data/tweets.csv'):
    """
    Convert raw text files per handle into a csv with columns 'tweet_id', 'timestamp', 'handle', 'tweet'
    :param input_directory: directory where text files are located
    :param output_file: path to csv file where data will be written to
    """
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


