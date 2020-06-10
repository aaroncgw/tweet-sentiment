import pandas as pd
import twint


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


scrape_twitter_handles()

