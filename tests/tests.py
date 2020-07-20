import tempfile
import pandas as pd
import sys
sys.path.append('.')
from modules.scrape import from_raw_txt_to_csv
from modules.tweet_data import get_clean_data


def test_tweet_data():
    pass
    # Create two text files
    text_file1 = '4 2020-06-08 16:37:45 +04 <handle1> @handle2 A twEet by hAndle1 a@email.com\n'
    text_file1 += '3 2020-05-08 15:37:45 +04 <handle1> Another tweet by handle1 #test #1 http://test.com\n'
    text_file1 += 'Error line'

    text_file2 = '1 2019-03-08 16:37:45 +04 <handle2> A tweet by handle 2 $SPX #1hashtag\n'
    text_file2 += '2 2020-05-08 15:37:44 +04 <handle2> @peter @paul Another tweet by handle 2\n'

    # The expected clean pandas df from these text files is:
    target_df = pd.DataFrame(columns=['timestamp', 'tweet_id', 'handle', 'tweet'])
    target_df.tweet_id = [1, 2, 3, 4]
    target_df.handle = ['handle2', 'handle2', 'handle1', 'handle1']
    target_df.tweet = ['a tweet by handle 2',
                       'another tweet by handle 2',
                       'another tweet by handle1 1',
                       'a tweet by handle1'
                       ]
    target_df.timestamp = pd.to_datetime(['2019-03-08 16:37:45',
                                          '2020-05-08 15:37:44',
                                          '2020-05-08 15:37:45',
                                          '2020-06-08 16:37:45'])
    target_df.set_index('timestamp', inplace=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create handle text files
        with open(tmpdir + '/handle1.txt', 'w') as tf1:
            tf1.write(text_file1)
        with open(tmpdir + '/handle2.txt', 'w') as tf2:
            tf2.write(text_file2)

        # Convert text files to csv
        from_raw_txt_to_csv(tmpdir, tmpdir + '/tweets.csv')
        # Get cleaned pandas df
        tweet_data = get_clean_data(tmpdir + '/tweets.csv')

        pd.testing.assert_frame_equal(target_df, tweet_data)


test_tweet_data()




