from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Using Spacy's list of stopwords plus all letters
STOP_WORDS = list(STOP_WORDS.union(set(string.ascii_lowercase)))


class TopicSeries:
    """
    Class that holds a time series of Topic models
    """
    def __init__(self, n_components=5, random_state=42):
        """
        Parameters:
            n_components: int, optional
                Number of topics for each topic model
            random_state: int, optional
                Random seed for NMF and LatentDirichletAllocation
        """
        self.n_components = n_components
        self.random_state = random_state
        self.cv_dict = {}
        self.tfidf_dict = {}
        self.nmf_dict = {}
        self.lda_dict = {}

    def calculate_nmf(self, date, data):
        """
        Fit TF-IDF and NMF models from Sklearn

        Parameters:
            date: string
                Date in 'yyyy-mm-dd' format
            data: Pandas DataFrame
                Data for a particular date range for the output of read_raw_data() method in modules.tweet_data
        """
        # TF-IDF model, token_pattern is alpha only words + hashtags
        tfidf = TfidfVectorizer(stop_words=STOP_WORDS ,
                                token_pattern='^[A-Za-z]+|#\w*[a-zA-Z]+\w*')
        tfidf_vecs = tfidf.fit_transform(data)
        # Fit NMF model with n_components topics with TF-IDF input
        nmf = NMF(n_components=self.n_components,
                  random_state=self.random_state)
        nmf.fit_transform(tfidf_vecs)
        # Add TFIDF and NMF models to their respective dictionaries, with date as key
        self.tfidf_dict[date] = tfidf
        self.nmf_dict[date] = nmf

    def calculate_lda(self, date, data):
        """
        Fit CountVectorizer and LDA models from Sklearn

        Parameters:
            date: string
                Date in 'yyyy-mm-dd' format
            data: Pandas DataFrame
                Data for a particular date range for the output of read_raw_data() method in modules.tweet_data
        """
        # TF-IDF model, token_pattern is alpha only words + hashtags
        cv = CountVectorizer(stop_words=STOP_WORDS ,
                             token_pattern='^[A-Za-z]+|#\w*[a-zA-Z]+\w*')
        count_vecs = cv.fit_transform(data)
        # Fit LDA model with n_components topics with cv input
        lda = LatentDirichletAllocation(n_components=self.n_components,
                                        random_state=self.random_state)
        lda.fit_transform(count_vecs)
        # Add CountVectorizer and LDA models to their respective dictionaries, with date as key
        self.cv_dict[date] = cv
        self.lda_dict[date] = lda


def display_components(model, word_features, top_display=5):
    """
    Displays the top words by probability in each topic of a topic model

    Parameters:
        model: Sklearn topic model such as LatentDirichletAllocation or NMF
        word_features:
            Output from get_feature_names() method from count vectorizer used to fit model
        top_display: int, optional
            Number of words per topic displayed
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        top_words_idx = topic.argsort()[::-1][:top_display]
        top_words = [word_features[i] for i in top_words_idx]
        print(" ".join(top_words))
