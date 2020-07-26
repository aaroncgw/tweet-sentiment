from .spacy import spacy_twitter_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Spacy model as input for twitter tokenizer. Loading it here to allow for both direct tokenization
# in CountVectorizer and TfidfVectorizer, and saving Topicseries class to pickle
nlp = spacy_twitter_model()


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
        # TF-IDF model, using twitter_tokenizer as tokenizer
        tfidf = TfidfVectorizer(tokenizer=self.twitter_tokenizer)
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
        # CountVectorizer model, using twitter_tokenizer as tokenizer
        cv = CountVectorizer(tokenizer=self.twitter_tokenizer)
        count_vecs = cv.fit_transform(data)
        # Fit LDA model with n_components topics with cv input
        lda = LatentDirichletAllocation(n_components=self.n_components,
                                        random_state=self.random_state)
        lda.fit_transform(count_vecs)
        # Add CountVectorizer and LDA models to their respective dictionaries, with date as key
        self.cv_dict[date] = cv
        self.lda_dict[date] = lda

    # Static method to be able to use the tokenizer in Count and TFIDF vectorizers,
    # and allow saving to pickle
    @staticmethod
    def twitter_tokenizer(data,
                          model=nlp,
                          urls=True,
                          stop_words=True,
                          lowercase=True,
                          alpha_only=True,
                          hashtags=False,
                          lemma=False):
        """
        Full tokenizer with flags for processing steps

        Parameters:
            data: string
                String to be tokenized
            model: Spacy model
                Ideally, an output from the method spacy_twitter_model() from modules.spacy
            urls: bool
                If True, remove URLs and Twitter picture links
            stop_words: bool
                If True, removes stop words
            lowercase: bool
                If True, turns all tokens to lowercase
            alpha_only: bool
                If True, removes all non-alpha characters
            hashtags: bool
                If True, removes hashtags
            lemma: bool
                If True, lemmatizes words
        """
        parsed = model(data)
        # token collector
        tokens = []
        for t in parsed:
            # remove URLs abd Twitter picture links
            if t.like_url or t._.is_piclink & urls:
                continue
            # remove stopwords
            if t.is_stop & stop_words:
                continue
            # alpha characters only
            if not t.is_alpha & alpha_only:
                # if not alpha only, remove hashtags
                if hashtags:
                    continue
                else:
                    if not t._.is_hashtag:
                        continue
            # lemmatize
            if lemma:
                t = t.lemma_
            else:
                t = t.text
            # turn to lowercase
            if lowercase:
                t = t.lower()
            tokens.append(t)
        return tokens


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
