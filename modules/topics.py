from .spacy import spacy_twitter_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


nlp = spacy_twitter_model()


class TopicSeries:
    def __init__(self):
        self.n_components = 5
        self.random_state = 42
        self.cv_dict = {}
        self.tfidf_dict = {}
        self.nmf_dict = {}
        self.lda_dict = {}

    def calculate_nmf(self, date, data):

        tfidf = TfidfVectorizer(tokenizer=self.twitter_tokenizer)
        tfidf_vecs = tfidf.fit_transform(data)

        nmf = NMF(n_components=self.n_components,
                  random_state=self.random_state)
        nmf.fit_transform(tfidf_vecs)

        self.tfidf_dict[date] = tfidf
        self.nmf_dict[date] = nmf

    def calculate_lda(self, date, data):

        cv = CountVectorizer(tokenizer=self.twitter_tokenizer)
        count_vecs = cv.fit_transform(data)

        lda = LatentDirichletAllocation(n_components=self.n_components,
                                        random_state=self.random_state)
        lda.fit_transform(count_vecs)

        self.cv_dict[date] = cv
        self.lda_dict[date] = lda

    # Static method to be able to use the tokenizer in Count and TfidfVectorizers,
    # and allow saving to pickle
    @staticmethod
    def twitter_tokenizer(d,
                          model=nlp,
                          urls=True,
                          stop_words=True,
                          lowercase=True,
                          alpha_only=True,
                          hashtags=False,
                          lemma=False):
        """Full tokenizer with flags for processing steps
        urls: If True, remove URLs
        stop_words: If True, removes stop words
        lowercase: If True, lowercases all tokens
        alpha_only: If True, removes all non-alpha characters
        lemma: If True, lemmatizes words

        The tokenizer also removes all URLS
        """
        parsed = model(d)
        # token collector
        tokens = []
        for t in parsed:
            # remove URLs
            if t.like_url or t._.is_piclink & urls:
                continue
            # only include stop words if stop words==True
            if t.is_stop & stop_words:
                continue
            # only include non-alpha is alpha_only==False
            if not t.is_alpha & alpha_only:
                if hashtags:
                    continue
                else:
                    if not t._.is_hashtag:
                        continue
            if lemma:
                t = t.lemma_
            else:
                t = t.text
            if lowercase:
                t.lower()
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
    # utility for displaying representative words per component for topic models
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        top_words_idx = topic.argsort()[::-1][:top_display]
        top_words = [word_features[i] for i in top_words_idx]
        print(" ".join(top_words))
