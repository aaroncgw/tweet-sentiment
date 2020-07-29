from spacy.tokenizer import _get_regex_pattern
from spacy.tokens import Token
import spacy
import re


# Idea from https://stackoverflow.com/questions/43388476/how-could-spacy-tokenize-hashtag-as-a-whole
def spacy_twitter_model(model='en_core_web_sm'):
    """
    Load Spacy model, adding capability to detect Twitter picture links and hashtags
    into Spacy's tokenizer
    Parameters:
        model: string, optional
            Name of Spacy model to load. Defaults to en_core_web_sm
    Returns:
        Spacy model
    """
    nlp = spacy.load(model)

    nlp.Defaults.stop_words |= {'yeah', 'yep', 'ah', 'nah', 'lol', 'oh', 'yes', 'ha', 'haha', 'hahaha', 'maybe',
                                'like', 'cc', 'let', 'thank', 'thanks', 'sorry', 'fwiw', 'wow', 'icymi'}
    # get default pattern for tokens that don't get split
    re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
    # add your patterns (here: hashtags and in-word hyphens)
    re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"
    # overwrite token_match function of the tokenizer
    nlp.tokenizer.token_match = re.compile(re_token_match).match
    # set a custom extension to match if token is a piclink and hashtag
    Token.set_extension('is_piclink',
                        getter=lambda token: bool(re.match("pic.twitter.com\S+", token.text)),
                        force=True)
    Token.set_extension('is_hashtag',
                        getter=lambda token: bool(re.match("#\w+", token.text)),
                        force=True)

    return nlp
