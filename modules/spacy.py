from spacy.tokenizer import _get_regex_pattern
from spacy.tokens import Token
import spacy
import re


def spacy_twitter_model():

    nlp = spacy.load('en_core_web_sm')

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
