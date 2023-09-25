"""
This is where the magic elves do the dirty work that no one wants
to look at, but no one can survive without either. Works may include
text pre-processing and other ugly functions.
"""

import spacy

# Load english
nlp = spacy.load('en_core_web_sm')

def tokenize(text) -> list:
    """
    Removes stop words, lemmatizes words and returns
    a tokenized string.

    Args:
        text (`str`): The text to tokenize.
    
    Returns:
        tokens (`list`): A list of tokens.
    """

    # Tokenize with SpaCy
    tokens = nlp(text)
    # Remove stop words and punctuation and lemmatize text
    return [t.lemma_ for t in tokens if not (t.is_puct or t.is_stop)]
