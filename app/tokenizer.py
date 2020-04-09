from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('wordnet')

class Tokenizer:

    def tokenize(text):
        """tokenization and lemmatization of raw text input
        
        Arguments:
            text {string} -- raw text
        
        Returns:
            list of tokens -- returns the tokenized and lemmatized text
        """
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens