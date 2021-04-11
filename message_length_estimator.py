from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import pandas as pd

from nltk.stem import WordNetLemmatizer

def tokenize(text):
    
    """
    INPUT:
    text - string
    OUTPUT:
    tokens - list of strings
    
    function takes raw text, removes punctuation signs, substitutes
    with spaces. Puts all characters in lower case, tokenizes text
    by words, removes stop words, lemmatizes, and returns list of tokens 
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class message_lengths_words(BaseEstimator, TransformerMixin):
    
    """
    Estimator that takes a column of strings, and converts them into a column
    of integers, representing the word count of each string
    """
        
    def message_length_words(self, text):
      # tokenize by words, how many words in message
      word_list_tok = word_tokenize(text)

      return len(word_list_tok)

      
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply length_word function to all values in X
        
        X_tagged_words = pd.Series(X).apply(self.message_length_words)

        return pd.DataFrame(X_tagged_words)
    
    
    
    
class message_length_char(BaseEstimator, TransformerMixin):
    
    """
    Estimator that takes a column of strings, and converts them into a column
    of integers, representing the character length of each string
    """
    
    #get how many characters in string
    def message_length_chars(self, text):
          
      tran = len(text)
      return tran
      
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply length_char function to all values in X
        X_tagged_char = pd.Series(X).apply(self.message_length_chars)
        

        return pd.DataFrame(X_tagged_char)
