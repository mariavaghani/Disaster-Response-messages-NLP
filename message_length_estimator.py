from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import pandas as pd


def normalize(x, x_min, x_max):
    
      # The output is the normalized value
      return (x - x_min)/(x_max - x_min)


class message_lengths_words(BaseEstimator, TransformerMixin):

    def message_length_words(self, text):
      # tokenize by words, how many words in message
      word_list_tok = word_tokenize(text)

      return len(word_list_tok)

      
    def fit(self, x, y=None):
        return self
    """
    def fit_transform(self, X):
        # apply length_word function to all values in X
        print(self.message_length_words)
        X_tagged_words = pd.Series(X).apply(self.message_length_words)


        return pd.DataFrame(X_tagged_words)
    """

    def transform(self, X):
        # apply length_word function to all values in X
        
        X_tagged_words = pd.Series(X).apply(self.message_length_words)
        #normalize the series
        
        x_min = min(X_tagged_words)
        x_max = max(X_tagged_words)
        
        X_tagged_words_norm = pd.Series(X_tagged_words).apply(normalize, 
                                                              x_min = x_min, 
                                                              x_max = x_max)

        return pd.DataFrame(X_tagged_words_norm)
    
    
    
    
class message_length_char(BaseEstimator, TransformerMixin):
    #get how many characters in string
    def message_length_chars(self, text):
          
      tran = len(text)
      return tran
      
    def fit(self, x, y=None):
        return self
    """
    def fit_transform(self, X):
        # apply length_char function to all values in X
        X_tagged_char = pd.Series(X).apply(self.message_length_char)

        return pd.DataFrame(X_tagged_char)
      
    """


    def transform(self, X):
        # apply length_char function to all values in X
        X_tagged_char = pd.Series(X).apply(self.message_length_chars)
        #normalize the series
        x_min = min(X_tagged_char)
        x_max = max(X_tagged_char)
        
        X_tagged_char_norm = pd.Series(X_tagged_char).apply(normalize, 
                                                            x_min = x_min, 
                                                            x_max = x_max)

        return pd.DataFrame(X_tagged_char_norm)
