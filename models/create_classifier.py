# import libraries
import sys

from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd
import re


import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin




from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#Classes

class message_length_char(BaseEstimator, TransformerMixin):
    #get how many characters in string
    def message_length_chars(self, text):
          
      tran = len(text)
      return tran
      
    def fit(self, x, y=None):
        return self

    def normalize(self, x, x_min, x_max):
    
      # TODO: Complete this function
      # The input is a single value 
      # The output is the normalized value
      return (x - x_min)/(x_max - x_min)


    def transform(self, X):
        # apply length_char function to all values in X
        X_tagged_char = pd.Series(X).apply(self.message_length_chars)
        #normalize the series
        x_min = min(X_tagged_char)
        x_max = max(X_tagged_char)
        
        X_tagged_char_norm = pd.Series(X_tagged_char).apply(self.normalize, x_min = x_min, x_max = x_max)

        return pd.DataFrame(X_tagged_char_norm)


class message_lengths_words(BaseEstimator, TransformerMixin):

    def message_length_words(self, text):
      # tokenize by words, how many words in message
      word_list_tok = word_tokenize(text)

      return len(word_list_tok)

      
    def fit(self, x, y=None):
        return self
    
    def normalize(self, x, x_min, x_max):
    
      
      return (x - x_min)/(x_max - x_min)


    def transform(self, X):
        # apply length_word function to all values in X
        
        X_tagged_words = pd.Series(X).apply(self.message_length_words)
        #normalize the series
        
        x_min = min(X_tagged_words)
        x_max = max(X_tagged_words)
        
        X_tagged_words_norm = pd.Series(X_tagged_words).apply(self.normalize, x_min = x_min, x_max = x_max)

        return pd.DataFrame(X_tagged_words_norm)


def make_model_for_pickle():

    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM disaster_resp_mes;", engine)
    
    #get category names
    inspector = inspect(engine)
    schema = inspector.get_schema_names()[0]
    colnames = []
    table_name = inspector.get_table_names(schema=schema)[0]
    for column in inspector.get_columns(table_name, schema=schema):
        colnames.append(column['name'])
    
    target_colnames = colnames[4:]
    
    #assign X and Y
    X = df['message']
    Y = df[target_colnames]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    
    pipeline_feat_base = Pipeline([('features', FeatureUnion([('nlp_pipeline', Pipeline([('vect', CountVectorizer()),
                                                                          ('tfidf', TfidfTransformer())])),
                                                      ('ml_wor', message_lengths_words()),
                                                      ('ml_char', message_length_char())
                                                      ])),
                      ('clf', MultiOutputClassifier(LogisticRegression(max_iter = 600)))])
    
    model = pipeline_feat_base.fit(X_train, Y_train)
    
    return model
