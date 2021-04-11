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


#Functions

def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
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

    return X, Y, target_colnames


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
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    
    pipeline_feat_base = Pipeline([('features', FeatureUnion([('nlp_pipeline', Pipeline([('vect', CountVectorizer()),
                                                                          ('tfidf', TfidfTransformer())])),
                                                      ('ml_wor', message_lengths_words()),
                                                      ('ml_char', message_length_char())
                                                      ])),
                      ('clf', MultiOutputClassifier(LogisticRegression(max_iter = 600)))])
    
    return pipeline_feat_base


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    f1_scores = []
    for ind, cat in enumerate(Y_test):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], y_pred[ind], zero_division = 1))
    
        f1_scores.append(f1_score(Y_test.values[ind], y_pred[ind], zero_division = 1))
  
    print('Base Model\nMinimum f1 score - {}\nBest f1 score - {}\nMean f1 score - {}'.format(min(f1_scores), max(f1_scores), round(sum(f1_scores)/len(f1_scores), 3)))
    
    


def save_model(model, model_filepath):
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


main()
"""
if __name__ == '__main__':
   
    
    main()
    
"""