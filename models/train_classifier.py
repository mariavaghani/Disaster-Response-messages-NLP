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
from sklearn.model_selection import GridSearchCV



from message_length_estimator import message_lengths_words, message_length_char
from message_length_estimator import tokenize

from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

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





def build_model():
    """
    generates an NLP model that is ready to be fit with training data
    -------

    """
    
    pipeline_feat_base = Pipeline([('features', FeatureUnion([('nlp_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                                          ('tfidf', TfidfTransformer())])),
                                                      ('ml_wor', message_lengths_words()),
                                                      ('ml_char', message_length_char())
                                                      ])),
                      ('clf', MultiOutputClassifier(LogisticRegression(max_iter = 600)))])
    
    # choose parameters
    parameters = {'clf__estimator__max_iter': [600, 800],
                  'clf__estimator__C': [0.5, 1.0]}

    # create grid search object
    pipeline_feat_base_cv = GridSearchCV(pipeline_feat_base, param_grid=parameters, scoring='recall_micro', cv=4)
    
    return pipeline_feat_base_cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    f1_scores = []
    for ind, cat in enumerate(Y_test):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], y_pred[ind], zero_division = 1))
    
        f1_scores.append(f1_score(Y_test.values[ind], y_pred[ind], zero_division = 1))
  
    print('Trained Model\nMinimum f1 score - {}\nBest f1 score - {}\nMean f1 score - {}'.format(min(f1_scores), max(f1_scores), round(sum(f1_scores)/len(f1_scores), 3)))
    print("\nBest Parameters:", model.best_params_)
    


def save_model(model, model_filepath):
    """

    Parameters
    ----------
    model : ML model
        trained and ready to be deployed to production.
    model_filepath : string
        distination to be saved.

    """
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



if __name__ == '__main__':
   
   
    main()
    
