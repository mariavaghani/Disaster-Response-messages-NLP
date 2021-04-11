import json
import plotly
import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression

from applic import app

#If attempting to run locally - local is set to 1, otherwise 0
local = 0

"""
if local == 0:
    from applic.message_length_estimator import message_lengths_words, message_length_char
else:
    from message_length_estimator import message_lengths_words, message_length_char
"""

#app = Flask(__name__)

#Define pickled classes

from sklearn.base import BaseEstimator, TransformerMixin

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

    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


print('going to load the database now')
# load data

if local == 0:
    engine = create_engine('sqlite:///data/DisasterResponse.db')    
else:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')


df = pd.read_sql_table('disaster_resp_mes', engine)


print('going to load the pickle now')

# load model
if local == 0: 
   model = joblib.load("models/classifier.pkl")
else:
    model = joblib.load("../models/classifier.pkl")

print('loaded the pickle now')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
      
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

"""

def main():
    if local == 0:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run()



if __name__ == '__main__':
    #from applic.message_length_estimator import message_lengths_words, message_length_char
    
    
    main()
"""