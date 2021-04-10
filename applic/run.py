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


#If attempting to run locally - local is set to 1, otherwise 0
local = 0

if local == 0:
    from app.message_length_estimator import message_lengths_words, message_length_char
else:
    from message_length_estimator import message_lengths_words, message_length_char


app = Flask(__name__)

    
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



def main():
    if local == 0:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run()


if __name__ == '__main__':
    from app.message_length_estimator import message_lengths_words, message_length_char
    main()