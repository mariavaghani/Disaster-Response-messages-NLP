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

from disaster import app


from .message_length_estimator import message_lengths_words, message_length_char
from .message_length_estimator import tokenize



print('going to load the database now')
# load data

engine = create_engine('sqlite:///data/DisasterResponse.db')  


df = pd.read_sql_table('disaster_resp_mes', engine)


print('going to load the pickle now')

# load model
model = joblib.load("models/classifier.pkl")

print('loaded the pickle now')




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
      
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    #Length char and length words data prep
    df['message_length_char'] = df['message'].apply(lambda x: len(x))
    df['message_length_words'] = df['message'].apply(lambda x: len(x.split()))

    df_char = pd.DataFrame(columns = ['message_length_char', 'message_length_words'])
    for feat in df.columns[4:-2]:
        row_to_append = df.groupby(by = feat).mean()[['message_length_char', 'message_length_words']].reset_index()
        df_char = df_char.append(row_to_append)

    df_char = df_char[df_char.index == 1][['message_length_char', 'message_length_words']]
    df_char['category'] = df.columns[4:-2]
    
    
    
    #assigning variables for plots
    y_val1 = df_char['message_length_char']
    y_val2 = df_char['message_length_words']
    x_val = df_char['category']
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
        },
        
        {
            'data': [
                Bar(
                    x=x_val,
                    y=y_val1
                )
            ],

            'layout': {
                'title': 'Average Character Length of Messages per Genre',
                'yaxis': {
                    'title': "Average Character Message Length"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=x_val,
                    y=y_val2
                )
            ],

            'layout': {
                'title': 'Average Word Count of Messages per Genre',
                'yaxis': {
                    'title': "Average Word Count"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        
    ]
    
    #me messing finish
    
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
    print(query)
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

 
#uncomment when running locally with disaster.py   
#main()

if __name__ == '__main__':

    
    
    main()
