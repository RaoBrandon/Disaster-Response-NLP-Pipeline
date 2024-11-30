'''Importing required libraries for flask application'''
# First party libraries
import json
import plotly
import pandas as pd
import os

# NLTK libraries for language processing
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import flask libraries for application deployment
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)
accuracy = pd.read_sql('SELECT * FROM model_accuracy ORDER BY Accuracy', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Calculate the correlation matrix for our second visualization
    corr_matrix = df.iloc[:, 4:].corr()

    # Save features and accuracy values for plotting
    feature_names = accuracy.Feature
    feature_accuracy = accuracy.Accuracy
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                        colorscale='YlGnBu',
                        color=genre_counts,
                    )
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
                Heatmap(
                    z=corr_matrix.values,       # Correlation values
                    x=corr_matrix.columns,      # Column names
                    y=corr_matrix.index, 
                    colorscale = 'YlGnBu',
                    colorbar = dict(title = 'Correlation')
                )
            ],

            'layout': {
                'title': 'Correlation Heatmap',
                'yaxis': {
                    'title': "Features"
                },
                'xaxis': {
                    'title': "Features"        
                }
            }
        },
        {
            'data': [
                Bar(
                    x=feature_accuracy,
                    y=feature_names,
                    orientation= 'h',
                    marker=dict(
                        colorscale='YlGnBu',
                        color=feature_accuracy,
                        colorbar=dict(title="Accuracy")
                    )
                )
            ],

            'layout': {
                'title': 'Genre Accuracy Ratings',
                'yaxis': {
                    'title': "Feature Name"
                },
                'xaxis': {
                    'title': "Accuracy Score",
                    'range': [0.75, 1]    
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()