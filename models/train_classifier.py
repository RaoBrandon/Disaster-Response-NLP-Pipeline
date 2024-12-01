'''Import libraries required for project, organized by component'''

# Import required first party libraries
import sys
import re
import pickle
import pandas as pd

# Import nltk libraries for langauge processing
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Create engine for interacting with databases
from sqlalchemy import create_engine

# Importing sklearn libraries for ML classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    '''
    This function takes in the file path of a database and returns a dataframe for the target input 
    and output variables as well as the list of column names.

    Args:
        database file path - a path to a sqlite database containing the prepped data
        from process_data.py

    Returns:
        x - independent variable 'message' for classification as a single-column pandas df 
        y - target variable, a classification of one of over 30 possible topics in a pandas 
        dataframe
    '''

    # load data from database, placing into target dataframes
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM categorized_messages", engine)
    x = df['message']
    y = df.drop(columns = ['id', 'message', 'original', 'genre'])

    return x, y, list(y.columns)


def tokenize(text):
    '''
    This function takes in a text string and normalizes, tokenizes and lemmatizes it for processing.

    Args:
        text - raw text, needs to be cleaned for further processing

    Returns:
        tokens - python list of cleaned, tokenized and lemmatized text for easy processing
    '''

    # Remove punctuation and normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Declare stop words
    stop_words = stopwords.words("english")

    # Turn text into tokenized words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model(X_train, Y_train):
    '''
    This function intakes no parameters and simply returns an initialized pipeline leveraging
    a random forest classifier in combination with multi output classifier as suggested.

    Args:
        None
    
    Returns:
        Intantiated ML pipeline, a model to be trained for a multi output classification using
        random forest
    '''

    # Intantiate and return model for training, leveraging multi output classifier
    rf_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Now, we will perform a Grid Search Cross Validation to optimize our model
    parameters = {
    'clf__estimator__n_estimators': [10, 20, 30],
    'clf__estimator__max_depth': [5, 25, 50]
    }

    cv = GridSearchCV(rf_pipeline, parameters)

    # Fit the grid search on the data and determine the best model, then return it
    cv.fit(X_train, Y_train)
    best_model = cv.best_estimator_

    return best_model


def evaluate_model(model, x_test, y_test, category_names):
    '''
    This function evaluates our model using classification_report on several key metrics.
    There is no return value, however this model evaluation will allow us to track our progress.
    '''

    # Make predictions, initialize empty list for accuracy results
    y_pred = model.predict(x_test)
    results = []


    #Iterate through the features to track our prediction accuracy
    for column in category_names:
        print(f"Metrics for {column}:")
        print(classification_report(y_test[column], y_pred[:, y_test.columns.get_loc(column)]))
        print("="*50)

        #Calculate accuracy and store it
        accuracy = accuracy_score(y_test[column], y_pred[:, y_test.columns.get_loc(column)])

        # Create separate table for accuracy scores to visualize later
        results.append({
            'Feature' : column,
            'Accuracy': accuracy
        })

    accuracy_df = pd.DataFrame(results)

    # create sqlite engine and save table to database for later usage
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    accuracy_df.to_sql('model_accuracy', engine, if_exists='replace', index=False)


def save_model(model, model_filepath):
    '''
    This function takes a model and a filepath and dumps a pickle file of that model into the path.

    Args:
        model - model that has been created for classification predictions for disasters
    
    Returns:
        None - simply writes a pickle file to the provided directory
    '''

    # Write pickle file to provided directory
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    This function executes all components of our classifier training, evaluation and output.
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(X_train, Y_train)

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
