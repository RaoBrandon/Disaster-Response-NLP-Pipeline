'''Importing required libraries for data processing'''

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function returns the concatenation of 2 dataframes given the paths of their CSV files

    Args:
        Messages filepath, Categories file path

    Returns:
        1 pandas dataframe, a horizontal concatenation of both files
    '''

    # Read both filepaths into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.concat([messages, categories], axis=1, join="inner")    

    return df


def clean_data(df):
    '''
    This function intakes a pandas dataframe and outputs a processed and clean version of that dataframe
    
    Args:
        Pandas dataframe to be cleaned and prepared for analysis

    Returns:
        Pandas dataframe, a clean version of original pandas dataframe
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # extract a list of new column names for categories, omitting the number, and renaming
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames

    # iterate through the category columns to keep only the last character in the string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
   
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    df.head()

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # filtering the dataframe to only consider rows that are classified either 0 or 1
    df = df[df.iloc[:, 5:].isin([0, 1]).all(axis=1)]

    # drop duplicates
    df = df.drop_duplicates()
    df = df.loc[:,~df.columns.duplicated()]

    return df


def save_data(df, database_filename):
    '''
    This function intakes a dataframe and a db filename, writing the dataframe as a table in 
    the provided database.
    '''

    # create sqlite engine and save table to database for later usage
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categorized_messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()