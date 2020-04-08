import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """loads and merges the message and categories file
    
    Arguments:
        messages_filepath {path} -- path to message file
        categories_filepath {path} -- path to categories file
    
    Returns:
        dataframe -- returns the merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    print('Loaded and merged messages and categories')
    return df


def clean_data(df):
    """splits the categories into separate columns and removes duplicates
    
    Arguments:
        df {dataframe} -- merged messages and categories
    
    Returns:
        dataframe -- returns expanded and duplicate free dataframe
    """
    # split the values in the category column
    df_expanded = df['categories'].str.split(';',expand=True)
    # extract the category names (replace the '-' and the number)
    col_names = df_expanded.iloc[0,:].str.replace('-\d','')
    # replace all non-digits with empty string to just get the numbers
    df_expanded.replace('\D','', regex=True, inplace=True)
    # to numeric
    df_expanded = df_expanded.apply(pd.to_numeric)
    # set the column names
    df_expanded.columns = col_names
    # set the id 
    df_expanded['id'] = df['id']
    # merge both dataframes and get rid of old category
    df_expanded = pd.merge(df_expanded, df)
    df_expanded.drop('categories', axis=1, inplace=True)
    # drop duplicates
    num_duplicates = len(df_expanded[df_expanded.duplicated()])
    df_expanded.drop_duplicates(inplace=True)
    print(f'Cleaned data, removed {num_duplicates} duplicates')
    return df_expanded

def save_data(df, database_filename):
    """saves dataframe to sql table
    
    Arguments:
        df {dataframe} -- the cleaned dataframe
        database_filename {string} -- name of the database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(f'DisasterTable', engine, index=False)
    print(f'Database {database_filename} was created')


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