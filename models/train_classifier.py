import os
import sys
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_data(database_filepath):
    """loads the data from the database
    
    Arguments:
        database_filepath {path} -- path to the database
    
    Returns:
        X, Y, category_names -- returns the features, labels and category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterTable',con=engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """tokenization and lemmatization of raw text input
    
    Arguments:
        text {string} -- raw text
    
    Returns:
        list of tokens -- returns the tokenized and lemmatized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """builds a classification pipeline and performs gridsearch cross-validation.
    
    Returns:
        sklearn pipeline -- the pipeline with the corresponding tuning parameters
    """
    # define the tuning parameters
    params = {
        'tfidf__use_idf': [True, False]
    }
    # define the pipeline with the transformation steps
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('ridge', MultiOutputClassifier(RidgeClassifier()))
    ])
    # perform gridsearch and cross-validation
    gs_pipeline = GridSearchCV(pipeline, params, cv=2, n_jobs=-1)
    return gs_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """prints the classification report for every class in the dataset
    
    Arguments:
        model {sklearn pipeline} -- fitted pipeline
        X_test {dataframe} -- features of the test dataset
        Y_test {dataframe} -- labels of the test dataset
        category_names {list} -- category names
    """
    prediction_df = pd.DataFrame(model.predict(X_test), columns=category_names)
    for i, category in enumerate(category_names):
        pred_cat = prediction_df.iloc[:,i]
        true_cat = Y_test.iloc[:,i]
        print(f'Category: {category}')
        print(classification_report(pred_cat, true_cat))
        print('='*80)


def save_model(model, model_filepath):
    """saves the pipeline model
    
    Arguments:
        model {sklearn pipeline} -- fitted pipeline
        model_filepath {string} -- path to fitted pipeline
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
