import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet'])

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def load_data(database_filepath):
    '''
    Loads database from .db file into dataframe
    and splits into data and labels

    Args:
    database_filepath: .db file path to be imported

    Return:
    X: Messages
    y: Categories
    '''
    # load data from database
    engine = create_engine('sqlite:///ETL_pipeline.db')
    df = pd.read_sql_table('ETL_pipeline', engine)

    X = df.message
    y = df.iloc[:,4:]

    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    '''
    Replace url, tokenize and lemmatize text
    
    Args:
    text: Input text
    
    Return:
    clean_tokens: Tokenized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(parameter = {
    'clf__estimator__n_estimators': [50,200],
    'clf__estimator__min_samples_split': [2,4]},
     clf = RandomForestClassifier()):
    '''
    Creates classifier model specified in the Argument

    Args:
    clf: classifier model, Random forest as default
    params: Dict of search space for classifier model

    Return:
    cv: Grid search model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))
        ])
    
    parameters = parameter

    cv = GridSearchCV(
        pipeline, param_grid=parameters,
        n_jobs=-1, verbose=2)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model on test data and returns the score

    Args:
    model: The ML classifier model
    X_test: Test input features
    Y_test: Test labels
    category_names: Output category of the dataset

    Return:
    Prints precision, recall and F1-score for each output category
    '''
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values,
     y_pred, target_names=category_names.values))

def save_model(model, model_filepath):
    '''
    Saves the ML model as a pickle file

    Args:
    model: ML model to be saved
    model_filepath: Save path of the model

    Return:
    None
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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