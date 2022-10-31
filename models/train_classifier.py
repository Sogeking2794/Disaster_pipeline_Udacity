import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import joblib

# Class for VerbExtractor
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

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
    engine = create_engine('sqlite:///' + database_filepath)
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
    'clf__estimator__n_estimators': [10, 30],
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
    ("feature", FeatureUnion([
        ('token_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())
    ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    

    cv = GridSearchCV(
        pipeline, param_grid=parameter,
        n_jobs=-1, verbose=3)
    
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
        joblib.dump(model.best_estimator_, f, compress=1)


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