import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download("omw-1.4")

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
import pickle
import numpy as np





def load_data(database_filepath):
    '''
    Load the data from database (.db file)

    Parameters:
    database_filepath (str) : The database file path

    Returns:
    X (list) : The X data of model
    Y (list) : The Y data of model
    category_names (list): The names of category in data.
    '''
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("Select * from disaster_data",con=conn)
    category_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    return df.iloc[:,1].values, df.iloc[:,3:].values.astype(np.int), category_names


def tokenize(text):
    '''
    Tokenization the text

    Parameters:
    text (str): The input message text

    Returns: 
    list: The token of message text.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build the pipeline model 
    
    Returns:
    GridSearchCV: the grid search method for choosing the optimized parameter for the pipeline model.
    '''
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'clf__n_estimators' : [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
     '''
     Print out the report (recall, precision, f1_score,...) of all category

     Parameters:
     model (GridSearchCV): input model
     X_test (list): The list of message that used to evaluate
     Y_test (list): The list of expected output of message
     category_names (list): the list of category names
     '''
     y_pred = model.predict(X_test)
     print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save model to the file pickle (.pkl)

    Parameters:
    model (GridSearchCV): input model
    model_filepath (str): the output path of file model pickle
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("XTest shape",X_test.shape)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.tolist(), Y_train.tolist())
        # # Load model:
        # model = joblib.load("classifier.pkl")
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()