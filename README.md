# Disaster Response Pipeline Project

## Contents

- [Motivation](#Motivation)
- [Used data and libraries](#Used_data_and_libraries)
- [ETL process](#ETL_process)
- [ML pipeline](#ML_pipeline)
- [Instructions](#Instructions)
- [Screenshots](#Screenshots)

### Motivation <a name="Motivation"> </a>

The main motivation of this project is to use messages from various sources to build classifier model that link it to distress under 36 categories of output. This can be used to connect various emergency services according to the classification of the messages. The data is provided by Figure Eight.

### Used data and libraries <a name="Used_data_and_libraries"> </a>

- Data:
    - disaster_messages.csv: This csv file contains messages that would potentially require aid.
    - disaster_categories.csv: This csv file labels for the messages in *disaster_messages.csv*
    
- Libraries used:
    - Pandas: To extract, merge and save data from two csv files
    - NLTK: Tokenizing, Vectorization and Lemmatization of messages done with NLTK functions
    - Scikit-learn: Scikit-learn is used to split the training and testing dataset, build pipeline and feature union, initializing multiclassifier models and performing hyperparameter optimization using grid search with cross-validation
    - Flask: Flask is used to create a simple UI to input message and visualize the classification
    - Plotly: This library is used to plot bar graphs on the home screen of GUI
    - re: Regex is used to identify parts of text and replace it with another text
    - sqlalchemy: This libray helps load and save database files
    
### ETL process <a name="ETL_process"> </a>

The extraction, transformation and loading of the dataset from csv files is done with help of modularized code, that takes in the csv file and saves the transformed and cleaned data in a .db file

    - process_data.py
        - Extract data from the csv files
        - Transforms the data by merging the messages and its catergories
        - Drop duplicates and corrects incorrect data values
        - Loads the cleaned data into a database using sql
        
### ML pipeline <a name="ML_pipeline"> </a>

Creates the ML classifier pipeline that reads the input data, initializes the classifier model including NLP transformations, optimizes the ML model and evaluates the model on all the categories.

    -train_classifier.py
        - Loades the data from .db file created by ETL process
        - Split training and testing data
        - Builds pipeline for the ML classifier model
        - Hyperparameter optimization using gridsearch CV
        - Evaluation of result using classification report from scikit-learn
        - saves the model in .pkl file
 
### Instructions <a name="Instructions"> </a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`. The web app should now be running if there were no errors.

3. Next, open another terminal and type
    `env|grep WORK`. You will see the **SPACE DOMAIN** and **SPACE ID**
    
4. In the new browser window, type in the **SPACE DOMAIN** and **SPACE ID** in the following format,
    https://`SPACEID`-3001.`SPACEDOMAIN`
    
    The web app should open now.

### Screenshots <a name="Screenshots"> </a>

Some screenshots from the run are also attached. The screenshots shows the UI interface and classfication results for a message. It also shows how to run process_data.py and train_classifier.py to get the data.db and classifier.pkl respectively.
