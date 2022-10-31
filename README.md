# Disaster Response Pipeline Project

## Contents
- [Motivation](#Motivation)
- [Used data and libraries](#Used data and libraries)
- [ETL process](#ETL process)
- [ML pipeline](#ML pipeline)
- [Instructions](#Instructions)

### Motivation <a name="Motivation"> </a>

The main motivation of this project is to use messages from various sources to build classifier model that link it to distress under 36 categories of output. This can be used to connect various emergency services according to the classification of the messages. The data is provided by Figure Eight.

### Used data and libraries <a name="Used data and libraries"> </a>

- Data:
    - disaster_messages.csv: This csv file contains messages that would potentially require aid.
    - disaster_categories.csv: This csv file labels for the messages in *disaster_messages.csv*
    
- Libraries used:
    - Pandas: To extract, merge and save data from two csv files
    - 
    -
    -
    -
    -
    
### ETL process <a name="ETL process"> </a>

### ML pipeline <a name="ML pipeline"> </a>
 
### Instructions <a name="Instructions"> </a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
