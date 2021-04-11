# Disaster Response Pipeline Project
This repository contains the source code for the Disaster Response Messages project, run by NLP pipelines. This web app let's user to input a message and instantly get a possible category based on training data. The mean accuracy score (f1 score) for all categories is 0.7.
The app is deployed on Heroku [here](https://disaster-response-app-mv.herokuapp.com/)

# File Structure
(applic) folder was renamed from (app) to better understand file structure and imports

.
├── _applic(folder containing app files)
│   ├── _templates
│   │  ├─ go.html >>>page rendering the classifier output, offering a form for another message to be classified
│   │  └─ master.html >>> initial page, showing some visuals and form to type in a message to be classified
│   ├── __init__.py >>> file to import message_length_estimator as a module
│   ├── message_length_estimator.py >>> estimator object for the pickle file to reference
│   └── run.py >>> back - end of the app
├── _data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py >>> ETL pipeline - python script that generates DisasterResponse database from csv files
├── _models
│   ├── __init__.py >>>> file to import message_length_estimator as a module
│   ├── classifier.pkl
│   ├── message_length_estimator.py
│   └── train_classifier.py >>> ML pipeline (NLP) - python script that generates the pickled model using 
│                               DisasterResponse database, using message_length_estimator
├── Procfile >>> file that instructs Heroku what to do first
├── README.md
├── message_length_estimator.py >>> file to import message_length_estimator as a module, to unpack pickle file using the app
├── nltk.txt >>> helping Heroku to download necessary nltk modules
├── runtime.txt >>> helping Heroku to choose version of Python
└── requirement.txt >>> helping Heroku with downloading packages

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the root directory of terminal to run your web app.
    `python disaster.py`

3. Go to http://0.0.0.0:5000/
   Alternatively, type http://localhost:5000/ in browser while 'disaster.py' runs in terminal
