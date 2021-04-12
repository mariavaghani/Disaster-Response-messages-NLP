# Disaster Response Pipeline Project
This repository contains the source code for the Disaster Response Messages project, run by NLP pipelines. This web app let's user to input a message and instantly get a possible category based on training data. The mean accuracy score (f1 score) for all categories is 0.7.
The app is deployed on Heroku [here](https://disaster-response-app-mv.herokuapp.com/)

# File Structure
(applic) folder was renamed from (app) to better understand file structure and imports

.<br />
├── _applic >>>folder containing app files<br />
│   ├── _templates<br />
│   │  ├─ go.html >>>page rendering the classifier output, offering a form for another message to be classified<br />
│   │  └─ master.html >>> initial page, showing some visuals and form to type in a message to be classified<br />
│   ├── __init__.py >>> file to import message_length_estimator as a module<br />
│   ├── message_length_estimator.py >>> estimator object for the pickle file to reference<br />
│   └── run.py >>> back - end of the app<br />
├── _data<br />
│   ├── DisasterResponse.db<br />
│   ├── disaster_categories.csv<br />
│   ├── disaster_messages.csv<br />
│   └── process_data.py >>> ETL pipeline - python script that generates DisasterResponse database from csv files<br />
├── _models<br />
│   ├── __init__.py >>>> file to import message_length_estimator as a module<br />
│   ├── classifier.pkl<br />
│   ├── message_length_estimator.py<br />
│   └── train_classifier.py >>> ML pipeline (NLP) - python script that generates the pickled model using<br />
│                               DisasterResponse database, using message_length_estimator<br />
├── Procfile >>> file that instructs Heroku what to do first<br />
├── README.md<br />
├── message_length_estimator.py >>> file to import message_length_estimator as a module, to unpack pickle file using the app<br />
├── nltk.txt >>> helping Heroku to download necessary nltk modules<br />
├── runtime.txt >>> helping Heroku to choose version of Python<br />
└── requirement.txt >>> helping Heroku with downloading packages<br />

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
