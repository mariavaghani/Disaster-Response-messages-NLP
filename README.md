# Disaster Response Pipeline Project Summary
This repository contains the source code for the Disaster Response Messages project, run by NLP pipelines. This web app let's user to input a message and instantly get a possible category based on training data. The mean accuracy score (f1 score) for all categories is 0.686.

Possible next steps to increase the accuracy of the model:
- Create ItemSelector transformers that allow to pass "genre" of the message as a dummy variable
- Normalize the Character Length and Word Count features based on training set to range 0-1. Transform incoming messages into a normalized version before passing it through the model. That would reducing the risk of these two features overpowering the model.
- Alternatively, passing the weight of these features manually to play with accuracy of the model 

The original dataset came with 36 categories, it is important to mention that during data research phase of the project, it appeared that "child_alone" category had 0 instances; therefore, we could not train the model on this category, so it was dropped. This web app features only 35 categories for that reason. GridSearchCV was performed during the research phase of the project, and it showed the base model of Logistic Regression Classifier performed slightly better than alternitives.

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

# Installation
All the libraries required for the app to run are listed in the requirement.txt file

# Project Motivation
The platform helps essential services and first responders to quickly identify relevant messages at the time of crisis to assess the situation and deploy adequate forces where needed.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. The code could be deployed on Heroku as-is. In order to run it locally, remove the (#) comment to allow the main function to run
    ![image](https://user-images.githubusercontent.com/54246143/114328522-eb06c600-9b0a-11eb-9254-ecda302629b3.png)

3. Run the following command in the root directory of terminal to run your web app.
    `python disaster.py`

4. Go to http://0.0.0.0:5000/
   Alternatively, type http://localhost:5000/ in browser while 'disaster.py' runs in terminal
   
# Acknowledgements & Licensing
Thank you to *Figure 8* for providing the dataset, Udacity for providing initial file structure, page layouts, and helpful mentors.

All files included in this repository are free to use.

# Gallery
![image](https://user-images.githubusercontent.com/54246143/114330904-c44b8e00-9b10-11eb-96d9-250569d3a55d.png)

Word Cloud for "Request" category

![image](https://user-images.githubusercontent.com/54246143/114330960-e80ed400-9b10-11eb-82cf-e8fbc2f7e465.png)

Word Cloud for "Medical Aid" category

![image](https://user-images.githubusercontent.com/54246143/114331139-5ce20e00-9b11-11eb-93e4-4b032517dcc6.png)
Average message length in characters by category by genre

![image](https://user-images.githubusercontent.com/54246143/114340092-c3bcf280-9b24-11eb-8890-1348acff556f.png)

Front page of the app

![image](https://user-images.githubusercontent.com/54246143/114340135-dd5e3a00-9b24-11eb-936f-ec7ef7e55e45.png)

Classification output screen
