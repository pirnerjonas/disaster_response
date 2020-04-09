# Disaster Response Pipeline Project
Classification of disaster responses in a machine learning pipeline. The results are explored in a Flask web app.

![img](/images/app.jpeg)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or alternatively `set FLASK_APP=run.py` and `flask run` on windows

### Explanation of the files
In this section I will go through all the folders in the project and state a little explanation of all the files in the folder. There are three main folders in the project which are **app** (for the Flask web app), **data** (the ETL process of the pipeline) and **models** (the classification pipeline).

Explanation of the **app** folder:

| File        | Description                                                                               |
|-------------|-------------------------------------------------------------------------------------------|
| go.html     | structure for the site which gets rendered when the user tests the model with a own query |
| master.html | general structure for the web app                                                         |
| run.py      | the logic of the web app                                                                  |
| tokenizer.py      | tokenization method                                                                 |

Explanation of the **data** folder:

| File                    | Description                                |
|-------------------------|--------------------------------------------|
| disaster_categories.csv | the dataset with the categories            |
| disaster_messages.csv   | the dataset with the disaster messages     |
| process_data.py         | implements the ETL process of the pipeline |

Explanation of the **models** folder:

| File                    | Description                                |
|-------------------------|--------------------------------------------|
| train_classifier.py | implements the text classification pipeline    |
| tokenizer.py      | tokenization method                              |


### Results and interpretation
Since the dataset is imbalanced (an issue that I haven't particularly addressed in the modeling) the quality of the classifier differs per category. This can be inspected after training (`train_classifier.py`), for some classes not a single observation is detected. Most of the time this is due to the fact that there is little training data for those classes (the actual distribution of the labels can be seen in the web app). There are some ways to potentially improve the performance measures for these classes:
- **get more data:** if possible it would be best to collect more observations for the classes which are very small
- **resampling techniques:** here either the minority class gets artificially bigger (e.g oversampling) or the majority class gets downsized (e.g. undersampling).
- **additional features:** there are two additional features in the dataset that I haven't used by now. These are `original` and the categorical variable `genre`. Incorporating these features could improve the general predictive power of the model.
- **additional tuning:** I only tried a bunch of machine learning models before sticking with the ridge regression. Furthermore I only tried a few parameter settings to get to the final results. With more time and effort more settings can be tried out to boost the results.

### Future Improvements ###
Importing `tokenizer.py` two times is currently the only way to properly load the pickeled model (including the tokenization method) and use it in the web app without changing the structure of the project. I couldn't manage to load the package with a relative path without running into other errors. 