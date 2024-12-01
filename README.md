# Disaster-Response-NLP-Pipeline
![Disaster Response Application](app/templates/screenshot.png)


## Project Summary
This repo contains a Udacity Data Science project that categorizes messages for disaster response purposes through the building, training and implementation of an end-to-end NLP (Natural Language Processing) pipeline along with a companion Flask application. There are several visualizations pre-loaded into the webpage for demonstration purposes.

## Prerequisites
In order to run this application, several installations must be complete upon executing the script "run.py":

1. Python 3.5 or greater is required in order to execute "python run.py" in the working directory
2. Library NLTK is required for natural language processing
3. Several libraries such as Pandas, Flask , Sci-Kit Learn, Plotly, Pickle and others must be pip installed accordingly when running each included Python script

## Intructions
After cloning this repository, our model .pickle file must first be generated by executing the following steps in order.

1. Upon meeting the above listed prerequisite libraries, execute "process_data.py" in terminal within the cloned directory.
```sh
python process_data.py disaster_messages.csv disaster_categories.csv
```
2. Once the data has been processed and cleaned through "process_data.py", we can build, train and save our model by executing "train_classifier.py".
```sh
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
3. Once our model has been trained and our model has been dumped into a .pickle file, we can execute "run.py" to launch and render our Flask web application.
```sh
pyhton run.py
```
4. Once the application has been launched, follow the address linked in the terminal and feel free to test out the classification app by typing in a message and clicking "Classify Message"!