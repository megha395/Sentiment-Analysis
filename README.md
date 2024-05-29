# Sentiment-Analysis

This repository contains a sentiment analysis project using a Logistic Regression model on Bag of Words (BoW) and TF-IDF features. The project is organized into several scripts and Jupyter notebooks, each handling different aspects of the data processing, feature extraction, model training, and evaluation.

## Repository Structure
1. *global_imports.py*: Contains all the necessary library imports used across different scripts and notebooks.

2. *analysis.ipynb*: Jupyter notebook for exploratory data analysis (EDA) on the dataset.

3. *data_preprocessing.py*: Handles the data preprocessing steps including data cleaning, tokenization, and splitting the dataset into training and testing sets.

4. *feature_extraction.py*: Script to extract features using Bag of Words (BoW) and TF-IDF methods.

5. *model.py*: Contains the implementation of the Logistic Regression model training.

6. *evaluation.ipynb*: for evaluating the model performance including prediction, accuracy, classification report, confusion matrix, and visualizing results using word clouds.

### Setup and Installation

1. Clone the repository:

*git clone https://github.com/yourusername/SENTIMENT-ANALYSIS.git*
*cd SENTIMENT-ANALYSIS*


2. Install the required packages:

 -    - pip install -r requirements.txt

### Running the Project
1. Data Preprocessing:
Run the data_preprocessing.py script to preprocess the data.
 -    - *python data_preprocessing.py*

2. Feature Extraction:
Run the feature_extraction.py script to extract features using BoW and TF-IDF.
 -    - *python feature_extraction.py*

3. Model Training:
Run the model.py script to train the Logistic Regression model.
 -    - *python model.py*

4. Evaluation:
Open the evaluation.ipynb notebook to evaluate the model's performance. This notebook includes:

 -      - Predictions on the test data
 -      - Model accuracy
 -      - Classification report (precision, recall, f1-score, support)
 -      - Confusion matrix
 -      - Word cloud visualization of the most frequent words

# Project Workflow
## Exploratory Data Analysis (EDA):

Conduct EDA in analysis.ipynb to understand the dataset.
1. **Data Preprocessing:**

Preprocess the raw data in data_preprocessing.py which includes cleaning, tokenization, and splitting into train/test sets.

2. **Feature Extraction:**

Extract features from the text data using BoW and TF-IDF in feature_extraction.py.

3. **Model Training:**

Train the Logistic Regression model using the extracted features in model.py.

4. **Evaluation:**

Evaluate the model's performance using various metrics and visualize results in evaluation.ipynb.

# Dependencies
The project relies on the following libraries:

    -    numpy
    -    pandas
    -    scikit-learn
    -    matplotlib
    -    seaborn
    -    wordcloud
    -    jupyter
Make sure to install all dependencies listed in requirements.txt.

# Contributing
Contributions are welcome! If you have any suggestions, bug fixes, or improvements, please submit a pull request or open an issue.

# Acknowledgements
Inspired by various tutorials and courses on sentiment analysis and natural language processing.















