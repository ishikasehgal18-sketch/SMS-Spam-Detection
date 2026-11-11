
# SMS Spam Detection Using Machine Learning

## Project Overview
This project is a Machine Learning based SMS Spam Detection System that classifies text messages into two categories:
- Spam
- Ham (Not Spam)

It uses Natural Language Processing and the Naive Bayes algorithm to identify unwanted messages.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit

## Dataset
The project uses the SMS Spam Collection dataset containing 5,572 messages:
- Ham: 4,825
- Spam: 747

## Machine Learning Model
- Feature Extraction: TF-IDF
- Classification Algorithm: Multinomial Naive Bayes
- Train-Test Split: 80:20
- Accuracy: 96.05%

## Project Files
- app.py: Streamlit web application
- train_model.py: Model training code
- SMSSpamCollection.txt: Dataset
- spam_model.pkl: Saved trained model
- requirements.txt: Required Python libraries

## How to Run

1. Install Python.
2. Install dependencies:

   pip install -r requirements.txt

3. Train the model:

   python train_model.py

4. Run the application:

   python -m streamlit run app.py

## Output
The application predicts whether an entered SMS message is Spam or Not Spam.

## Author
Ishika Sehgal
