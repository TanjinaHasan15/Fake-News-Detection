#  Fake News Detection using Machine Learning

This project is a **Fake News Detection System** built with **Python, Scikit-learn, and Streamlit**.  
It predicts whether a given news article is **Fake** or **Real** using a trained Logistic Regression model.

---

##  Introduction

With the rise of social media and online platforms, the spread of **fake news** has become a major challenge.  
This project aims to address this issue by building a **machine learning model** that can classify news articles as **Fake** or **Real**.  

By combining **Natural Language Processing (NLP)** techniques and a **Logistic Regression model**, the system provides a quick and easy way to verify the authenticity of news articles.

---

##  Project Overview

- The project uses **NLP techniques** to process news text.  
- The dataset consists of two parts: **Fake news** and **True news**.  
- A **vectorizer** (TF-IDF / CountVectorizer) converts text into numerical form.  
- A **Logistic Regression** model is trained on the dataset.  
- The trained model is deployed with **Streamlit** so users can interact through a simple web interface.  
- The user inputs a news article → The app predicts whether it is **Fake**  or **Real** .  

---

## Dataset

We used the **Fake and Real News Dataset** from Kaggle:  

- **Fake.csv** → Collection of fake news articles.  
- **True.csv** → Collection of genuine news articles.  

This dataset provides a balanced set of real and fake news, making it suitable for classification tasks.

---

##  Dependencies

Make sure you have the following libraries installed:

```bash
streamlit
scikit-learn
pandas
joblib

---

##  Technologies Used

###  Python
- The main programming language used to develop the project.

###  Pandas → Data handling
- Used for loading, cleaning, and managing the dataset (`Fake.csv`, `True.csv`).

###  Scikit-learn → Machine Learning (Logistic Regression, TF-IDF)
- **TF-IDF Vectorizer**: Converts text into numerical features.
- **Logistic Regression**: Classifier used to predict whether a news article is Fake or Real.

### Joblib → Model persistence
- Saves and loads the trained model (`lr_model.joblib`) and vectorizer (`vectorizer.joblib`).
- Prevents retraining the model every time the app is run.

###  Streamlit → Frontend web interface
- Provides an interactive user interface where:
  - Users can input a news article.
  - Click a button to check the result.
  - Instantly see whether the news is **Fake ** or **Real **.

---

## Results

- The Logistic Regression model achieved **high accuracy** in distinguishing Fake vs Real news (accuracy score depends on training, typically ~90%+ on this dataset).
- The model was tested on multiple examples and performed well.

---

###  Example Predictions:

```python
example_real = "Government announces new policy to improve education system."
example_fake = "Celebrity claims drinking hot water cures all diseases."

print("Real Example →", model.predict(vectorizer.transform([example_real]))[0])
print("Fake Example →", model.predict(vectorizer.transform([example_fake]))[0])

