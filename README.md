# Fake News Detector - ML & NLP

A web application built with **Streamlit** that uses Machine Learning and Natural Language Processing techniques to detect fake news. The model is trained using popular Python libraries and deployed online for easy access.

**Live Demo**: [https://fake-news-detector-ml-nlp.streamlit.app/](https://fake-news-detector-ml-nlp.streamlit.app/)

---

## Features

- Input any news text and get instant prediction: Real or Fake
- NLP preprocessing pipeline with `nltk`
- Machine learning classification using `scikit-learn`
- Visualizations using `matplotlib` and `seaborn`

---

## Prerequisites

Make sure you have **Python 3.7+** installed. Then, install the required packages using pip:

```bash
pip install streamlit pandas numpy nltk scikit-learn matplotlib seaborn
```

## 💻 How to Run Locally

**Clone the repository:**

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

## Project Structure
```bash
fake-news-detector/
│
├── streamlit/
    ├── logreg_model.pkl                  # Trained model file
    ├── requirements.txt                  # List of dependencies
    ├── streamlit_app.py                  # Main Streamlit app
    ├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
├── LICENSE                               # License
├── README.md                             # Project documentation                 
├── model_training.ipynb                  # jupyter notebook containing model training and evaluation
└── news.csv                              # Sample dataset
```

## License
This project is licensed under the MIT License.

## Acknowledgements
Built with Streamlit

NLP tools from NLTK

ML using Scikit-learn

Feel free to fork, contribute, or give feedback!
