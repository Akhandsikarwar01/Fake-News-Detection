# 📰 Fake News Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete Machine Learning project that detects fake news articles using Natural Language Processing (NLP) techniques. Built with Python, Scikit-learn, NLTK, and Streamlit.

---

## 🔍 Features

- Text preprocessing pipeline (cleaning, tokenization, stemming, TF-IDF)
- Multiple ML classifiers: Logistic Regression, Decision Tree, Random Forest, Passive Aggressive
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Command-line interface for training, evaluation, and prediction
- Interactive Streamlit web application for live predictions
- Exploratory Data Analysis (EDA) notebook

---

## 🛠️ Tech Stack

| Category        | Tools                                |
|-----------------|--------------------------------------|
| Language        | Python 3.8+                          |
| ML Libraries    | Scikit-learn, NLTK                   |
| Data            | Pandas, NumPy                        |
| Visualization   | Matplotlib, Seaborn                  |
| Web App         | Streamlit                            |
| Model Storage   | Joblib                               |

---

## 📁 Project Structure

```
Fake-News-Detection/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Python .gitignore
├── data/
│   └── README.md              # Dataset download instructions
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Text preprocessing & TF-IDF
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation
│   ├── predict.py             # Prediction on new text
│   └── utils.py               # Utility/helper functions
├── app/
│   └── app.py                 # Streamlit web application
├── models/                    # Saved models (git-ignored)
└── tests/
    ├── __init__.py
    └── test_preprocess.py     # Unit tests
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Akhandsikarwar01/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 📊 Dataset

This project uses the [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data).

**Columns:**
| Column   | Description                        |
|----------|------------------------------------|
| `id`     | Unique article identifier          |
| `title`  | Article headline                   |
| `author` | Article author                     |
| `text`   | Full article body                  |
| `label`  | `1` = Fake News, `0` = Real News   |

See `data/README.md` for download instructions.

---

## 🚀 Usage

### Step 1 — Preprocess the Data

```bash
python src/preprocess.py --data data/train.csv
```

### Step 2 — Train a Model

```bash
# Train all models
python src/train.py --data data/train.csv --model all

# Train a specific model
python src/train.py --data data/train.csv --model logistic_regression
python src/train.py --data data/train.csv --model random_forest
python src/train.py --data data/train.csv --model decision_tree
python src/train.py --data data/train.csv --model passive_aggressive
```

### Step 3 — Evaluate a Model

```bash
python src/evaluate.py --data data/train.csv --model models/logistic_regression.joblib
```

### Step 4 — Predict on New Text

```bash
python src/predict.py --text "Breaking: Scientists discover new planet" --model models/logistic_regression.joblib
```

### Step 5 — Launch the Web App

```bash
streamlit run app/app.py
```

---

## 📈 Results / Performance Metrics

The following table shows typical performance metrics on the test set:

| Model                     | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression        | ~98%     | ~98%      | ~98%   | ~98%     |
| Passive Aggressive         | ~97%     | ~97%      | ~97%   | ~97%     |
| Random Forest              | ~96%     | ~96%      | ~96%   | ~96%     |
| Decision Tree              | ~93%     | ~93%      | ~93%   | ~93%     |

> Results may vary depending on data preprocessing and hyperparameter choices.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes docstrings.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Akhand Sikarwar**  
GitHub: [@Akhandsikarwar01](https://github.com/Akhandsikarwar01)