# 📚 BookMatch — Hybrid Book Recommendation Engine

A hybrid recommendation system built on the **Book-Crossing dataset** (270K books, 1M+ ratings).
Combines content-based filtering and collaborative filtering to solve both the similarity and personalisation problems.

**Live Demo:** https://book-recommendation-system-palash.streamlit.app/

---

## Models

### 1. Content-Based Filtering (TF-IDF + Cosine Similarity)
- Vectorizes book metadata (title, author, publisher) using TF-IDF (5000 features)
- Computes cosine similarity between item vectors to find similar books
- Handles the **item cold-start problem** — works even with no user history

### 2. Collaborative Filtering (SVD Matrix Factorization)
- Builds a user-book ratings matrix (mean-centered to remove user bias)
- Applies **Truncated SVD** (k=50 latent factors) to decompose the sparse matrix
- Reconstructs predicted ratings for all user-book pairs
- Recommends top-N unread books per user based on predicted rating score

### 3. Popularity Baseline (Cold-Start Fallback)
- Returns highest average-rated books with 50+ reviews
- Used when a new user has no rating history

---

## Results

| Model | Metric | Score |
|---|---|---|
| Content-Based | Avg cosine similarity (top-5) | ~0.72 |
| Collaborative (SVD) | RMSE on filtered ratings | ~0.91 |
| Coverage | Books with recommendations | 100% |

---

## Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data loading and matrix operations
- **Scikit-learn** — TF-IDF vectorization, cosine similarity
- **SciPy** — sparse SVD (svds)
- **Streamlit** — interactive web UI

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/book-recommendation-system
cd book-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Go to https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
# Place Books.csv, Ratings.csv, Users.csv in the project root

# Run the app
streamlit run app.py
```

---

## Project Structure

```
book-recommendation-system/
├── recommender.py     # Core model logic (content-based + collaborative)
├── app.py             # Streamlit web UI
├── requirements.txt   # Dependencies
├── README.md
└── data/
    ├── Books.csv
    ├── Ratings.csv
    └── Users.csv
```

---

## Key Concepts Demonstrated

- **TF-IDF vectorization** for text-based item similarity
- **Matrix factorization via SVD** for latent-factor collaborative filtering
- **Cold-start problem handling** via content-based + popularity fallback
- **Mean-centering** to normalize user rating biases
- **Sparse matrix operations** for memory-efficient computation on large datasets
- **End-to-end ML pipeline** from raw data to deployed web app
