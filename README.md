# 🎬 Movie Recommender System

This project is a content-based Movie Recommender System built using Python. It recommends movies based on similarity in content (like genres, overview, cast, etc.) and helps users discover films similar to their favorites.

---

## 📌 Project Overview

The main objective is to recommend movies that are similar to a given movie based on various content features using natural language processing and similarity metrics.

### ✅ Key Features

- Recommends movies based on content similarity
- Uses **TF-IDF** and **cosine similarity** for matching
- Text preprocessing to clean and structure data
- Interactive user input for real-time recommendations
- Efficient implementation using vectorization

---

## 📂 Dataset

- Source: [TMDB Movie Dataset]
- The dataset includes fields like:
  - Title
  - Overview
  - Genres
  - Cast
  - Keywords
  - Crew

---

## 🧰 Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK / spaCy**
- **Jupyter Notebook**

---

## 🧠 How It Works

1. **Preprocessing**:
   - Handle missing values
   - Combine important features (overview, genres, cast, etc.)
   - Apply tokenization, stemming, and lowercase conversion

2. **Feature Extraction**:
   - Convert movie content into vectors using **TF-IDF Vectorizer**

3. **Similarity Calculation**:
   - Use **cosine similarity** to find the most similar movies

4. **Recommendation Engine**:
   - When a user inputs a movie title, it outputs the top similar movies

