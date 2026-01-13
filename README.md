# Movie Review Sentiment Analysis using Custom K-Nearest Neighbors (KNN)

This project implements sentiment analysis on movie reviews using a custom-built K-Nearest Neighbors (KNN) classifier. The goal is to classify reviews as positive or negative using Natural Language Processing (NLP) and machine learning techniques.

## Objective
- Convert raw movie review text into numerical features
- Implement a custom KNN classifier without using predefined KNN libraries
- Compare cosine similarity and Euclidean distance metrics
- Improve efficiency using dimensionality reduction
- Evaluate performance using cross-validation

## Dataset
- 25,000 labeled movie reviews
- Classes: Positive (+1), Negative (-1)
- Dataset split into training and testing sets  
(Dataset files are not included due to size and academic restrictions.)

## Data Preprocessing
- Text normalization (lowercasing, punctuation and special character removal)
- Stopword removal using NLTK
- Stemming using Porter Stemmer
- HTML content removal using BeautifulSoup
- Tokenization and cleaning of raw text

## Feature Extraction
- TF-IDF vectorization with n-gram range (1–3)
- Minimum document frequency: 5
- Maximum document frequency: 0.5

## Dimensionality Reduction
- Applied Truncated Singular Value Decomposition (SVD)
- Reduced TF-IDF features to 50 components to improve runtime and reduce overfitting

## Custom KNN Classifier
- Implemented from scratch
- Supported distance metrics:
  - Cosine Similarity
  - Euclidean Distance
- Predictions based on majority voting among k nearest neighbors

## Model Evaluation
- Used Stratified K-Fold Cross-Validation
- Tested multiple k values
- Compared distance metrics for performance

| k Value | Euclidean Accuracy | Cosine Accuracy |
|-------|-------------------|----------------|
| 3  | 0.76 | 0.78 |
| 5  | 0.78 | 0.79 |
| 7  | 0.79 | 0.80 |
| 10 | 0.80 | 0.81 |
| 15 | 0.81 | 0.82 |
| 20 | 0.81 | 0.82 |

**Best accuracy achieved: 82% (Cosine Similarity)**

## Performance Optimization
- TF-IDF vectorization for efficient feature generation
- SVD for dimensionality reduction
- Optimized NLP preprocessing using NLTK
- Preprocessing time: ~3–4 minutes
- Training and evaluation time: ~7–10 minutes

## Output
- Final sentiment predictions generated on the test dataset
- Results saved to an output file

## Technologies Used
- Python
- NumPy
- scikit-learn
- NLTK
- BeautifulSoup
- TF-IDF
- SVD
- Custom KNN implementation

## Conclusion
This project demonstrates an end-to-end sentiment analysis pipeline using a custom KNN classifier. It highlights practical experience in NLP, machine learning, algorithm design, and performance optimization on large-scale text data.
