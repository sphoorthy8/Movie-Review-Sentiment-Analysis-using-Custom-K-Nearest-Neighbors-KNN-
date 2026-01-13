import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Uncomment once if needed
# nltk.download('stopwords')

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()     # remove HTML
    text = text.lower()
    text = re.sub(r'@\w+|#\w+', ' ', text)                   # mentions/hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)                    # punctuation/numbers
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Custom KNN Classifier
# -----------------------------
class CustomKNN:
    def __init__(self, k=5, metric='cosine'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        if self.metric == 'cosine':
            sims = cosine_similarity(X, self.X_train)
            for i in range(sims.shape[0]):
                idx = np.argsort(sims[i])[-self.k:]
                labels = self.y_train[idx]
                preds.append(1 if np.sum(labels) >= 0 else -1)
        else:  # Euclidean
            for x in X:
                dists = np.linalg.norm(self.X_train - x, axis=1)
                idx = np.argsort(dists)[:self.k]
                labels = self.y_train[idx]
                preds.append(1 if np.sum(labels) >= 0 else -1)
        return np.array(preds)

# -----------------------------
# Load Data (example format)
# Each line: <label>\t<review text>
# -----------------------------
def load_data(path):
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t', 1)
            labels.append(int(label))
            texts.append(clean_text(text))
    return texts, np.array(labels)

# -----------------------------
# Main Pipeline
# -----------------------------
def main(train_path, test_path):
    X_text, y = load_data(train_path)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.5
    )
    X_tfidf = vectorizer.fit_transform(X_text)

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    X_reduced = normalize(X_reduced)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_k, best_acc = 0, 0
    for k in [3, 5, 7, 10, 15, 20]:
        accs = []
        for train_idx, val_idx in skf.split(X_reduced, y):
            knn = CustomKNN(k=k, metric='cosine')
            knn.fit(X_reduced[train_idx], y[train_idx])
            preds = knn.predict(X_reduced[val_idx])
            accs.append(np.mean(preds == y[val_idx]))
        mean_acc = np.mean(accs)
        print(f"k={k}, accuracy={mean_acc:.3f}")
        if mean_acc > best_acc:
            best_acc, best_k = mean_acc, k

    print(f"\nBest k: {best_k}, Best Accuracy: {best_acc:.3f}")

    # Train final model
    knn = CustomKNN(k=best_k, metric='cosine')
    knn.fit(X_reduced, y)

    # Predict test data
    X_test_text, _ = load_data(test_path)
    X_test_tfidf = vectorizer.transform(X_test_text)
    X_test_reduced = svd.transform(X_test_tfidf)
    X_test_reduced = normalize(X_test_reduced)

    test_preds = knn.predict(X_test_reduced)
    np.savetxt("predicted_sentiments.txt", test_preds, fmt="%d")
    print("Predictions saved to predicted_sentiments.txt")

if __name__ == "__main__":
    # Update paths before running
    main("train_data.txt", "test_data.txt")
