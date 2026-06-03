import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

#DATA CLEANING  

def load_data():
    books   = pd.read_csv('Books.csv',   encoding='latin-1', on_bad_lines='skip')
    ratings = pd.read_csv('Ratings.csv', encoding='latin-1', on_bad_lines='skip')

    # Standardize column names
    books.columns   = books.columns.str.strip().str.upper().str.replace('-', '_').str.replace(' ', '_')
    ratings.columns = ratings.columns.str.strip().str.upper().str.replace('-', '_').str.replace(' ', '_')

    # Keep only explicit ratings (1–10), drop 0 = implicit
    ratings = ratings[ratings['BOOK_RATING'] > 0]

    # Keep only books with 5+ ratings (noise reduction)
    popular = ratings['ISBN'].value_counts()
    popular = popular[popular >= 5].index
    ratings = ratings[ratings['ISBN'].isin(popular)]

    # Keep only users with 10+ ratings (active users)
    active_users = ratings['USER_ID'].value_counts()
    active_users = active_users[active_users >= 10].index
    ratings = ratings[ratings['USER_ID'].isin(active_users)]

    # Merge with book metadata
    df = ratings.merge(books[['ISBN', 'BOOK_TITLE', 'BOOK_AUTHOR', 'YEAR_OF_PUBLICATION', 'PUBLISHER']], on='ISBN', how='left')
    df.dropna(subset=['BOOK_TITLE'], inplace=True)

    print(f"✅ Loaded {len(df):,} ratings | {df['ISBN'].nunique():,} books | {df['USER_ID'].nunique():,} users")
    return books, ratings, df


#tf idf
class ContentBasedRecommender:
    def __init__(self, books_df):
        self.books = books_df.drop_duplicates(subset='ISBN').reset_index(drop=True)
        self.books.columns = self.books.columns.str.strip().str.upper().str.replace('-', '_').str.replace(' ', '_')

        # Build feature string from title + author + publisher
        self.books['FEATURES'] = (
            self.books.get('BOOK_TITLE', '').fillna('') + ' ' +
            self.books.get('BOOK_AUTHOR', '').fillna('') + ' ' +
            self.books.get('PUBLISHER', '').fillna('')
        )

        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.books['FEATURES'])
        print("✅ Content-based model ready")

    def recommend(self, book_title, n=5):
        # Find closest matching title (case-insensitive)
        titles = self.books['BOOK_TITLE'].str.lower()
        matches = titles[titles.str.contains(book_title.lower(), na=False)]

        if matches.empty:
            return pd.DataFrame(), f"No book found matching '{book_title}'"

        idx = matches.index[0]
        matched_title = self.books.loc[idx, 'BOOK_TITLE']

        # Compute cosine similarity
        query_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1][1:n+1]  # exclude itself

        results = self.books.iloc[sim_indices][['BOOK_TITLE', 'BOOK_AUTHOR']].copy()
        results['SIMILARITY'] = sim_scores[sim_indices].round(3)
        results.reset_index(drop=True, inplace=True)

        return results, matched_title


#svd

class CollaborativeRecommender:
    def __init__(self, df):
        self.df = df

        # Build user-book pivot matrix
        self.pivot = df.pivot_table(
            index='USER_ID',
            columns='BOOK_TITLE',
            values='BOOK_RATING'
        ).fillna(0)

        # Mean-center ratings per user (removes bias)
        self.user_ratings_mean = self.pivot.mean(axis=1)
        self.pivot_norm = self.pivot.sub(self.user_ratings_mean, axis=0)

        # SVD decomposition
        sparse_matrix = csr_matrix(self.pivot_norm.values)
        k = max(1, min(50, min(sparse_matrix.shape) - 1))
        if k < 1:
            raise ValueError(f"Matrix too small for SVD: shape {sparse_matrix.shape}. Need more users/books.")
        U, sigma, Vt = svds(sparse_matrix, k=k)
        sigma = np.diag(sigma)

        # Reconstruct full ratings matrix
        predicted = np.dot(np.dot(U, sigma), Vt)
        self.predicted_ratings = pd.DataFrame(
            predicted + self.user_ratings_mean.values.reshape(-1, 1),
            index=self.pivot.index,
            columns=self.pivot.columns
        )
        print("✅ Collaborative filtering model ready")

    def recommend(self, user_id, n=5):
        if user_id not in self.predicted_ratings.index:
            return pd.DataFrame(), "User not found"

        # Get predicted ratings for this user
        user_pred = self.predicted_ratings.loc[user_id]

        # Exclude books already rated by this user
        already_rated = self.pivot.loc[user_id]
        already_rated = already_rated[already_rated > 0].index

        user_pred = user_pred.drop(index=already_rated, errors='ignore')
        top_books = user_pred.nlargest(n).reset_index()
        top_books.columns = ['BOOK_TITLE', 'PREDICTED_RATING']
        top_books['PREDICTED_RATING'] = top_books['PREDICTED_RATING'].round(2)

        return top_books, user_id

    def get_sample_users(self, n=10):
        return list(self.predicted_ratings.index[:n])




def get_popular_books(df, n=10):
    """Fallback for cold-start: return highest avg rated books with 50+ ratings"""
    popular = (
        df.groupby('BOOK_TITLE')['BOOK_RATING']
        .agg(['mean', 'count'])
        .reset_index()
    )
    popular.columns = ['BOOK_TITLE', 'AVG_RATING', 'NUM_RATINGS']
    popular = popular[popular['NUM_RATINGS'] >= 50]
    popular = popular.sort_values('AVG_RATING', ascending=False).head(n)
    popular['AVG_RATING'] = popular['AVG_RATING'].round(2)
    return popular.reset_index(drop=True)



if __name__ == '__main__':
    books, ratings, df = load_data()

    print("\n── Content-Based Test ──")
    cb = ContentBasedRecommender(books)
    results, title = cb.recommend("Harry Potter")
    print(f"Because you liked: {title}")
    print(results.to_string(index=False))

    print("\n── Collaborative Filtering Test ──")
    cf = CollaborativeRecommender(df)
    sample_user = cf.get_sample_users(1)[0]
    recs, uid = cf.recommend(sample_user)
    print(f"Recommendations for user {uid}:")
    print(recs.to_string(index=False))

    print("\n── Popular Books (Cold-Start) ──")
    print(get_popular_books(df).to_string(index=False))