import streamlit as st
import pandas as pd
from recommender import load_data, ContentBasedRecommender, CollaborativeRecommender, get_popular_books

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="BookMatch — Book Recommendation Engine",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .rec-card {
        background: #fff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 8px;
    }
    .badge {
        background: #e7f3ff;
        color: #0066cc;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LOAD MODELS (cached)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    books, ratings, df = load_data()
    cb_model = ContentBasedRecommender(books)
    cf_model = CollaborativeRecommender(df)
    return books, ratings, df, cb_model, cf_model

books, ratings, df, cb_model, cf_model = load_models()


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────

st.title("📚 BookMatch — Personalised Book Recommendations")
st.markdown("A hybrid recommendation engine using **TF-IDF content similarity** + **SVD collaborative filtering** on the Book-Crossing dataset.")

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Books", f"{df['ISBN'].nunique():,}")
with col2:
    st.metric("Total Ratings", f"{len(df):,}")
with col3:
    st.metric("Active Users", f"{df['USER_ID'].nunique():,}")
with col4:
    st.metric("Avg Rating", f"{df['BOOK_RATING'].mean():.2f} / 10")

st.divider()


# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Find Similar Books", "👤 Personalised For You", "🔥 Popular Books"])


# ── TAB 1: Content-Based ──────────────────

with tab1:
    st.subheader("Content-Based Recommendations")
    st.caption("Enter a book title and we'll find similar books using TF-IDF + cosine similarity on title, author, and publisher features.")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        book_input = st.text_input("Enter a book title", placeholder="e.g. Harry Potter, The Alchemist, Lord of the Rings")
    with col_btn:
        n_recs = st.selectbox("Results", [5, 10, 15], index=0)

    if st.button("Find Similar Books", type="primary"):
        if not book_input.strip():
            st.warning("Please enter a book title.")
        else:
            with st.spinner("Finding similar books..."):
                results, matched = cb_model.recommend(book_input, n=n_recs)

            if isinstance(results, pd.DataFrame) and not results.empty:
                st.success(f"Showing recommendations based on: **{matched}**")
                for i, row in results.iterrows():
                    with st.container():
                        c1, c2, c3 = st.columns([3, 2, 1])
                        with c1:
                            st.markdown(f"**{row['BOOK_TITLE']}**")
                        with c2:
                            st.markdown(f"✍️ {row['BOOK_AUTHOR']}")
                        with c3:
                            st.markdown(f"<span class='badge'>sim: {row['SIMILARITY']}</span>", unsafe_allow_html=True)
                        st.divider()
            else:
                st.error(matched)


# ── TAB 2: Collaborative Filtering ────────

with tab2:
    st.subheader("Personalised Recommendations via SVD")
    st.caption("Select a user ID to see personalised recommendations based on similar users' reading patterns (collaborative filtering via matrix factorization).")

    sample_users = cf_model.get_sample_users(20)
    col_u, col_n = st.columns([3, 1])
    with col_u:
        selected_user = st.selectbox("Select User ID", sample_users)
    with col_n:
        n_cf = st.selectbox("Results", [5, 10], index=0, key="cf_n")

    if st.button("Get My Recommendations", type="primary"):
        with st.spinner("Generating personalised recommendations..."):
            recs, uid = cf_model.recommend(selected_user, n=n_cf)

        if not recs.empty:
            st.success(f"Top {n_cf} personalised picks for user **{uid}**")

            # Show what user has already rated
            user_history = df[df['USER_ID'] == selected_user][['BOOK_TITLE', 'BOOK_RATING']].sort_values('BOOK_RATING', ascending=False).head(5)
            with st.expander("📖 This user's top rated books"):
                st.dataframe(user_history, use_container_width=True, hide_index=True)

            st.markdown("**Recommended for you:**")
            for i, row in recs.iterrows():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{i+1}. {row['BOOK_TITLE']}**")
                with c2:
                    st.markdown(f"<span class='badge'>pred: {row['PREDICTED_RATING']}</span>", unsafe_allow_html=True)
                st.divider()
        else:
            st.error("Could not generate recommendations for this user.")

    st.info("💡 **Cold-start note:** For new users with no rating history, we fall back to popularity-based recommendations (see Popular Books tab).")


# ── TAB 3: Popular / Cold-Start ──────────

with tab3:
    st.subheader("Top Rated Books — Cold Start Baseline")
    st.caption("Books with the highest average ratings among those with 50+ reviews. Used as fallback for new users.")

    n_pop = st.slider("Number of books", 5, 20, 10)
    popular = get_popular_books(df, n=n_pop)

    for i, row in popular.iterrows():
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            st.markdown(f"**{i+1}. {row['BOOK_TITLE']}**")
        with c2:
            st.markdown(f"⭐ {row['AVG_RATING']}")
        with c3:
            st.markdown(f"💬 {int(row['NUM_RATINGS'])} ratings")
        st.divider()


# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────

st.markdown("---")
st.markdown(
    "Built with Python · Scikit-learn · SVD · TF-IDF · Streamlit &nbsp;|&nbsp; "
    "Dataset: [Book-Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)",
    unsafe_allow_html=True
)
