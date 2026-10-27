
import pickle
from pathlib import Path

import requests
import streamlit as st

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# OMDb API KEY
# -----------------------------
OMDB_API_KEY = "ba106b0c"

# -----------------------------
# LOAD YOUR EXISTING ML MODEL
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    movie_path = BASE_DIR / "model" / "movie_list.pkl"
    similarity_path = BASE_DIR / "model" / "similarity.pkl"

    with open(movie_path, "rb") as file:
        movies = pickle.load(file)

    with open(similarity_path, "rb") as file:
        similarity = pickle.load(file)

    return movies, similarity


# -----------------------------
# FETCH MOVIE POSTER FROM OMDb
# -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_title):
    url = "https://www.omdbapi.com/"

    try:
        response = requests.get(
            url,
            params={
                "apikey": OMDB_API_KEY,
                "t": movie_title,
                "type": "movie"
            },
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        poster = data.get("Poster", "N/A")

        if data.get("Response") == "True" and poster != "N/A":
            return poster

    except (requests.RequestException, ValueError):
        pass

    return None


# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend(movie_title, movies, similarity):
    matched = movies[
        movies["title"].astype(str).str.casefold()
        == movie_title.casefold()
    ]

    if matched.empty:
        return []

    index = matched.index[0]

    distances = sorted(
        enumerate(similarity[index]),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    for movie_index, score in distances:
        if movie_index == index:
            continue

        title = str(movies.iloc[movie_index]["title"])

        if title.casefold() == movie_title.casefold():
            continue

        recommendations.append((title, float(score)))

        if len(recommendations) == 10:
            break

    return recommendations


# -----------------------------
# CUSTOM DESIGN
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101426, #191b35, #251535);
    color: #ffffff;
}

h1, h2, h3, p, label {
    color: #ffffff !important;
}

div[data-baseweb="select"] > div {
    background-color: #292d49;
    color: white;
    border: 1px solid #7775ff;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
}

.stButton > button:hover {
    border: 1px solid white;
    color: white;
}

.movie-title {
    color: white;
    font-size: 15px;
    font-weight: 600;
    text-align: center;
    min-height: 42px;
    margin-top: 8px;
}

.movie-card {
    background: #20233d;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# MAIN WEBSITE
# -----------------------------
st.title("🎬 Movie Recommender")
st.caption("Discover movies similar to your favourites.")

try:
    movies, similarity = load_model()
except FileNotFoundError:
    st.error(
        "Model files nahi mile. Check karo ki "
        "model folder ke andar movie_list.pkl "
        "aur similarity.pkl dono hain."
    )
    st.stop()
except Exception as error:
    st.error(f"Model load nahi hua: {error}")
    st.stop()

if "title" not in movies.columns:
    st.error("movie_list.pkl mein 'title' column nahi mila.")
    st.stop()

movie_list = sorted(
    movies["title"].dropna().astype(str).unique().tolist()
)

if not movie_list:
    st.error("Movie dataset empty hai.")
    st.stop()

selected_movie = st.selectbox(
    "🔎 Search your favourite movie",
    movie_list,
    index=None,
    placeholder="Type a movie name..."
)

if st.button("✨ Recommend Movies", use_container_width=True):

    if not selected_movie:
        st.warning("Pehle koi movie search karke select karo.")
    else:
        recommendations = recommend(
            selected_movie, movies, similarity
        )

        if not recommendations:
            st.warning("Try some other movies.")
        else:
            st.subheader(f"🎥 Movies similar to {selected_movie}")

            # Display the selected movie too
            st.markdown("### Your selected movie")

            selected_poster = fetch_poster(selected_movie)

            selected_col, _ = st.columns([1, 4])

            with selected_col:
                if selected_poster:
                    st.image(selected_poster, width=180)
                else:
                    st.info("Selected movie poster not available.")

                st.markdown(
                    f'<div class="movie-title">{selected_movie}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("### 🍿 Recommended for you")

            # Five medium-size posters per row
            for start in range(0, len(recommendations), 5):
                batch = recommendations[start:start + 5]
                cols = st.columns(5)

                for col, (title, score) in zip(cols, batch):
                    with col:
                        poster = fetch_poster(title)

                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                        if poster:
                            st.image(poster, use_container_width=True)
                        else:
                            st.markdown(
                                "🎬 **Poster unavailable**",
                                help="OMDb mein is movie ka poster nahi mila."
                            )

                        st.markdown(
                            f'<div class="movie-title">{title}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown('</div>', unsafe_allow_html=True)

            st.caption(
                "Recommendations are generated by your existing "
                "Cosine Similarity model. Posters are fetched from OMDb."
            )
