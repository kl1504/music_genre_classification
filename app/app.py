import pandas as pd
import streamlit as st

from pipeline import load_artifacts, predict_genre


@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

st.title("🎵 Music Genre Classifier")
st.write("Chargez un extrait audio au format WAV (idéalement 30 s) : "
         "le modèle prédit son genre musical.")

uploaded = st.file_uploader("Choisissez un fichier .wav", type=["wav"])

if uploaded is not None:
    st.audio(uploaded)

    with st.spinner("Classifying..."):
        genre, proba = predict_genre(uploaded, artifacts)

    if genre is None:
        st.error("Fichier trop court : il faut au moins 3 secondes.")
    else:
        st.markdown(f"<h1 style='text-align:center;'>🎶 {genre.upper()}</h1>",
                    unsafe_allow_html=True)
        st.caption(f"Confiance : {proba.max():.0%}")

        st.subheader("Probabilités par genre")
        st.bar_chart(pd.Series(proba, index=artifacts["le"].classes_))
        
