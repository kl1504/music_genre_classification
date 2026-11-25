# 🎵 Music Genre Classification

Web application that predicts the genre of a music track from a `.wav` audio file.

🔗 **Live Demo: https://musical-classification.streamlit.app**

## Recognized Genres

Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.

## How It Works

1. Each 30-second track is split into 3-second segments.
2. For each segment, 56 audio features are extracted using `librosa` (MFCC, Chroma, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate).
3. A Machine Learning model predicts the genre of each segment.
4. Segment probabilities are averaged to produce the final prediction.

The dataset used is **GTZAN** (1,000 30-second clips, 100 per genre). The train/test split is performed **by track** to prevent segments from the same song from leaking across both sets.

## Project Structure

```text
├── app/
|   |── app.py
|   |── Dockerfile              
├── src/
│   ├── features.py             # Audio feature extraction
│   └── pipeline.py             # Model loading and prediction
├── models/                     # Selected model, scaler, encoder
├── music_classification.ipynb  # Analysis, training, and evaluation
├── Dockerfile
├── requirements.txt
└── packages.txt                # System dependencies (Streamlit Cloud)
```

## Run the Project

```bash
git clone https://github.com/kl1504/music_genre_classification.git
cd music_genre_classification
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

The application opens at `http://localhost:8501`.

## With Docker

```bash
docker build -t music-genre-app .
docker run -p 8501:8501 music-genre-app
```

## Technologies

Python · librosa · scikit-learn · TensorFlow (notebook) · Streamlit · Docker

## Future Improvements

- Data augmentation (pitch shift, time stretch)
- 2D CNN on Mel spectrograms
- Testing pre-trained models (YAMNet, AST)
