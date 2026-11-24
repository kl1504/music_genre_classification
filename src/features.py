import numpy as np
import librosa

SAMPLE_RATE = 22050
TRACK_DURATION = 30
NUM_SEGMENTS = 10
N_MFCC = 13
HOP_LENGTH = 512
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = SAMPLES_PER_TRACK // NUM_SEGMENTS


def extract_features(segment, sr=SAMPLE_RATE):
    feats = {}

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(y=segment, sr=sr, hop_length=HOP_LENGTH)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = np.mean(chroma[i])
        feats[f"chroma{i+1}_std"] = np.std(chroma[i])

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=HOP_LENGTH)
    feats["spectral_centroid_mean"] = np.mean(centroid)
    feats["spectral_centroid_std"] = np.std(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=HOP_LENGTH)
    feats["spectral_rolloff_mean"] = np.mean(rolloff)
    feats["spectral_rolloff_std"] = np.std(rolloff)

    zcr = librosa.feature.zero_crossing_rate(segment, hop_length=HOP_LENGTH)
    feats["zcr_mean"] = np.mean(zcr)
    feats["zcr_std"] = np.std(zcr)

    return feats
