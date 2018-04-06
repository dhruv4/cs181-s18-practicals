import numpy as np
import pandas as pd

import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


SAMPLE_RATE = 22050  # number of samples of the audio per second


def eval(pred, actual):

    count = 0.0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count += 1

    return count / len(actual)


def extract_feature(X, sample_rate, BINS_OCTAVE, NUM_BINS):
    # X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(
        librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(
        librosa.feature.tonnetz(
            y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    return np.array(ext_features)


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i, p in enumerate(predictions):
            f.write(str(i) + "," + str(p) + "\n")


# Given a wav time series, makes a mel spectrogram
# which is a short-time fourier transform with
# frequencies on the mel (log) scale.
def mel_spec(y, SAMPLE_RATE, BINS_OCTAVE, NUM_BINS):
    Q = librosa.cqt(
        y=y, sr=SAMPLE_RATE, bins_per_octave=BINS_OCTAVE, n_bins=NUM_BINS)
    Q_db = librosa.amplitude_to_db(Q, ref=np.max)
    return Q_db


def tempogram(y, SAMPLE_RATE):
    Q = librosa.feature.tempogram(y=y, sr=SAMPLE_RATE)
#     Q_db = librosa.amplitude_to_db(Q,ref=np.max)
    return Q


def main():

    # JUST SOME FOURIER TRANSFORM PARAMETERS
    BINS_OCTAVE = 12 * 2
    N_OCTAVES = 7
    NUM_BINS = BINS_OCTAVE * N_OCTAVES

    GENERATE_FEATURES = True
    GENERATE_TEST = False
    N = None

    # Just some re-shaping and dimension finding
    # This means that the spectrograms are 168 rows (frequencies)
    # By 173 columns (time frames)
    if GENERATE_TEST:
        test = np.array(pd.read_csv("test.csv", header=None, nrows=N))
        N_test = test.shape[0]
        test = test[:, 1:-1]
        tmp_test = np.zeros((N_test, 193))

        for i in range(N_test):
            f = extract_feature(test[i], SAMPLE_RATE, BINS_OCTAVE, NUM_BINS)
            tmp_test[i] = np.array(f)

        np.save("features-test-dhruv.npy", tmp_test)
    else:
        tmp_test = np.load("features-test-dhruv.npy")

    if GENERATE_FEATURES:
        train = np.array(pd.read_csv("train.csv", header=None, nrows=N))

        print "N:", N
        # train = signal[np.newaxis,:]
        print "Train shape", train.shape
        N_train = train.shape[0]
        # NUM_SAMPLES = train.shape[1] - 1

        y_train = train[:, -1]
        y_train = y_train.reshape(N_train, 1)

        # remove first and last two elements to drop the mic
        X_train = train[:, 1:-2]

        song = X_train[0]
        test_spec = mel_spec(song, SAMPLE_RATE, BINS_OCTAVE, NUM_BINS)
        print test_spec.shape

        tmp_train = np.zeros((N_train, 193))

        for i in range(N_train):
            f = extract_feature(X_train[i], SAMPLE_RATE, BINS_OCTAVE, NUM_BINS)
            tmp_train[i] = np.array(f)

        np.save("features-train-dhruv.npy", tmp_train)
        np.save("y-train-features-dhruv.npy", y_train)

    else:
        tmp_train = np.load("features-train-dhruv.npy")
        y_train = np.load("y-train-features-dhruv.npy")

    X_train, X_valid, y_train, y_valid = train_test_split(
        tmp_train, y_train, test_size=0.01)

    print "fitting"
    print X_train.shape

    param_grid = {
        # "max_depth": [3, 10, 200, 1000],
        # "learning_rate": [x / 100.0 for x in range(1, 100, 10)],
        "max_features": [x / 100.0 for x in range(20, 101, 10)]
    }

    rf = RandomForestClassifier(n_jobs=-1)
    rfr = GridSearchCV(rf, param_grid, verbose=10, n_jobs=-1)
    rfr.fit(X_train, y_train)
    valid_pred = rfr.predict(X_valid)

    print "Random Forest Evaluation score:", eval(valid_pred, y_valid)

    rfr_test_pred = rfr.predict(tmp_test)

    write_to_file("rfr_preds.csv", rfr_test_pred)


if __name__ == '__main__':
    main()
