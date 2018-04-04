import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050 # number of samples of the audio per second

def eval(pred, actual):

    count = 0.0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count += 1

    return count / len(actual)

# Given a wav time series, makes a mel spectrogram
# which is a short-time fourier transform with
# frequencies on the mel (log) scale.
def mel_spec(y, SAMPLE_RATE, BINS_OCTAVE, NUM_BINS):
	Q = librosa.cqt(y=y, sr=SAMPLE_RATE, bins_per_octave=BINS_OCTAVE,n_bins=NUM_BINS)
	Q_db = librosa.amplitude_to_db(Q,ref=np.max)
	return Q_db

def tempogram(y, SAMPLE_RATE):
    Q = librosa.feature.tempogram(y=y, sr=SAMPLE_RATE)
#     Q_db = librosa.amplitude_to_db(Q,ref=np.max)
    return Q

def main():

	N = 10000

	# Just some re-shaping and dimension finding

	train = np.array(pd.read_csv("train.csv", header=None, nrows=N))
	test = np.array(pd.read_csv("test.csv", header=None, nrows=N))

	print "N:",N
	# train = signal[np.newaxis,:]
	print "Train shape",train.shape
	N_train = train.shape[0]
	N_test = test.shape[0]
	NUM_SAMPLES = train.shape[1]-1

	test = test[:,1:-1]

	X_train = train[:,1:-2] # remove first and last two elements to drop the mic
	y_train = train[:,-1]
	y_train = y_train.reshape(N_train,1)

	# JUST SOME FOURIER TRANSFORM PARAMETERS
	BINS_OCTAVE = 12*2
	N_OCTAVES = 7
	NUM_BINS = BINS_OCTAVE * N_OCTAVES

	# This means that the spectrograms are 168 rows (frequencies)
	# By 173 columns (time frames)
	song = X_train[0]
	test_spec = mel_spec(song, SAMPLE_RATE, BINS_OCTAVE, NUM_BINS)
	print test_spec.shape
	FEATS = test_spec.shape[0]
	FRAMES = test_spec.shape[1]

	print "features"

	tmp_train = np.zeros((N_train,FEATS,FRAMES))
	for i in range(N_train):
	    tmp_train[i,:,:] = mel_spec(X_train[i], SAMPLE_RATE, BINS_OCTAVE, NUM_BINS)

	X_train, X_valid, y_train, y_valid = train_test_split(tmp_train, y_train, test_size=0.2)

	regr = LinearRegression()

	print "fitting"

	# print X_train[0], y_train[0]

	print X_train.shape

	nsamples, nx, ny = X_train.shape
	d2_train_dataset = X_train.reshape((nsamples,nx*ny))

	nsamples, nx, ny = X_valid.shape
	d2_valid_dataset = X_valid.reshape((nsamples,nx*ny))

	regr.fit(d2_train_dataset, y_train)

	valid_pred = regr.predict(d2_valid_dataset)

	print "Evaluation score:", eval(valid_pred, y_valid)

	# tmp_test = np.zeros((N_test,FEATS,FRAMES))
	# for i in range(N_test):
	#     tmp_test[i,:,:] = mel_spec(test[i])

if __name__ == '__main__':
	main()

