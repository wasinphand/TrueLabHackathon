import argparse
from tensorflow import keras
import pandas as pd
import librosa
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler


ap = argparse.ArgumentParser()
ap.add_argument("-y", "--y_test", required=True, help="path to csv file")
ap.add_argument("-x", "--x_test", required=True, help="path to voice file")
ap.add_argument("-m", "--model", required=True,
                default="./model_v1.h5", help="path to model file")

args = vars(ap.parse_args())


def extract_features(filename):
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


X_TEST_PATH = args['x_test']
Y_TEST_PATH = args['y_test']
MODEL_PATH = args['model']

model = keras.models.load_model(MODEL_PATH, compile=False)
df = pd.read_excel(Y_TEST_PATH)
df.columns = df.iloc[0]
df = df.drop([0])
data = {'speaker': [], 'file': []}
print(df)
for index, row in df.iterrows():

    for i in range(row['No. of Files']):
        data['speaker'].append(int(row['ID No.'][2:5]))
        x = str(i+1) if len(str(i+1)) == 2 else '0'+str(i+1)
        data['file'].append(row['ID No.'] + x+'.mp3')
data = pd.DataFrame(data=data)

train_features = []
for index, row in data.iterrows():
    train_features.append(extract_features(X_TEST_PATH+'/'+row['file']))


features_train = []
for i in range(0, len(train_features)):
    features_train.append(np.concatenate((
        train_features[i][0],
        train_features[i][1],
        train_features[i][2],
        train_features[i][3],
        train_features[i][4]), axis=0))

X_test = np.array(features_train)
y_test = np.array(data['speaker'])


lb = LabelEncoder()
ss = StandardScaler()
X = ss.fit_transform(X_test[0:1])
X_test = ss.transform(X_test)
#y_test = to_categorical(lb.fit_transform(y_test))


predictions = model.predict_classes(X_test)
print(y_test, predictions)

#predictions = lb.inverse_transform(predictions)
predictions = np.array(predictions)
score = float(np.where(predictions == y_test)[
              0].shape[0]/predictions.shape[0]) * 100
print('score = ', score, ' out of 100')
