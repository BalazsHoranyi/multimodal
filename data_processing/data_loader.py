import pandas as pd
import os
from data_processing.paths import base_path, image_path, audio_path
# we have text, audio and image.
import librosa
from image_processing.utils import get_vector as get_image_vector
from text_processing.utils import get_vector as get_text_vector
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_train = pd.read_csv(os.path.join(base_path, 'train_sent_emo.csv'))
features = []
for i, row in tqdm(df_train.iterrows()):
    try:
        base_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        image_file = os.path.join(image_path, base_name, 'frame_0.jpg')
        audio_file = os.path.join(audio_path, f"{base_name}.wav")

        # extract features
        # TODO think about what features you should be extracting for each modal. These are all dummies.
        # once we decide, these should be cahced somehwere so we don't have to keep recomputing things.
        audio_feature = librosa.feature.melspectrogram(librosa.load(audio_file)[0])
        audio_feature = audio_feature.mean(axis=1)
        image_feature = get_image_vector(image_file)
        text_feature = get_text_vector(row['Utterance'])

        features.append(np.hstack([audio_feature, image_feature, text_feature]))
    except Exception as e:
        #figure out corrupted file? dia125_utt3.mp4
        print(e)

# TODO add sparse feature here. speaker id, season, episode etc...

features = np.stack(features, axis=0)

y = df_train[~((df_train['Dialogue_ID']==125) & (df_train['Utterance_ID']==3))]['Sentiment'].factorize()[0]

# TODO use train test splits given by competition
X_train, X_test, y_train, y_test = train_test_split(features, y)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)
param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 3}
num_round = 10
bst = xgb.train(param, dtrain, num_round)

print(f"accuracy: {accuracy_score(y_test, bst.predict(dtest))}")
# gets about 60 percent