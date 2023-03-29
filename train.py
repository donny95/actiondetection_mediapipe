import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# Actions that we try to detect
actions = np.array(['hello2', 'hush2', 'no', 'stand', 'what'])
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)


no_sequences = 60
# Videos are going to be 30 frames in length  window?
sequence_length = 30



# Path for exported data, numpy arrays
path = "C:/Users/N-215/PycharmProjects/pythonbasic/"
# DATA_PATH = "C:/Users/N-215/PycharmProjects/pythonbasic/"
DATA_PATH = os.path.join(path,'MP_Data4')




sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(np.array(sequences).shape)
X = np.array(sequences)
print("X shape: ", X.shape)
y = to_categorical(labels).astype(int)
# print(y.shape())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
# input_shape = (frame window, keypoints)

model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



# tb_callback = TensorBoard(log_dir=log_dir)
# callback = EarlyStopping(monitor='categorical_accuracy',patience=20)

mc = ModelCheckpoint("./models/0323_5a_30f_newhush_loss2_1000.h5", monitor='loss', verbose=1, save_best_only=True)
mc2 = ModelCheckpoint("./models/0323_5a_30f_newhush_acc2_1000.h5", monitor='categorical_accuracy', verbose=1, save_best_only=True)

# history = model.fit(X_train, y_train, epochs=2000, batch_size=2, callbacks=[tb_callback, callback, mc], verbose=1)
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback, mc, mc2], verbose=1)

model.summary()

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
print(accuracy_score(ytrue, yhat))

res = model.predict(X_test)
model.save('./models/0323_5a_30f_newhush2_1000.h5')
print("model saved!")
# print(res)

print("evaluate : ", model.evaluate(X_test, y_test))


