#!/usr/bin/python3
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def window(npa, n=2):
    for start in range(npa.size-n+1):
        yield npa[start:start+n:1]

input_size=3

seq=np.array([2,5,7,10,12]).astype('float64')
print("learn input: " + str(seq))

scaler=StandardScaler()
seq=seq.reshape(-1,1)
seq=scaler.fit_transform(seq)
seq=seq.reshape(-1)

X=np.array([])
y=np.array([])

for chunk in window(seq, n=input_size+1):
    X=np.append(X, chunk[:-1])
    y=np.append(y, chunk[-1])

X=X.reshape((-1,input_size,1))
y=y.reshape((-1,1))

model = Sequential()
model.add(LSTM(5,input_shape=(input_size,1)))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error",optimizer="rmsprop")
model.fit(X,y, epochs=500, verbose=0)

print("\nwyniki:")
for input in X:
    input=input.reshape(1,input_size,1)
    pred=model.predict(input)
    print(scaler.inverse_transform(input.reshape(-1,1)))
    print(scaler.inverse_transform(pred.reshape(-1,1)))

test=seq[-input_size::1]
print(scaler.inverse_transform(test.reshape(-1,1)))
test=test.reshape(1,input_size,1)
y1=model.predict(test)
print(scaler.inverse_transform(y1.reshape(-1,1)))