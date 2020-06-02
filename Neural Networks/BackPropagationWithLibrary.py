

import pandas as pd
import numpy as np
import tensorflow
from keras.models import Sequential
from keras import optimizers
from keras.initializers import RandomUniform
from keras.layers import Dense
from skimage import io
from sklearn import metrics


def parseInput(file):
    df = pd.read_csv(file, names=['path'])
    df['image'] = [1 if "down" in x else 0 for x in df['path']] 
    
    input = []
    for imagePath in df['path']:
        img =io.imread(imagePath, -1)
        reshaped_img = list(img.reshape(len(img) * len(img[0])))
        input.append(reshaped_img)
    
    input = np.array(input)
    input = input/255
    output = df['image'].to_numpy()
    return input, output

def main():
    train_in, train_out = parseInput("downgesture_train.list")
    test_in, test_out = parseInput("downgesture_test.list")
    model = Sequential([
        Dense(100, input_shape=(960,), activation='sigmoid', kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)),
        Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=None)),
    ])
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    model.fit(train_in, train_out, epochs=1000)
    loss, accuracy = model.evaluate(test_in, test_out)

    predictions = model.predict_classes(test_in)
    print("\nPrediction on test data:")
    print(predictions.reshape(1,83)[0])

    print("\nAccuracy on test data:",accuracy*100,"%")

if __name__=="__main__":
    main()
