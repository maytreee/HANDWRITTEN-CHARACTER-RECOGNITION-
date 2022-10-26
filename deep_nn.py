import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import cv2

# load the mnist dataset from the keras module
mnist = tf.keras.datasets.mnist

# Loading data and splitting into train and test
(xtrain, ytrain), (xtest,y_test) = mnist.load_data()

# normalize the split data
x_train = tf.keras.utils.normalize(xtrain, axis = 1)
x_test  = tf.keras.utils.normalize(xtest, axis = 1)

# rshape the test and train data
x_train_np = np.array(xtrain).reshape(-1, 28, 28, 1)
x_test_np  = np.array(xtest).reshape(-1, 28, 28, 1)

# Creating the NN model
model = Sequential()

# create each layer sequentially
model.add(Conv2D(64, (3,3), input_shape = x_train_np.shape[1:])) 
model.add(Activation('relu')) # activation function
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu')) # activation function
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu')) # activation function
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu")) # activation function

model.add(Dense(32))
model.add(Activation("relu")) # activation function

model.add(Dense(10))
model.add(Activation("softmax")) # activation function

# print the model summary
model.summary()

# compile and fit the model
model.compile(loss="sparse_categorical_crossentropy", metrics=["mae", "acc", "mape"])
model.fit(x_train_np, ytrain, validation_split = 0.3, epochs = 5)

loss, mae, accuracy, mape = model.evaluate(x_test_np, y_test)
print("Loss:", loss, "\tTest Accuracy:", accuracy)
print("MAE:", mae, "\tMAPE:", mape)

preds = model.predict([x_test_np])
print("Predictions: ", preds)

plt.imshow(x_test[0])
plt.show()
plt.title("Predicted value for the image ->"  + str(np.argmax(preds[0])))
print("Predicted value for the image ->", np.argmax(preds[0]))

import sys
# run the demo server if command line arguments speficied
if len(sys.argv) > 1:
    from flask import Flask, render_template, request
    from werkzeug.utils import secure_filename
    import imageio
    import numpy

    app = Flask(__name__)

    @app.route('/upload')
    def upload_file():
        return render_template('upload.html')
        
    @app.route('/uploader', methods = ['GET', 'POST'])
    def upload_file2():
        if request.method == 'POST':
            f = request.files['file']
            f.save(secure_filename(f.filename))

            img = cv2.imread( secure_filename(f.filename))
            plt.imshow(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
            newimg = tf.keras.utils.normalize(resized, axis = 1)
            newimg = np.array(newimg).reshape(-1, 28, 28, 1)
            predictions = model.predict(newimg)
            print(np.argmax(predictions[0]))

            res1 = str(np.argmax(predictions[0]))
        

            return render_template('upload.html', result=res1) #'Prediction: ' + str(np.argmax(predictions[0]))
            
    if __name__ == '__main__':
        app.run(debug = True)