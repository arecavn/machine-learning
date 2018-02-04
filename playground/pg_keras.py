# sudo pip install keras (keras-2.0.8 pyyaml-3.12 scipy-0.19.1)
# sudo pip install tensorflow (tensorflow-1.3.0 tensorflow-tensorboard-0.1.5)

# from __future__ import print_function

def simple_test():
    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    # X has shape (num_rows, num_cols), where the training data are stored
    # as row vectors
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    # y must have an output vector for each input vector
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)

    # Create the Sequential model
    model = Sequential()

    # 1st Layer - Add an input layer of 32 nodes with the same input shape as
    # the training samples in X
    print X.shape[1], X.shape[0]
    model.add(Dense(32, input_dim=X.shape[1]))

    # 2rd Layer - Add a softmax activation layer
    model.add(Activation('softmax'))

    # 4th Layer - Add a fully connected output layer
    model.add(Dense(1))

    # 5th Layer - Add a sigmoid activation layer
    model.add(Activation('sigmoid'))


def network_exercise():
    # This has 75% accurate (run on web) with only 20 inputs and 60 epochs
    import numpy as np
    # tf.python.control_flow_ops = tf # This line cause error no 'python'

    # Set random seed
    np.random.seed(55)

    # Our data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype('float32')
    y = np.array([[0], [1], [1], [0]]).astype('float32')

    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    # Building the model
    xor = Sequential()

    # Add required layers
    xor.add(Dense(20, input_dim=X.shape[1]))
    xor.add(Activation('tanh'))
    xor.add(Dense(1))
    xor.add(Activation('sigmoid'))

    # Specify loss as "binary_crossentropy", optimizer as "adam",
    # and add the accuracy metric
    # xor.compile()
    xor.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Uncomment this line to print the model architecture
    xor.summary()

    # Fitting the model
    history = xor.fit(X, y, nb_epoch=60, verbose=1)

    # Scoring the model
    score = xor.evaluate(X, y)
    print("\nAccuracy: ", score[-1])

    # Checking the predictions
    print("\nPredictions:")
    print(xor.predict_proba(X))


def network_solution():
    # This has only 75% accurate (run on web), provided by Udacity but take up to 32 inputs and 1000 epoch
    import numpy as np
    from keras.utils import np_utils
    # tf.python.control_flow_ops = tf # This line cause error no 'python'

    # Set random seed
    np.random.seed(42)

    # Our data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype('float32')
    y = np.array([[0], [1], [1], [0]]).astype('float32')

    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    # One-hot encoding the output
    y = np_utils.to_categorical(y)

    # Building the model
    xor = Sequential()
    xor.add(Dense(32, input_dim=2))
    xor.add(Activation("sigmoid"))
    xor.add(Dense(2))
    xor.add(Activation("sigmoid"))

    xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Uncomment this line to print the model architecture
    # xor.summary()

    # Fitting the model
    history = xor.fit(X, y, nb_epoch=1000, verbose=0)

    # Scoring the model
    score = xor.evaluate(X, y)
    print("\nAccuracy: ", score[-1])

    # Checking the predictions
    print("\nPredictions:")
    print(xor.predict_proba(X))


def keras_mnist_mlp():
    '''Trains a simple deep NN on the MNIST dataset.
    Gets to 98.40% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a K520 GPU.
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    '''

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop

    batch_size = 128
    num_classes = 10
    epochs = 20

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# network_exercise()
keras_mnist_mlp()
