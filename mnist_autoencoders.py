import sys
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def preprocess(array: np.array):
    """ Normalizes the supplied array and reshapes it into the appropriate format """
    array = array.astype("float32")/255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    print("Final Shape:", array.shape)
    return array


def noise(array):
    """ Adds random noise to each image in the supplied array """
    noise_factor = 0.5
    noise_array = array + noise_factor * \
        np.random.normal(loc=0.0, scale=1.0, size=array.shape)
    return np.clip(noise_array, 0.0, 1.0)


def load_data():
    """ Loading the data and applying the preprocessing steps """
    (train_data, _), (test_data, _) = keras.datasets.mnist.load_data()

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    return train_data, test_data


train_data, test_data = load_data()

# create a copy of data with noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)


def build_model(input_shape=(28, 28, 1)):
    """ Building the autoencoder model for mnist """
    input = layers.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', name="Conv1")(input)
    x = layers.MaxPooling2D((2, 2), padding='same', name='Pool1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), padding='same', name='Pool2')(x)

    # decoder
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation='relu', padding='same', name="Conv1_transpose")(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation='relu', padding='same', name='Conv2_transpose')(x)
    output = layers.Conv2D(1, (3, 3), activation='sigmoid',
                           padding='same', name="output_layer")(x)

    autoencoder = keras.models.Model(input, output, name='AutoEncoder-Model')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


def train_model():
    autoencoder = build_model()
    autoencoder.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'tmp',
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    autoencoder.fit(
        x=noisy_train_data,
        y=train_data,
        epochs=100,
        batch_size=128,
        shuffle=True,
        validation_data=(noisy_test_data, test_data),
        callbacks=[early_stopping, model_checkpoint])


def display(array1, array2):
    """
    Displays n random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def show_output(img_array):
    """ function for showing the output """
    try:
        autoencoder = keras.models.load_model(
            "tmp")  # loading model from tmp folder
    except Exception:
        print("There is no model please train the model first then use the run command")

    predictions = autoencoder.predict(noisy_test_data)
    display(noisy_test_data, predictions)


if __name__ == '__main__':
    try:
        if sys.argv[1] == "train":
            train_model()
        if sys.argv[1] == "run":
            show_output(noisy_test_data[9])
    except Exception:
        print("Please Use train and run argument to run the process. check the Readme for more details")
