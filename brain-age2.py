import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import keras
from keras import layers
from keras.utils import to_categorical, layer_utils
from keras.models import Sequential, Input, Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
import keras.backend as K

data_dir = '/scratch/c.sapmt8/camcan/real_crop'  # path to mris on supercomputer
mris = os.listdir(data_dir)
labels_df = pd.read_table(
    '/scratch/c.sapmt8/camcan/CC700_mt.txt', engine='python', sep='\t')
# labels_df.head()  -- displays table of mri data

# convert file CC700.txt into .npy by selecting all the ages where the SubCCIDc matches the list of sample MRIs and adding them to a list which gets converted into .npy
age = []
mri_array = []

for mri in mris:
    # loads each mri as numpy array before adding to array containing all mris
    mri_as_array = np.load(data_dir + '/' + mri)
    # adds each ny mri file to an array called mri_array
    mri_array.append(mri_as_array)
    label = labels_df.loc[labels_df.SubCCIDc ==
                          os.path.splitext(mri)[0], 'Age'].iloc[0]
    age.append(label)

# saving mris and ages into their own npy arrays
np.save('/nfshome/store01/users/c.c1732102/ages.npy', age)
np.save('/nfshome/store01/users/c.c1732102/mris.npy', mri_array)

# load them
ages = np.load('/nfshome/store01/users/c.c1732102/ages.npy')
mris = np.load('/nfshome/store01/users/c.c1732102/mris.npy')

# clean up data and ensure in correct format
# 80/20 train test split
# reshape
X_train, X_test, y_train, y_test = train_test_split(
    mris, ages, test_size=0.2, random_state=13, stratify=ages)  # added stratify for ages

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))

X_train = X_train.reshape(-1, 96, 112, 96, 1)
X_test = X_test.reshape(-1, 96, 112, 96, 1)

# building the model


def BrainAgeModel(input_shape):

    # define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    # 1st layer CONV -> BN -> MaxPooling applied to X
    X = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(X_input)
    X = BatchNormalization(axis=4)(X)
    X = MaxPooling3D((2, 2, 2), strides=2, padding='same')(X)

    # 2nd layer CONV -> BN -> MaxPooling applied to X
    X = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=4)(X)
    X = MaxPooling3D((2, 2, 2), strides=2, padding='same')(X)

    # 3rd layer CONV -> BN -> MaxPooling applied to X
    X = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=4)(X)
    X = MaxPooling3D((2, 2, 2), strides=2, padding='same')(X)

    # 4th layer CONV -> BN -> MaxPool applied to X
    X = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=4)(X)
    X = MaxPooling3D((2, 2, 2), strides=2, padding='same')(X)

    # 5th layer CONV -> BN -> MaxPooling applied to X
    X = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=4)(X)
    X = MaxPooling3D((2, 2, 2), strides=2, padding='same')(X)

    # flatten X (convert it to a vector)
    X = Flatten()(X)

    # fully connected layers
    X = Dense(64, activation='relu')(X)
    X = Dense(32, activation='relu')(X)

    X = Dense(1)(X)

    # create the model instance, used to train/test the model
    model = Model(inputs=X_input, outputs=X, name='BrainAgeModel')

    model.summary()
    plot_model(model, to_file='dinsdaleish.png')

    return model


# 1. create the model by calling function above
brainAgeModel = BrainAgeModel(X_train.shape[1:])

# 2. compile model
brainAgeModel.compile(
    optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

# 3. train the model
history = brainAgeModel.fit(x=X_train, y=y_train, epochs=200,
                            batch_size=4, validation_data=(X_test, y_test))

# 4. evaluate the model
test_eval = brainAgeModel.evaluate(x=X_test, y=y_test)
print()
print('Loss: ', test_eval[0])
print('Mean Squared Error: ', test_eval[1])
print('Mean Absolute Error: ', test_eval[2])


def plot_results(brainAgeModel):
    mse = brainAgeModel.history['mse']
    val_mse = brainAgeModel.history['val_mse']
    mae = brainAgeModel.history['mae']
    val_mae = brainAgeModel.history['val_mae']
    loss = brainAgeModel.history['loss']
    val_loss = brainAgeModel.history['val_loss']
    epochs = range(len(mse))

    plt.plot(epochs, mse, label='Training MSE')
    plt.plot(epochs, val_mse, label='Val MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('mris_mse.png')

    plt.figure()
    plt.plot(epochs, mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('mris_mae.png')

    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Loss evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('mris_loss.png')


plot_results(history)

# testing it
my_test = np.load(data_dir + '/CC721519.npy')
plt.axis('off')
plt.imshow(my_test[:, :, 48].T, cmap='gray', origin='lower')
my_test = my_test.reshape(-1, 96, 112, 96, 1)
print(brainAgeModel.predict(my_test))
