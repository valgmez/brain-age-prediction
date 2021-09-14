import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import keras
from keras import layers
from keras.utils import to_categorical, layer_utils
from keras.models import Sequential, Input, Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.utils import plot_model
import keras.backend as K
import re
import nibabel as nb
from skimage.transform import resize
from skimage.util import crop
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def extract_data():
    # This function extracts the mris individually and puts them in a numpy array
    mri_array = []
    ages = []

    # path to highres mris on supercomputer
    data_dir = '/scratch/c.sapmt8/camcan/cc700_hires/mri/pipeline/release004/data/aamod_dartel_normmni_00001'
    labels_df = pd.read_table(
        '/scratch/c.sapmt8/camcan/CC700_mt.txt', engine='python', sep='\t')  # demographics file

    # iterate over all the folders (one per scan), get the grey-matter mri, convert .nii into numpy and match the subject id with age in demographics
    for folder in sorted(os.listdir(data_dir)):
        if not os.path.isdir(data_dir + '/' + folder):
            pass
        else:
            # smwc1 for GM, smwc2 for WM and smwc3 for CSF
            x = os.listdir(data_dir + '/' + str(folder) + '/structurals/')
            mri_file = [mri for mri in x if re.search("smwc1", mri)][0]
            mri_img = nb.load(data_dir + '/' + folder +
                              '/structurals/' + mri_file)
            mri_img_data = mri_img.get_fdata()
            mri_img_data = crop(mri_img_data, 19)
            mri_img_data = resize(
                mri_img_data, (96, 112, 96))  # low res version
            # mri_img_data = resize(
            #     mri_img_data, (143, 167, 143))  # high res version
            mri_array.append(mri_img_data)

            print(data_dir + '/' + folder + '/structurals/' + mri_file)

            mri_age = labels_df.loc[labels_df.SubCCIDc ==
                                    str(folder), 'Age'].iloc[0]  # matching SubCCIDc with each MRI's ID to get the age
            ages.append(mri_age)

    print('Saving the numpy arrays...')
    np.save('/nfshome/store01/users/c.c1732102/ages2.npy', ages)
    np.save('/nfshome/store01/users/c.c1732102/mris2.npy', mri_array)


def run_model(model, n_epochs):
    # --- load datasets, mris = X, ages = y
    # mris2 and ages2 for lowres; mris and ages for highres
    mris = np.load('/nfshome/store01/users/c.c1732102/mris2.npy')
    ages = np.load('/nfshome/store01/users/c.c1732102/ages2.npy')

    # --- training only for transfer learning
    X_train = mris
    y_train = ages

    print('Reshaping data...')
    # X_train = X_train.reshape(-1, 143, 167, 143, 1) #highres
    X_train = X_train.reshape(-1, 96, 112, 96, 1)  # lowres

    # --- 1. create the model by calling the model passed as a paremeter
    print('Creating the model...')
    model = model(X_train.shape[1:])

    # --- folder to save model and plots
    folder_name = "models/" + str(n_epochs) + '_' + model.name

    # --- 2. compile model
    print('Compiling the model...')
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mae', rmse])

    # stops training early when there is no improvement in the validation loss for 10 consecutive epochs
    early = EarlyStopping(monitor='loss', patience=10, verbose=1)

    # model checkpoint, saves the model when best weights are found
    mc = ModelCheckpoint('models/BrainAgeModel-camcan', monitor='mae',
                         verbose=1, mode='min', save_best_only=True)

    # --- 3. train the model
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, epochs=n_epochs,
                        batch_size=4, callbacks=[early, mc])

    return


def rmse(y_true, y_pred):
    '''calculates the root mean squared error between labels and predictions '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


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

    X = Dropout(0.2)(X)
    # fully connected layers
    X = Dense(256, activation='relu')(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(1)(X)

    # create the model instance, used to train/test the model
    model = Model(inputs=X_input, outputs=X, name='BrainAgeModel')

    model.summary()

    return model


def main():
    extract_data()
    run_model(BrainAgeModel, 200)


if __name__ == '__main__':
    main()
