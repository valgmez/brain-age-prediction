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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.utils import plot_model
import keras.backend as K
import re
import nibabel as nb
from skimage.transform import resize
from skimage.util import crop
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm
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
            x = os.listdir(data_dir + '/' + str(folder) + '/structurals/')
            # smwc1 for GM, smwc2 for WM and smwc3 for CSF
            mri_file = [mri for mri in x if re.search("smwc1", mri)][0]
            mri_img = nb.load(data_dir + '/' + folder +
                              '/structurals/' + mri_file)
            mri_img_data = mri_img.get_fdata()
            mri_img_data = crop(mri_img_data, 19)
            mri_img_data = resize(
                mri_img_data, (96, 112, 96))  # low res version
            # mri_img_data = resize(mri_img_data, (143, 167, 143))  # high res version
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

    # --- save a picture of a brain slice, and print its age
    plt.axis('off')
    plt.imsave('mri_big.png', mris[10, :, :,
                                   48].T, cmap='gray', origin='lower')
    print("Age: ", ages[10])

    # --- split data into train and test, going for 80:20 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        mris, ages, test_size=0.2)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("y_test shape: " + str(y_test.shape))

    print('Reshaping data...')
    # highres
    # X_train = X_train.reshape(-1, 143, 167, 143, 1)
    # X_test = X_test.reshape(-1, 143, 167, 143, 1)

    # lowres
    X_train = X_train.reshape(-1, 96, 112, 96, 1)
    X_test = X_test.reshape(-1, 96, 112, 96, 1)

    K.clear_session()

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

    # --- 3. train the model and plot results
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, epochs=n_epochs,
                        batch_size=4, validation_data=(X_test, y_test), callbacks=[early])

    print('Plotting results...')
    plot_results(history, folder_name)

    # --- 4. predict
    print('Making predictions...')
    y_pred = model.predict(X_test)
    print()
    print('Mean Absolute Error uncorrected (predictions): ',
          mean_absolute_error(y_test, y_pred))

    # --- 5. correction of brain age bias
    # X_train = mris, y_train = chronological ages, yhat_train = predicted ages
    yhat_train = model.predict(X_train)
    y_train = y_train.reshape(-1, 1)
    bias = Lasso().fit(y_train, yhat_train)
    yhat_train = yhat_train.reshape(-1,)
    y_train = y_train.reshape(-1,)

    alpha = bias.coef_[0]
    intercept = bias.intercept_

    corrected_age_train = yhat_train + \
        (y_train - (alpha * y_train + intercept))

    yhat_test = model.predict(X_test)
    yhat_test = yhat_test.reshape(-1,)
    corrected_age_test = yhat_test + (y_test - (alpha * y_test + intercept))

    print()
    print('Correlation between uncorrected predicted age and chronological age (training): ',
          np.corrcoef(y_train, yhat_train)[0, 1])
    print('Correlation between corrected predicted age and chronological age (training): ',
          np.corrcoef(y_train, corrected_age_train)[0, 1])
    print('Correlation between uncorrected predicted age and chronological age (testing): ',
          np.corrcoef(y_test, yhat_test)[0, 1])
    print('Correlation between corrected predicted age and chronological age (testing): ',
          np.corrcoef(y_test, corrected_age_test)[0, 1])

    print()
    print('Correlation between uncorrected brain age delta and chronological age (training): ',
          np.corrcoef(y_train, (yhat_train - y_train))[0, 1])
    print('Correlation between uncorrected brain age delta and chronological age (testing): ',
          np.corrcoef(y_test, (yhat_test - y_test))[0, 1])
    print('Correlation between corrected brain age delta and chronological age (training): ',
          np.corrcoef(y_train, (corrected_age_train - y_train))[0, 1])
    print('Correlation between corrected brain age delta and chronological age (testing): ',
          np.corrcoef(y_test, (corrected_age_test - y_test))[0, 1])

    print()
    print('MAE after correction (training): ',
          (np.abs(corrected_age_train - y_train)).mean())
    print('MAE after correction (testing): ',
          (np.abs(corrected_age_test - y_test)).mean())

    mae = mean_absolute_error(y_test, y_pred)
    root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, root_mse


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


def LenetModel(input_shape):
    """
    Not fully original Lenet because it works better with relu and max pooling
    instead of sigmoid and avg pooling, each conv layer uses 5x5x5 filter, 2x2 pooling
    https://colab.research.google.com/drive/1CVm50PGE4vhtB5I_a_yc4h5F-itKOVL9
    https://d2l.ai/chapter_convolutional-neural-networks/lenet.html
    """
    X_input = Input(input_shape)
    X = Conv3D(filters=32, kernel_size=(
        5, 5, 5), activation='relu')(X_input)
    X = MaxPooling3D((2, 2, 2))(X)

    X = Conv3D(filters=64, kernel_size=(5, 5, 5), activation='relu')(X)
    X = MaxPooling3D((2, 2, 2))(X)

    X = Flatten()(X)

    X = Dense(units=120, activation='relu')(X)
    X = Dense(units=84)(X)
    X = Dense(units=1)(X)

    model = Model(inputs=X_input, outputs=X, name='LeNet5')
    model.summary()

    return model


def VGGModel(input_shape):
    """
    VGG-13 adaptation
    Jiang, H. et al. (2020) ‘Predicting Brain Age of Healthy Adults Based on Structural MRI
    Parcellation Using Convolutional Neural Networks’, Frontiers in Neurology, 10(January).
    doi: 10.3389/fneur.2019.01346.
    """
    X_input = Input(input_shape)

    X = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(X_input)
    X = Conv3D(8, (3, 3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=2)(X)

    X = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(X)
    X = Conv3D(16, (3, 3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=2)(X)

    X = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(X)
    X = Conv3D(32, (3, 3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=2)(X)

    X = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(X)
    X = Conv3D(64, (3, 3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=2)(X)

    X = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(X)
    X = Conv3D(128, (3, 3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=2)(X)

    X = Flatten()(X)

    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    X = Dense(1)(X)

    model = Model(inputs=X_input, outputs=X, name='VGG16')
    model.summary()

    return model


def plot_results(history, folder_name):
    rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(mae))

    plt.figure()
    plt.plot(epochs, rmse, label='Training RMSE')
    plt.plot(epochs, val_rmse, label='Validation RMSE')
    plt.title('Root Mean Squared Error (RMSE)')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(folder_name + '_mris_rmse.png')

    plt.figure()
    plt.plot(epochs, mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(folder_name + '_mris_mae.png')

    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Loss evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(folder_name + '_mris_loss.png')


def main():
    # need to run extract data first time, make sure to change to change lowres and highres as needed
    extract_data()

    all_maes = []
    all_rmses = []
    for i in range(10):
        # can change BrainAgeModel to LenetModel or VGGModel
        mae, rmse = run_model(BrainAgeModel, 200)
        all_maes.append(mae)
        all_rmses.append(rmse)

    print('Average MAE for 10 repetitions: %.4f +/- %.3f' %
          (np.mean(all_maes), np.std(all_maes)))
    print('Average RMSE for 10 repetitions: %.4f +/- %.3f' %
          (np.mean(all_rmses), np.std(all_rmses)))

    # run_model(BrainAgeModel, 200)
    # run_model(LenetModel, 200)
    # run_model(VGGModel, 200)


if __name__ == '__main__':
    main()
