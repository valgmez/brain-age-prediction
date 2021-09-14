import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import keras
import csv
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
from skimage.transform import resize, rotate
from skimage.util import crop
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# code for Lengau cluster


def extract_data():
    # This function extracts the mris individually and puts them in a numpy array
    mri_array = []
    ages = []

    # path to highres mris on supercomputer
    data_dir = '/mnt/lustre/groups/HEAL0793/T1w_out'
    sr = pd.read_spss(
        'demographics.sav')  # demographics file

    # iterate over all the folders (one per scan), get the t1 mri, convert .nii into numpy and match the subject id with age in demographics
    for folder in sorted(os.listdir(data_dir)):
        if not os.path.isdir(data_dir + '/' + folder):
            pass
        elif str(folder) == '.anat':  # not interested in .anat folders with no subject id
            pass
        else:
            x = os.listdir(data_dir + '/' + str(folder))
            for t1 in x:
                # check if file T1_to_MNI_lin exists
                if re.search("T1_to_MNI_lin.nii", t1):
                    mri_file = t1
                    mri_img = nb.load(data_dir + '/' + folder + '/' + mri_file)
                    mri_img_data = mri_img.get_fdata()  # convert nii into numpy
                    mri_img_data = resize(
                        mri_img_data, (96, 112, 96))

                    # ---- Separating into subgroups
                    controls = sr[sr['patientcontrol'] == 'Control']
                    schizo = sr[(sr['demopatient'] == 'Schizophrenia')
                                & (sr['patientcontrol'] == 'Patient')]
                    ptsd = sr[(sr['demopatient'] == 'PTSD') & (
                        sr['patientcontrol'] == 'Patient')]
                    parkinsons = sr[(sr['demopatient'] == 'Parkinsons Disease') & (
                        sr['patientcontrol'] == 'Patient')]
                    patients = sr[sr['patientcontrol'] == 'Patient']

                    # ---- Change these variables to either controls, schizo, ptsd, parkinsons, patients to extract data for that group
                    sr = controls
                    sr.name = 'controls'

                    # --- Ignore .anat from folder name, we just want the subject id
                    folder = folder.rsplit('.', 1)[0]

                    # ---- Check if subject folder is in demographics file, skip to the next if not
                    if (sr[sr['eciinvestigator'].str.lower().replace('-', "").replace('_', "") == folder.lower().replace('-', "").replace('_', "")]).empty:
                        pass

                    else:
                        print(data_dir + '/' + folder + '/' + mri_file)
                        mri_array.append(mri_img_data)
                        mri_age = sr.loc[sr.eciinvestigator.str.lower().replace('-', "").replace('_', "") ==
                                         folder.lower().replace('-', "").replace('_', ""), 'calculated_age'].iloc[0]  # matching subject id with each MRI's ID to get the age
                        ages.append(round(mri_age))
                else:
                    pass

    print('Saving the numpy arrays...')
    np.save('/mnt/lustre/users/vgomezramirez/' + sr.name + '_ages.npy', ages)
    np.save('/mnt/lustre/users/vgomezramirez/' +
            sr.name + '_mris.npy', mri_array)


def predict_clinical(model, n_epochs):
    # --- load datasets, mris = X, ages = y
    # uncomment whichever group is needed for clinical
    mri_control = np.load('/mnt/lustre/users/vgomezramirez/controls_mris.npy')
    ages_control = np.load('/mnt/lustre/users/vgomezramirez/controls_ages.npy')

    mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/schizo_mris.npy')
    ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/schizo_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/ptsd_mris.npy')
    # ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/ptsd_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/parkinsons_mris.npy')
    # ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/parkinsons_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/patients_mris.npy')
    # ages_clinical = np.load(
    #     '/mnt/lustre/users/vgomezramirez/patients_ages.npy')

    # --- loading the BrainAgeModel and freezing the first 17 layers
    model = load_model('models/' + model)

    for layer in model.layers[:17]:
        layer.trainable = False

    model.summary()

    # --- saving images of healthy and diseased brains for comparison
    plt.axis('off')
    plt.imsave('mri_healthy.png', mri_control[10, :, :,
                                              48].T, cmap='gray', origin='lower')
    print("Age: ", ages_control[10])

    plt.axis('off')
    plt.imsave('mri_pathology.png', mri_clinical[10, :, :,
                                                 48].T, cmap='gray', origin='lower')
    print("Age: ", ages_clinical[10])

    # --- defining train (controls) and test (clinical) sets
    X_train = mri_control
    y_train = ages_control
    X_test = mri_clinical
    y_test = ages_clinical

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("y_test shape: " + str(y_test.shape))

    print('Reshaping data...')
    # X_train = X_train.reshape(-1, 143, 179, 143, 1)
    # X_test = X_test.reshape(-1, 143, 179, 143, 1)
    X_train = X_train.reshape(-1, 96, 112, 96, 1)
    X_test = X_test.reshape(-1, 96, 112, 96, 1)

    # --- folder to save model and plots, change schizo to desired one
    folder_name = "models/schizo/" + str(n_epochs) + '_' + model.name

    # # --- 2. compile model
    print('Compiling the model...')
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mae', rmse])

    # stops training early when there is no improvement in the validation loss for 10 consecutive epochs
    early = EarlyStopping(monitor='loss', patience=10, verbose=1)

    # --- 3. train the model and plot results
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, epochs=n_epochs,
                        batch_size=4, validation_data=(X_test, y_test), callbacks=[early])

    # --- 4. predict
    print('Making predictions...')
    y_pred = model.predict(X_test)
    print('Mean Absolute Error (predictions): ',
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
    print('MAE after correction (healthy): ',
          (np.abs(corrected_age_train - y_train)).mean())
    print('MAE after correction (patients): ',
          (np.abs(corrected_age_test - y_test)).mean())

    mae_train = (np.abs(corrected_age_train - y_train)).mean()
    mae_test = (np.abs(corrected_age_test - y_test)).mean()
    return mae_train, mae_test, y_train, yhat_train, corrected_age_train, y_test, yhat_test, corrected_age_test, folder_name


def rmse(y_true, y_pred):
    '''calculates the root mean squared error between labels and predictions '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def main():
    extract_data()
    all_maes_train = []
    all_maes_test = []
    all_y_train = []
    all_yhat_train = []
    all_corrected_age_train = []
    all_y_test = []
    all_yhat_test = []
    all_corrected_age_test = []

    for i in range(10):
        mae_train, mae_test, y_train, yhat_train, corrected_age_train, y_test, yhat_test, corrected_age_test, folder_name = predict_clinical(
            'BrainAgeModel-camcan', 200)
        all_maes_train.append(mae_train)
        all_maes_test.append(mae_test)
        all_y_train.append(y_train)
        all_yhat_train.append(yhat_train)
        all_corrected_age_train.append(corrected_age_train)
        all_y_test.append(y_test)
        all_yhat_test.append(yhat_test)
        all_corrected_age_test.append(corrected_age_test)

    print('Average MAE healthy for 10 repetitions: %.4f +/- %.3f' %
          (np.mean(all_maes_train), np.std(all_maes_train)))
    print('Average MAE clinical 10 repetitions: %.4f +/- %.3f' %
          (np.mean(all_maes_test), np.std(all_maes_test)))

    dftrain = pd.DataFrame({"y_train": np.mean(all_y_train, axis=0), "yhat_train": np.mean(
        all_yhat_train, axis=0), "corrected_age_train": np.mean(all_corrected_age_train, axis=0)})
    dftrain.to_csv(folder_name + '_results_train.csv', index=False)
    dftest = pd.DataFrame({"y_test": np.mean(all_y_test, axis=0), "yhat_test": np.mean(
        all_yhat_test, axis=0), "corrected_age_test": np.mean(all_corrected_age_test, axis=0)})
    dftest.to_csv(folder_name + '_results_test.csv', index=False)


if __name__ == '__main__':
    main()
