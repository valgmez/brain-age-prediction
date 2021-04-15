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
from skimage.transform import resize, rotate
from skimage.util import crop
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# code for Lengau cluster


def extract_data():
    # This function extracts the highres mris individually and puts them in a numpy array
    mri_array = []
    ages = []
    # path to highres mris on supercomputer
    data_dir = '/mnt/lustre/groups/HEAL0793/T1_nifti'
    sr = pd.read_spss(
        'demographics.sav')  # ages file

    for folder in sorted(os.listdir(data_dir)):
        if not os.path.isdir(data_dir + '/' + folder):
            pass
        else:
            x = os.listdir(data_dir + '/' + str(folder))
            # scan through subfolders for each patient and choose the one starting with 1_00
            gm = [gm for gm in x if re.search("_00", gm)][0]
            x = os.listdir(data_dir + '/' + str(folder) + '/' + gm + '/')
            mri_file = [mri for mri in x if re.search(
                "nii", mri)][0]  # find the nii file
            mri_img = nb.load(data_dir + '/' + folder +
                              '/' + gm + '/' + mri_file)
            mri_img_data = mri_img.get_fdata()  # convert nii into numpy

            # ---- Cropping and resizing
            mri_img_data = crop(mri_img_data, 19)
            mri_img_data = resize(
                mri_img_data, (96, 112, 96))
            mri_img_data = rotate(mri_img_data, 180)

            # mri_array.append(mri_img_data)

            # ---- Separating into subgroups
            controls = sr[sr['patientcontrol'] == 'Control']
            schizo = sr[(sr['demopatient'] == 'Schizophrenia')
                        & (sr['patientcontrol'] == 'Patient')]
            ptsd = sr[(sr['demopatient'] == 'PTSD') & (
                sr['patientcontrol'] == 'Patient')]
            parkinsons = sr[(sr['demopatient'] == 'Parkinsons Disease') & (
                sr['patientcontrol'] == 'Patient')]
            patients = sr[sr['patientcontrol'] == 'Patient']

            # ---- Change these variables for either controls, schizo, ptsd, parkinsons to extract data for that group
            sr = controls
            sr.name = 'controls'

            # ---- Check if folder subject is in demographics file, skip to the next if not
            if (sr[sr['eciinvestigator'].str.lower().replace('-', "").replace('_', "") == folder.lower().replace('-', "").replace('_', "")]).empty:
                pass

            else:
                print(data_dir + '/' + folder + '/' + gm + '/' + mri_file)
                mri_array.append(mri_img_data)
                mri_age = sr.loc[sr.eciinvestigator.str.lower().replace('-', "").replace('_', "") ==
                                 folder.lower().replace('-', "").replace('_', ""), 'calculated_age'].iloc[0]  # convert demographics file into .npy by selecting all the ages where the eciinvestigator matches the list of MRIs and adding them to a list which gets converted into .npy
                print(round(mri_age))
                ages.append(round(mri_age))

    print('Saving the numpy arrays...')
    np.save('/mnt/lustre/users/vgomezramirez/' + sr.name + '_ages.npy', ages)
    np.save('/mnt/lustre/users/vgomezramirez/' +
            sr.name + '_mris.npy', mri_array)


def predict_clinical(model, n_epochs):
    # --- load datasets, mris = X, ages = y
    mri_control = np.load('/mnt/lustre/users/vgomezramirez/controls_mris.npy')
    ages_control = np.load('/mnt/lustre/users/vgomezramirez/controls_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/schizo_mris.npy')
    # ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/schizo_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/ptsd_mris.npy')
    # ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/ptsd_ages.npy')

    # mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/parkinsons_mris.npy')
    # ages_clinical = np.load('/mnt/lustre/users/vgomezramirez/parkinsons_ages.npy')

    mri_clinical = np.load('/mnt/lustre/users/vgomezramirez/patients_mris.npy')
    ages_clinical = np.load(
        '/mnt/lustre/users/vgomezramirez/patients_ages.npy')

    # --- loading the BrainAgeModel and setting the freezing the first layers
    model = load_model('models/' + model)

    for layer in model.layers[:15]:
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

    # --- reshaping data into (96,112,96)
    print('Reshaping data...')
    # X_train = X_train.reshape(-1, 143, 179, 143, 1)
    # X_test = X_test.reshape(-1, 143, 179, 143, 1)
    X_train = X_train.reshape(-1, 96, 112, 96, 1)
    X_test = X_test.reshape(-1, 96, 112, 96, 1)

    # --- folder to save model and plots
    folder_name = "models/" + str(n_epochs) + '_' + model.name

    # # --- 2. compile model
    print('Compiling the model...')
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mae', rmse])

    # stops training early when there is no improvement in the validation loss for 10 consecutive epochs
    early = EarlyStopping(monitor='mae', patience=20, verbose=1)

    # --- 3. train the model and plot results
    print('Training the model...')
    history = model.fit(x=X_train, y=y_train, epochs=n_epochs,
                        batch_size=4, validation_data=(X_test, y_test), callbacks=[early])

    print('Plotting results...')
    plot_results(history, folder_name)

    # --- 5. predict
    print('Making predictions...')
    y_pred = model.predict(X_test)
    print('Mean Absolute Error (predictions): ',
          mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error (predictions): ',
          np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2: ', r2_score(y_test, y_pred))

    plt.figure()
    plt.scatter(y_test, y_pred, label='Prediction')
    y_range = np.arange(np.min(y_test), np.max(y_test))
    plt.plot(y_range, y_range, c='black', ls='dashed', label='45 deg line')
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Prediction with transfer learning SharedRoots')
    plt.legend()
    plt.savefig(folder_name + '_r2.png')

    mae = mean_absolute_error(y_test, y_pred)
    root_mse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, root_mse


def rmse(y_true, y_pred):
    '''calculates the root mean squared error between labels and predictions '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


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
    # extract_data()
    predict_clinical('BrainAgeModel-camcan', 200)


if __name__ == '__main__':
    main()
