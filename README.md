# brain-age-prediction
Final year project - deep learning CNNs to predict brain age 

· brain_age_prediction_testing.ipynb is the GoogleColab notebook with initial experimentation

· data_distributions_and_analysiss.ipynb is the GoogleColab notebook analysing the demographics in each dataset

· brain-age.py contains the code for training and testing on the Cam-CAN dataset with the BrainAgeModel, VGG and Lenet. Has to be run on the Supercomputing Wales cluster 

· brain-age-trainingonly.py contains the code for training and saving the model's weights, to be run on the Supercomputing Wales cluster

· brain-age-clinical.py contains the code for applying the model through transfer learning to SharedRoots (the clinical dataset), has to be run on the CHPC cluster and requires the 'BrainAgeModel-camcan' folder

· graphs.ipynb analysis of the results from clinical data
