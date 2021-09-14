The Cam-CAN data is stored on the Supercomputing Wales cluster; SharedRoots is stored in the CHPC cluster.
Code included here is for inspection purposes.

· brain_age_prediction_testing.ipynb is the GoogleColab notebook with initial experimentation
· data_distributions_and_analysiss.ipynb is the GoogleColab notebook analysing the demographics in each dataset
· brain-age.py contains the code for training and testing on the Cam-CAN dataset with the BrainAgeModel, VGG and Lenet. Has to be run on the Supercomputing Wales cluster 
· brain-age-trainingonly.py contains the code for training and saving the model's weights, to be run on the Supercomputing Wales cluster
· brain-age-clinical.py contains the code for applying the model through transfer learning to SharedRoots (the clinical dataset), has to be run on the CHPC cluster and requires the 'BrainAgeModel-camcan' folder
· graphs.ipynb analyses of the results from clinical data

The folder raw results clinical contains all the results produced from testing on the clinical dataset. 
 