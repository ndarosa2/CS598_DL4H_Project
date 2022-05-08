# CS598 DL4H Project: Reproduction of "Real-world Patient Trajectory Prediction from Clinical Notes Using Artiﬁcial Neural Networks and UMLS-Based Extraction of Concepts"
---------------------

Contents
---------------------
 * Group Members
 * Selected Paper
 * Dependencies
 * Data Download Instructions
 * Data Preprocessing Instructions
 * Model Training and Evaluation Instructions - Diagnoses Prediction
 * Model Training and Evaluation Instructions - Mortality Prediction
 * Model Training and Evaluation Instructions - Readmission Prediction
 * Results
 * References

Group Members
------------

 * Nicholas DaRosa (ndarosa2@illinois.edu)

Selected Paper
------------

The research paper selected for reproduction was:  

Zaghir, Jamil & Rodrigues Jr, Jose & Goeuriot, Lorraine & Amer-Yahia, Sihem. (2021). Real-world Patient Trajectory Prediction from Clinical Notes Using Artificial Neural Networks and UMLS-Based Extraction of Concepts. Journal of Healthcare Informatics Research. 5. 10.1007/s41666-021-00100-z. 

GitHub link: https://github.com/JamilProg/patient_trajectory_prediction

Dependencies
------------

* Python 3.7
* Cuda version 10.2
 	* If a GPU that is CUDA enabled is not used to accelerate training, then the Pytorch scripts need to be edited to not use a GPU; however, training solely on CPU is not feasible due to the computation time and so is not recommended. 
* cuda / cudatoolkit 11.3.1
* dill 0.3.4
* matplotlib 3.5.1
* nltk 3.7
* numpy 1.19.2
* PyTorch version 1.5.0
* scikit-learn 1.0.2
* scipy 1.6.2
* spaCy 3.2.1 (needed for QuickUMLS)
	
Data Download Steps
------------	

 1. Apply for a UMLS license through your research organization. Access is granted by the National Library of Medicine. Typically takes a couple business days for approval. Once access has been granted, download the full release of UMLS (5.1 GB compressed) [2].
 2. Get access to, download, and uncompress the MIMIC-III dataset (6.6 GB compressed). Access can be granted through Physionet [4]. 

Data Preprocessing Instructions 
------------
The original paper's GitHub has a that README.md provides data processing reproduction steps; however, more detailed steps along with possible necessary troubleshooting are provided below: 

 1. Download this GitHub's repository. Other than a couple files, this repository hosts the same files found in the original paper's repository (patient_trajectory_prediction), but with minor script modifications and with some files moved to a different folder. 
 	* Modifications included: added lines to calculate total parameters, added with.torch_nograd() to some files to save memory and increase speed, and changed torch.cuda.set_device(1) to torch.cuda.set_device(0) since only tested with one active GPU. 
 	* Alternatively, download the original paper's repository (patient_trajectory_prediction) instead [1]
 3. Download the QuickUMLS repository (QuickUMLS) [5]
 4. In patient_trajectory_prediction, do the following
 	* Copy MIMIC's NOTEEVENTS.csv into the data_cleaning folder
 	* Change directory to data_cleaning 
	* Run noteEvents_preproc.py. This will generate output.csv
		* python noteEvents_preproc.py NOTEEVENTS.csv
		*  According to the original paper's GitHub it takes about 4 hours to finish, but in reproduction testing it took approximately 2 hours.
	* Run python MIMIC_smart_splitter.py. Splits the preprocessed text into files of 50 Mb and puts those file in a new folder called data. 
		* python MIMIC_smart_splitter.py output.csv
		*  According to the original paper's GitHub it takes about 1 hour to finish, but in reproduction testing it took approximately 20 minutes.
4. Install UMLS
	* Unzip UMLS folder umls-2021AB-full
	* Follow the directions provided at https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html [2]
		* If on Linux, need to run the command ./run_linux.sh	
		* Select Level 0 for Subset

5. Install QuickUMLS. After installation, there should now be a QuickUMLS folder with the subfolders cui-semtypes.db, umls-sumstring.db, language.flag, and lowercase.flag
	* Follow the directions provided at QuickUMLS GitHub https://github.com/Georgetown-IR-Lab/QuickUMLS [5]
	* Note that MRCONSO.RRF and MRSTY.RRF are in the META folder under 2021AB 
	* Example install commandL
		* python -m quickumls.install <umls_installation_path> <destination_path> 	
6. Copy the data folder generated in step 3 and the QuickUMLS folder generated in Step 4 into the patient_trajectory_prediction/concept_annotation folder.
7. Change directory to the concept_annotation 
8. Run the quickUMLS_getCUI.py for each dataset parameters. This script generates approximately 70 .csv.output files in the outputchunkssmall folder.  
	* The parameters are:
		* --t : Float that is the QuickUMLS similarity Threshold. It is between 0 and 1. Default is 0.9. 
    		* --TUI : String that represents the TUI List filter. It is either "Alpha" or "Beta". Default is Beta.
    	* Example command:
    		* python quickUMLS_getCUI.py --t=0.9 --TUI=Alpha
    			* The generation of this dataset took approximately 8 hours using a 32 thread CPU and required around 70GB of RAM. 
9. Since quickUMLS_getCUI.py generated about 70 .csv.output files in the outputchunkssmall folder, change directory to data/outputchunkssmall and run the following command to concatenate the files and produce concatenated_output.csv. 
	* cat 1.csv.output 2.csv.output 3.csv.output 4.csv.output 5.csv.output 6.csv.output 7.csv.output 8.csv.output 9.csv.output 10.csv.output 11.csv.output 12.csv.output 13.csv.output 14.csv.output 15.csv.output 16.csv.output 17.csv.output 18.csv.output 19.csv.output 20.csv.output 21.csv.output 22.csv.output 23.csv.output 24.csv.output 25.csv.output 26.csv.output 27.csv.output 28.csv.output 29.csv.output 30.csv.output 31.csv.output 32.csv.output 33.csv.output 34.csv.output 35.csv.output 36.csv.output 37.csv.output 38.csv.output 39.csv.output 40.csv.output 41.csv.output 42.csv.output 43.csv.output 44.csv.output 45.csv.output 46.csv.output 47.csv.output 48.csv.output 49.csv.output 50.csv.output 51.csv.output 52.csv.output 53.csv.output 54.csv.output 55.csv.output 56.csv.output 57.csv.output 58.csv.output 59.csv.output 60.csv.output 61.csv.output 62.csv.output 63.csv.output 64.csv.output 65.csv.output 66.csv.output 67.csv.output 68.csv.output 69.csv.output 70.csv.output 71.csv.output > concatenated_output.csv

10. Copy quickumls_processing.py from the concept_annotation folder into the outputchunkssmall folder and run the following command:
	* python quickumls_processing.py concatenated_output.csv
	* This command generates post_process_output.csv

11. Copy post_process_output.csv,ADMISSIONS.csv (MIMIC-III file), and DIAGNOSES_ICD.csv (MIMIC-III file) into patient_trajectory_prediction/Pytorch_scripts/diagnoses_prediction, patient_trajectory_prediction/Pytorch_scripts/mortality_prediction, and patient_trajectory_prediction/Pytorch_scripts/readmission_prediction

12. Change directory to patient_trajectory_prediction-master/PyTorch_scripts/diagnoses_prediction and run the following command to prepare the data for diagnoses prediction:
	* python 01_data_preparation.py --admissions_file ADMISSIONS.csv --diagnoses_file DIAGNOSES_ICD.csv --notes_file post_processed_output.csv
	* This command creates prepared_data.npz which is the data used in model training and testing. 
13. Change directory to patient_trajectory_prediction-master/PyTorch_scripts/mortality_prediction and run the following command to prepare the data for mortality prediction:
	* python 01_data_prep_mortality.py --admissions_file ADMISSIONS.csv --diagnoses_file DIAGNOSES_ICD.csv --notes_file post_processed_output_beta_09.csv
	* This command creates prepared_data.npz and prepared_data_deathTime.npz which are used in model training and testing. 
14. Change directory to patient_trajectory_prediction-master/PyTorch_scripts/readmission_prediction and run the following command to prepare the data for readmission prediction:
	* python 01_data_prep_readmission.py --admissions_file ADMISSIONS.csv --diagnoses_file DIAGNOSES_ICD.csv --notes_file post_processed_output.csv
	* This command creates prepared_data.npz which is the data used in model training and testing. 
15. To generate the other datasets steps, repeat Steps 7-14. The datasets are as follows:
	* Dataset A: Theshold is 0.7 and using TUI list Alpha
	* Dataset B: Theshold is 0.7 and using TUI list Beta
	* Dataset C: Theshold is 0.9 and using TUI list Alpha
	* Dataset D: Theshold is 0.9 and using TUI list Beta
16. Note that Datasets A,B,C are only used for feed forward network diagnoses prediction. The primary dataset used is Dataset D (is used for diagnoses, mortality, and readmission, prediction). 
 
Model Training and Evaluation Instructions - Diagnoses Prediction
------------
The original paper's GitHub has a README.md that provides model training and evaluation reproduction steps for diagnoses prediction; however, more detailed steps are provided below: 

1. Using a feed forward network (FFN), train and evaluate the models for diagnoses prediction only using the CUI datasets:
	*  time python 02_FFN_diagprediction.py --inputdata=prepared_data.npz --nEpochs=5000 --kFold=5 withCCS=0
	*  Repeat command for Datasets A,B,C,D
2. Using a feed forward network (FFN) architecture, train and evaluate the models for diagnoses prediction only using the CCS datasets:
	*  time python 02_FFN_diagprediction_CCS_only.py --inputdata=prepared_data.npz --nEpochs=5000 --kFold=5 withCCS=1
3. Using a feed forward network (FFN), train and evaluate the models for diagnoses prediction using both CUI data and the CCS data:
	*  time python 02_FFN_diagprediction.py --inputdata=prepared_data.npz --nEpochs=5000 --kFold=5 withCCS=1

4.  Using the gated recurrent unit (GRU) architecture, train and evaluate the models for diagnoses prediction using CUI data only:
	*  time python 02_GRU_train_GPU.py --inputdata=prepared_data.npz --nEpochs=1500 --withCCS=0
	*  time python 03_GRU_test.py --Xinputdata=X-test.data --Yinputdata=Y-test.data --inputModel=model_output.pt
5.  Using the gated recurrent unit (GRU) architecture, train and evaluate the models for diagnoses prediction using both CUI and CCS data:
	*  time python 02_GRU_train_GPU.py --inputdata=prepared_data.npz --nEpochs=1500 --withCCS=1
	*  time python 03_GRU_test.py --Xinputdata=X-test.data --Yinputdata=Y-test.data --inputModel=model_output.pt

6. Using the long short term memory (LSTM) architecture, train and evaluate the models for diagnoses prediction using CUI data only:
	*  time python 02_LSTM_train_GPU.py --inputdata=prepared_data.npz --nEpochs=1500 --withCCS=0
	*  time python 03_LSTM_test.py --Xinputdata=X-test.data --Yinputdata=Y-test.data --inputModel=model_output.pt

 
Model Training and Evaluation Instructions - Mortality Prediction
------------
The original paper's GitHub has a README.md that provides model training and evaluation reproduction steps for mortality prediction; however, more detailed steps are provided below: 

Note: All runs for mortality prediction are done using Dataset D. 
1. Using the FFN architecture, train and evaluate the models for mortality prediction only using the CUI datasets:
	* time python 02_FFN_mortality.py --inputdata=mort_prepared_data_beta_09.npz --inputdata2=mort_prepared_data_deathTime_beta_09.npz --nEpochs=5000 --kFold=5 --lr=0.01 --withCCS=1
2.  Using the GRU architecture, train and evaluate the models for diagnoses prediction using CUI data only:
	* time python 02_GRU_mortality.py --inputdata=prepared_data.npz --inputdata2=prepared_data_deathTime.npz --nEpochs=1500 --kFold=5 --withCCS=0
	
3.  Using the GRU architecture, train and evaluate the models for mortality prediction using CCS data only:
	* time python 02_GRU_mortality_CCS_only.py --inputdata=prepared_data.npz --inputdata2=prepared_data_deathTime.npz --nEpochs=1500 --kFold=5 --withCCS=1

4.  Using the GRU architecture, train and evaluate the models for mortality prediction using both CUI and CCS data:
	* time python 02_GRU_mortality.py --inputdata=prepared_data.npz --inputdata2=prepared_data_deathTime.npz --nEpochs=1500 --kFold=5 --withCCS=1

 
Model Training and Evaluation Instructions - Readmission Prediction
------------
The original paper's GitHub has a README.md that provides model training and evaluation reproduction steps for readmission prediction; however, more detailed steps are provided below: 

Note: All runs for mortality prediction are done using Dataset D.

1. Using the FFN architecture, train and evaluate the models for readmission prediction only using the CUI data:
	* time python 02_FFN_readmission.py --inputdata=prepared_data.npz --withCCS=0
2. Using the FFN architecture, train and evaluate the models for readmission prediction only using the CCS data:
	* time python 02_FFN_readmission_CCS_only.py --inputdata=prepared_data.npz --withCCS=1
3. Using the FFN architecture, train and evaluate the models for readmission prediction only using both the CUI and CCS data:
	* time python 02_FFN_readmission.py --inputdata=prepared_data.npz --withCCS=1
4. Using the GRU architecture, train and evaluate the models for mortality prediction using both CUI and CCS data:
	* time python 02_GRU_readmission.py --inputdata=prepared_data.npz --withCCS=1

Results
------------

Raw results and completion times are provided in the text file Raw Results. 

Provided below are tables summarizing the reproduction results. Discussions of these results can be found in the associated reproduction paper.

Table 1: The dataset configurations used in the original
and reproduction experiments.
![image](https://user-images.githubusercontent.com/90882619/167284776-e0b52db9-bb89-4454-bf62-0dd411a00ed6.png)

Table 2: Diagnoses prediction metrics Precision@, Re-call@, and AUC-ROC computed over the four dataset configurations using a Feed-Forward network (5000 epochs, 5 fold cross validation). Reproduction results are compared to original paper results.
![image](https://user-images.githubusercontent.com/90882619/167320402-4a3ee975-6dff-4405-99a7-a7455e7b23ae.png)

Table 3: Metric comparison for inputs CUI codes only, CCS codes only, and CUI codes combined with CCS codes for diagnoses  prediction using the FFN model with default hyperparameters (e.g. 5,000 epochs, hidden size of 10,000). Since Dataset D performed the best in previous experiments, it was used for all runs shown.
![image](https://user-images.githubusercontent.com/90882619/167320319-4b54a7ba-f927-4ace-b20c-329642f34906.png)

References
------------
1. Zaghir, Jamil & Rodrigues Jr, Jose & Goeuriot, Lorraine & Amer-Yahia, Sihem. (2021). Real-world Patient Trajectory Prediction from Clinical Notes Using Artificial Neural Networks and UMLS-Based Extraction of Concepts. Journal of Healthcare Informatics Research. 5. 10.1007/s41666-021-00100-z. 
GitHub link: https://github.com/JamilProg/patient_trajectory_prediction

2. Unified medical language system (UMLS). https://www.nlm.nih.gov/research/umls/index.html, accessed April, 2022. Installation instructions https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html

3. Metathesaurus’s unique identifiers. https://www.nlm.nih.gov/research/umls/new users/online learning/Meta 005.html, accessed April, 2022

4. MIT’s MIMIC-III. https://mimic.physionet.org/, accessed April, 2022

5. Quickumls github link. https://github.com/Georgetown-IR-Lab/QuickUMLS, Accessed April, 2022
