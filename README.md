# CS598 DL4H Project: Reproduction of "Real-world Patient Trajectory Prediction from Clinical Notes Using Artiﬁcial Neural Networks and UMLS-Based Extraction of Concepts"
---------------------

Contents
---------------------
 * Selected Paper
 * Reproduction Steps
 * Results
 * References

Group Members
------------

 * Nicholas DaRosa (ndarosa2@illinois.edu)

Selected Paper
------------

The research paper selected for reproduction was:  

Zhang, D., Yin, C., Zeng, J. et al. Combining structured and unstructured data for predictive models: a deep learning approach. BMC Med Inform Decis Mak 20, 280 (2020). https://doi.org/10.1186/s12911-020-01297-6 

Github link: https://github.com/JamilProg/patient_trajectory_prediction

Reproduction Steps
------------

The original paper's README.md provides reproduction steps; however, more detailed steps along with possible necesary troubleshooting and additional needed packages that are not listed in the original paper's README.md are provided below: 
 * Environment
 	* Python 3.7
 	* Cuda version 10.2
 		* If Cuda is not used to accelerate training, then the Pytorch scripts need to be edited.
 	* PyTorch version 1.5.0
	* spaCy
	* dill
	* nltk
	* cuda / cudatoolkit
	* scikitlearn
	* matplotlib

 * Steps
 	1. Download the original paper's repository [1]
 	2. Apply for UMLS license through your research organization. Typically takes a couple business days for approval [2].
 	3. Get access to, download, and uncompress the MIMIC-III dataset [4].
 	4. Download the QuickUMLS repository [5]
 	5. Copy MIMIC's NOTEEVENTS.csv into data_cleaning folder
	6. Run noteEvents_preproc.py
	* Example commands: 
		* cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/data_cleaning	
		* python noteEvents_preproc.py NOTEEVENTS.csv
 	

Copy MIMIC's NOTEEVENTS.csv into data_cleaning folder

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/data_cleaning

python noteEvents_preproc.py NOTEEVENTS.csv

python MIMIC_smart_splitter.py output.csv

Download the full release of UMLS here https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

Unzip UMLS folder umls-2021AB-full

Navigate to umls-2021AB-full/2021AB-full and unzip mmsys.zip

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/umls-2021AB-full/2021AB-full/mmsys

Unzip the files into the same directory https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/umls-2021AB-full/2021AB-full

./run_linux.sh

Selected Level 0 for subset

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/concept_annotation/QuickUMLS-master

python setup.py install

Create a QuickUMLS installation 
-MRCONSO.RRF and MRSTY.RRF are in the META folder under 2021AB 
-python -m quickumls.install <umls_installation_path> <destination_path>
-With actual file paths 
python -m quickumls.install /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthpython 02_FFNcare/Project/Paper_111/patient_trajectory_prediction-master/concept_annotation/2021AB/META /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/concept_annotation/testInstall

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/concept_annotation

python quickUMLS_getCUI.py --t=0.9 --TUI=Alpha

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/concept_annotation/data/outputchunkssmall

### Command to concatenate the outputs from 1.csv to 71.csv [QuickUMLS AND MetaMap] :
cat 1.csv.output 2.csv.output 3.csv.output 4.csv.output 5.csv.output 6.csv.output 7.csv.output 8.csv.output 9.csv.output 10.csv.output 11.csv.output 12.csv.output 13.csv.output 14.csv.output 15.csv.output 16.csv.output 17.csv.output 18.csv.output 19.csv.output 20.csv.output 21.csv.output 22.csv.output 23.csv.output 24.csv.output 25.csv.output 26.csv.output 27.csv.output 28.csv.output 29.csv.output 30.csv.output 31.csv.output 32.csv.output 33.csv.output 34.csv.output 35.csv.output 36.csv.output 37.csv.output 38.csv.output 39.csv.output 40.csv.output 41.csv.output 42.csv.output 43.csv.output 44.csv.output 45.csv.output 46.csv.output 47.csv.output 48.csv.output 49.csv.output 50.csv.output 51.csv.output 52.csv.output 53.csv.output 54.csv.output 55.csv.output 56.csv.output 57.csv.output 58.csv.output 59.csv.output 60.csv.output 61.csv.output 62.csv.output 63.csv.output 64.csv.output 65.csv.output 66.csv.output 67.csv.output 68.csv.output 69.csv.output 70.csv.output 71.csv.output > concatenated_output.csv

Copy quickumls_processing.py into the outputchunkssmall folder 

python quickumls_processing.py concatenated_output.csv

Copy post_process_output.csv into Pytorch_scripts/diagnoses_prediction

Copy ADMISSIONS.csv and DIAGNOSES_ICD.csv into Pytorch_scripts/diagnoses_prediction

cd /home/nick/Documents/UIUC/CS_598_Deep_Learning_for_Healthcare/Project/Paper_111/patient_trajectory_prediction-master/PyTorch_scripts/diagnoses_prediction

python 01_data_preparation.py --admissions_file ADMISSIONS.csv --diagnoses_file DIAGNOSES_ICD.csv --notes_file post_processed_output_alpha_09.csv

That creates prepared_data.npz 

Run the model for diagnoses prediction using a feed forward network
python 02_FFN_diagprediction.py --inputdata=prepared_data.npz --nEpochs=50 --kFold=1 

References
------------
1. Zhang, D., Yin, C., Zeng, J. et al. Combining structured and unstructured data for predictive models: a deep learning approach. BMC Med Inform Decis Mak 20, 280 (2020). https://doi.org/10.1186/s12911-020-01297-6 Github link: https://github.com/JamilProg/patient\_trajectory\_prediction

2. Unified medical language system (UMLS). https://www.nlm.nih.gov/research/umls/index.html, accessed April, 2022

3. Metathesaurus’s unique identifiers. https://www.nlm.nih.gov/research/umls/new users/online learning/Meta 005.html, accessed April, 2022

4. MIT’s MIMIC-III. https://mimic.physionet.org/, accessed April, 2022

5. Quickumls github link. https://github.com/Georgetown-IR-Lab/QuickUMLS, Accessed April, 2022
