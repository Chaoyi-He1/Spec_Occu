# Data_Generation_Wireless_Signal_Experiment.m - MATLAB Code for 802.11 ax data generation code with multiple users

# 1. Project Name and Authors

This code was prepared for the project and associated paper titled "Stochastic Spectrum Prediction via Diffusion Probabilistic Models and Representation Learning" by Chaoyi He, Arya Menon, and Linda Katehi. At the time of this creation, all authors are associated with the Department of Electrical and Computer Engineering at Texas A&M Univerisity, College Station, TX. The project period is 2023-2024. Corresponding author is Linda Katehi (katehi@tamu.edu).

# 2. Project Summary
This project introduces a DNN-based spectrum sharing and prediction method which incorporates encoder-contrastive learning and a diffusion model or compressing historical wireless signal data and predicting future time signals, respectively. The method is validated in a dynamic 802.11ax environment with multiple users and realistic audio payload data.  

# 3. Summary of 802.11ax Data Generation 

The following are the main parameters of the MATLAB code Data_Generation_Wireless_Signal_Experiment.m: 
a) The center frequency is 5.25 GHz. The total system bandwidth is 80 MHz which consists of four 20 MHz frequency bands. If this is changed, then change the part of the code where the final allocation table is generated. 
b) Each 20 MHz frequency band can hold a max of 9 users. Spectrum allocation is assigned using 'allocation index' as defined in MATLAB. Permissiable allocation index values are embedded into the code.  
c) New users appear approximately every 1s using a Poisson distribution. Queue is serviced every one second. Total simulation time is set to 5s. The preamble time is included while determining if it is time for a queue to be serviced
d) All data is upsampled to 80 MHz sampling rate and preamble is removed from final data. 
d) Data is saved in baseband IQ format with I (real) and Q (imaginary) parts stored separately. Data is arranged into a matrix wihere each row corresponds to an OFDM symbol and successive rows denote passage of time. The label is concatinated to the data and represents the occupation of 10 MHz of the spectrum. There are 8 fields in the label  - 0 indicates the corresponding 10 MHz frequency band is unoccupied while 1 indicates that the 10 MHz frequency band is occupied. 

# 4. How to Run this Code 

a) This code was written in MATLAB R2021a and requires the Communication Toolbox and WLAN Toolbox installed. The code was prepared by Arya Menon in collaboration with Chaoyi He and Linda Katehi. 
b) Ensure that the following files/folders are available: (4.b.i) folder: binary_data_files with 114 .txt files (4.b.ii) data_list.mat(4.b.iii) allocation_table.m  (4.b.iv) Channel Model - Options.xlsx. Descriptions of each of these files/folders are available below.
c) In the MATLAB file Data_Generation_Wireless_Signal_Experiment.m, edit variable read_path (line 19) to provide the path to the folder binary_data_files.
d) Edit line 20 to load the matlab datafile data_list.mat.
e) Edit variables write_path and write_name to provide the path and file name for saving generated data files
d) Using Channel Model - Options.xlsx, edit the variables channel_model, num_walls, and LS_fading (lines 29-31).
e) Run the code. 

# 5. Descriptions of Associated Files

binary_data_files: A folder containing audio files converted binary format. These files serve as the payload for the wireless communication experiment. A copy of the files with the corresponding audio files is available at https://doi.org/10.18738/T8/B85VC9. 

data_list.mat: A MATLAB datafile containing the attributes and names of all files in the folder binary_data_files

allocation_table.m: A function that assigns spectrum resources in this experiment based on number of users. 

Channel Model - Options.xlsx: A excel sheet that lists valid channel models in the experiment. Suggested data file name for write_name variable is also provided in this excel sheet. 
