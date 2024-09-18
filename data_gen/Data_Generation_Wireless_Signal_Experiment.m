%% This code generates the data for the wireless communication experiment described in the paper
%"Stochastic Spectrum Prediction via Diffusion Probabilistic Models and
%Representation Learning" by Arya Menon in collaboration with Chaoyi He,
%and Linda Katehi at Texas A&M University. 
% Code development date: June 2023. 

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% --- Please view the read me file before running the experiment. --------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

clc
clear all
close all
%% Edit this section before running the experiment

% Define read paths and load the file listing the names of all data files
read_path = 'C:\Users\binary_data_files\';
load('C:\Users\data_list.mat')
files_available = length(list);
disp('User data files read')

%Defining write path and file name
write_path = 'C:\Users\';
write_name = 'train_1'; %Enter name of file here

% Define Channel Model - see excel file Channel Model - Options.xlsx 
channel_model = 'Model-A'; %Typical indoor office channel model - predefined in the standard and available in MATLAB
num_walls = 0; %Number of walls in channel model
LS_fading = 'Pathloss'; %Large Scale Fading effect. Options are 'Pathloss', and 'Pathloss and shadowing'


%% ------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------- Experiment starts from here -----------------------------
% ------------------------------------------------------------------------
%-------------------------------------------------------------------------


%%  Setting up queue parameters ---

lambda_cycle = 5; % Period of lambda variation in seconds
sim_time = ceil(lambda_cycle); %Simulation time in seconds as a function of lambda_cycle
x = (0:1:sim_time)*180/lambda_cycle; %Defining x value for generating time-varying lambda (for Poisson Distribution)
lambda = floor(10+10*abs(sind(x))); %Poisson variable lambda

%% Setting up service strucuture. This structure monitors users currently being serviced
% Users are assigned to channels on a first in - first out basis
service = struct; % Defining a structure to monitor service parameters
service.user_name = []; %Name of user in the R.B
service.user_file = strings(1); %File name to be transmitted
service.user_file_bits = []; % Total bits to be transmitted
service.user_file_bits_tx = []; % number of bits already transmitted
service.MCS = []; % Modulation and coding scheme of user
service.APEP = []; % APEP Length for the user. 
max_MCS = 6; % maximum MCS values allowed. Note: Make sure that SNR values are compatible
MCS_del = 2; % This constant value gets rid of lower MCS values
APEP_options = [1000, 1500]; %Available APEP length options for user

%% Waveform and Channel Details
fs_max = 80e6; % All data will be upsampled to this common sampling rate
max_users = 36; % Max number of users that be serviced at a time depending on max BW
numPackets = 1; %Number of packets to be transmitted in one iteration
Ts = 1/fs_max; % Sampling time 
ch_BW = 'CBW80'; % Specifying maximum channel bandwidth (same as above but this required for creating the channel
fc = 5.25e9; % Carrier frequency - required for channel modeling 
GI = 3.2; % Guard interval in OFDM


TX_RX_distance_min = 2; % Minimum transmitter receiver distance
TX_RX_distance_max = 5; % Maximum transmitter receiver distance
SNR_min = 20; %Minimum SNR in dB
SNR_max = 25; %Maximum SNR in dB

f_OFDM_spacing = 78.125e3; % subcarrier spacing in 802.11ax
OFDM_symbol_time = 1/f_OFDM_spacing; %OFDM symbol time
OFDM_frame_length = OFDM_symbol_time*fs_max; % calculating the number of samples that correspond to one OFDM symbol
CP_frame_length = GI*(10^-6)*fs_max; % calculating the number of samples that correspond to Cyclic Prefix
Net_frame_length = OFDM_frame_length+CP_frame_length;
data_frame =[]; % This variable stores desired data as a complex number that is not normalized
label_frame = []; %This variable stores the desired label which is spectrum occupacation in 10 MHz steps. 0 = 10MHz not occupied

%% Defining variables that holds values for the next packet
% TX = struct;
TX_Psdu = {}; % Bits to be transmitted in next packet. This is a should be a vector cell array as per matlab rules
TX_Length = []; %Number of bits to be transmitted in the next packet

%% Control and Debugging variables 
time_calc = 0; % This variable is used in the loop to track the time elapsed. This is calculated based on the full waveform (header+ CP+ data)
service_track = {}; % This variable will be used to track the service user name for debugging purposes
iteration_number = 0; %This variable will keep track of the number of times a packet was generated in the simulation

%% Defining Queue Parameters and Access Rule 

queue = []; %Defining an empty variable to represent the queue. Each user that is added to the queue gets a unique name
seconds_track = 0; % This value is an integer. When this number changes, the queue is updated. This number also determines lambda
%  Initial Queue Generation 
lambda_val = lambda(1); %Get current lambda value
user_track = 0; %Most recent user number 
% --- Generate initial users ------
new_users = poissrnd(lambda_val); % Calculate number of new users
while (new_users == 0) % Removes the case for zero initial users. Zero new users are allowed in the rest of the code
    new_users = poissrnd(lambda_val); % Calculate number of new users
end
if new_users > 0
    for ii = 1:new_users
        new_user_name = user_track+1; %creates a new user name
        queue = [queue,new_user_name]; %adds the user to the queue
        user_track = new_user_name; %increments the latest user number
    end
end

% Defining Initial Serivce Flag 
service_full_flag = 0; %If this flag is 1, then no more users can be serviced

%% Moving entries from Queue to Service - From here on, everything is repeated in the main loop (with some additions)

while ((service_full_flag == 0)&&((length(queue)) > 0))
        % checking to see if new users can be serviced
        service_pointer = length(service.user_name);
        if (service_pointer == max_users)
            service_full_flag = 1;
            break;
        end
        % Extracting user from queue
        qii = 1; % the leading user in the queue is always serviced first
        user_being_serviced = queue(qii); %Picking the user that is being serviced
        % -- Collecting information of user being serviced ---
        file_num = randi(files_available);
        user_file_name = list(file_num).name;
        user_total_bits = list(file_num).totalbits;
        user_MCS = randi(max_MCS)+MCS_del;
        APEP_toss = randi(length(APEP_options));
        user_APEP = APEP_options(APEP_toss);
        % Updating service structure
        service.user_name(service_pointer+1) =  user_being_serviced; 
        service.user_file(service_pointer+1) = user_file_name;
        service.user_file_bits(service_pointer+1) = user_total_bits;
        service.user_file_bits_tx(service_pointer+1) = 0; % number of bits already transmitted
        service.MCS(service_pointer+1) = user_MCS; % Modulation and coding scheme of user
        service.APEP(service_pointer+1) = user_APEP; % APEP Length for the user is the number of payload bytes that is allowed. It is set in the MAC layer of the user. 
        queue(qii) = [];
end

%% Creating 802.11ax Transmission Parameters 

% Generating the allocation table based on the total number of users

total_servicing = length(service.user_name);
if total_servicing < 10
   allocation_index = allocation_table(total_servicing);
   packet_label = [0,0,0,1,1,0,0,0]; % 20 MHz banwidth in steps of 10 MHz
elseif total_servicing < 18
   allocation_index = [allocation_table(9), allocation_table(total_servicing-9)]; 
   packet_label = [0,0,1,1,1,1,0,0]; %40 MHz BW
elseif total_servicing == 18
   allocation_index = [allocation_table(9), allocation_table(9)]; 
   packet_label = [0,0,1,1,1,1,0,0]; %40 MHz
elseif total_servicing < 27
   allocation_index = [allocation_table(9), allocation_table(9), allocation_table(total_servicing-18), 113];
    packet_label = [1,1,1,1,1,1,0,0]; %60 MHz
elseif total_servicing == 27
   allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), 113];   
   packet_label = [1,1,1,1,1,1,0,0]; %60 MHz
elseif total_servicing < 36
   allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), allocation_table(total_servicing-27)];  
    packet_label = [1,1,1,1,1,1,1,1]; %80 MHz
else 
   allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), allocation_table(9)]; 
    packet_label = [1,1,1,1,1,1,1,1]; %80 MHz
end


    % Creating WLAN Transmission Object 
    heMUCfg = wlanHEMUConfig(allocation_index, 'LowerCenter26ToneRU', 0, ...
        'NumTransmitAntennas', 1, ...
        'STBC', 0, ...
        'GuardInterval', GI , ...
        'HELTFType', 4, ...
        'SIGBCompression', 0, ...
        'SIGBMCS', 0, ...
        'SIGBDCM', 0, ...
        'UplinkIndication', 0, ...
        'BSSColor', 0, ...
        'SpatialReuse', 0, ...
        'TXOPDuration', 127, ...
        'HighDoppler', 0, ...
        'SIGBCompression', 0);

    % Configuring the transmission parameters for each user and extracting the
    % bits to be transmitted
    for ii=1:total_servicing
        heMUCfg.User{ii}.APEPLength = service.APEP(ii);
                heMUCfg.User{ii}.MCS = service.MCS(ii);
                heMUCfg.User{ii}.NumSpaceTimeStreams = 1;
                heMUCfg.User{ii}.DCM = 0;
                heMUCfg.User{ii}.ChannelCoding = 'LDPC';
                heMUCfg.User{ii}.STAID = 0;
                heMUCfg.User{ii}.NominalPacketPadding = 0;            
    end

    psduLength = getPSDULength(heMUCfg); % Length of data field to transmit in bytes
    TX_Length = 8*psduLength; % %Number of bits to be transmitted in the next packet  
    max_pdsu = max(TX_Length);


    % Creating the trasmission structure for the next packet
    for ii = 1:total_servicing

        data_file = service.user_file(ii); % copying file name
        rr_min = service.user_file_bits_tx(ii) + 1; % read range min value

        if ((rr_min+ TX_Length(ii)-1) >= service.user_file_bits(ii)) %checking if range needed is out of file bounds
            rr_max = service.user_file_bits(ii); %terminating data extraction at max bit
        else
            rr_max = rr_min+ TX_Length(ii)-1;
        end
        data_extract = readmatrix(sprintf('%s%s',read_path,data_file),'Range',[rr_min 1 rr_max 1]); % Reading the required databits for the user
        TX_Psdu{ii}  = data_extract'; % Compiling data for one packet for one user
        service.user_file_bits_tx(ii) = rr_max; %Updating the number of bits transmitted

    end

    %% Generate the waveform

    full_waveform = wlanWaveformGenerator(TX_Psdu, heMUCfg, ...
        'NumPackets', numPackets, ...
        'IdleTime', 0, ...
        'ScramblerInitialization', 93, ...
        'WindowTransitionTime', 1e-07);

    % Extracting the data from the header 
    ind = wlanFieldIndices(heMUCfg); %Creates a strucutre with the index of different fields
    points = ind.HEData; %Location of data in the waveform

    waveform = full_waveform(points(1):points(2)); %This is the waveform without the header
    waveform = waveform.'; % Converting it to a row vector
    % Upsampling the data to the fs_max value 
    Fs = wlanSampleRate(heMUCfg); % Calculate the sample rate of waveform
    resample_rate = fs_max/Fs; % Calculating how many times one sample should be repeated
    if resample_rate>1
        up_m = repmat(waveform,[resample_rate,1]); % create copies of the waveform array
        upsampled_waveform = reshape(up_m,1,[]); % reshape to obtain upsampled signal
    else 
        upsampled_waveform = waveform;
    end


    upsampled_waveform = upsampled_waveform.'; %channel model later requires column vector

    %% Adding noise to upsampled waveform 

    % Creating and passing the signal through an indoor channel
    Tx_dist = TX_RX_distance_min + rand*(TX_RX_distance_max-TX_RX_distance_min); %A variable distance between trasmitter and receiver 
    tgax = wlanTGaxChannel('SampleRate',fs_max,'DelayProfile',channel_model,'ChannelBandwidth',ch_BW,...
        'CarrierFrequency',fc,'TransmitReceiveDistance',Tx_dist,...
        'LargeScaleFadingEffect',LS_fading,'NumPenetratedWalls',num_walls); % Defining an indoor WLAN channel

    % Adding awgn noise after calculating the signal power
    preChSigPwr_dB = 20*log10(mean(abs(upsampled_waveform))); %calculating the signal power before adding the channel
    sigPwr = 10^((preChSigPwr_dB-tgax.info.Pathloss)/10); % Refer to https://www.mathworks.com/help/wlan/gs/wlan-channel-models.html

    SNR_req = SNR_min +  rand* (SNR_max-SNR_min);
    chNoise = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)',...
        'SNR',SNR_req,'SignalPower', sigPwr);

    postCh_waveform = chNoise(tgax(upsampled_waveform)); % Passing the signal through the indoor channel and then AWGN channel


    %% Reshaping the waveform in the format needed
    frame_nums = length(postCh_waveform )/Net_frame_length;
    reshaped_frame = (reshape(postCh_waveform,Net_frame_length,frame_nums)).'; %reshaping so that one OFDM frame and symbol is in one row
    data_frame = [data_frame;reshaped_frame(:,CP_frame_length+1:end)];
    label_frame_packet = repmat(packet_label,frame_nums,1);
    label_frame = [label_frame; label_frame_packet];


    %% Calculating full packet duration 

    t_packet = (1/Fs)*(length(full_waveform)-1); %Calculated using the original BW and original waveform
    time_calc = time_calc+t_packet; 

    iteration_number = iteration_number+1; 
    service_track{iteration_number} = service.user_name;


 
%% Removing users who have been serviced 




while(sum(service.user_file_bits <= service.user_file_bits_tx)>0)
    for ii = 1:length(service.user_name)
        if (service.user_file_bits(ii) == service.user_file_bits_tx(ii))
            service.user_name(ii) = [];
            service.user_file(ii) = [];
            service.user_file_bits(ii) = [];
            service.user_file_bits_tx(ii) = [];
            service.MCS(ii) = [];
            service.APEP(ii) = [];
            break;
        end        
    end        
end  

TX_Psdu = {}; % clear TX_Psdu after every iteration
%% Main Loop


while(time_calc<sim_time)
    if (floor(time_calc) ~= seconds_track) % Checking if it is time for the queue to the updated
        seconds_track = seconds_track + 1;
        lambda_val = lambda(seconds_track+1); %Get current lambda value
        % --- Generate new users ------
        new_users = poissrnd(lambda_val); % Calculate number of new users
        if new_users > 0
            for ii = 1:new_users
                new_user_name = user_track+1; %creates a new user name
                queue = [queue,new_user_name]; %adds the user to the queue
                user_track = new_user_name; %increments the latest user number
            end
        end
    end
    % Servicing new users
    if (length(service.user_name) == max_users)
        service_full_flag = 1;
    else
        service_full_flag = 0;        
    end
    
    while ((service_full_flag == 0)&&((length(queue)) > 0))
            % checking to see if new users can be serviced
            service_pointer = length(service.user_name);
            if (service_pointer == max_users)
                service_full_flag = 1;
                break;
            end
            % Extracting user from queue
            qii = 1; % the leading user in the queue is always serviced first
            user_being_serviced = queue(qii); %Picking the user that is being serviced
            % -- Collecting information of user being serviced ---
            file_num = randi(files_available);
            user_file_name = list(file_num).name;
            user_total_bits = list(file_num).totalbits;
            user_MCS = randi(max_MCS)+MCS_del;
            APEP_toss = randi(length(APEP_options));
            user_APEP = APEP_options(APEP_toss);
            % Updating service structure
            service.user_name(service_pointer+1) =  user_being_serviced; 
            service.user_file(service_pointer+1) = user_file_name;
            service.user_file_bits(service_pointer+1) = user_total_bits;
            service.user_file_bits_tx(service_pointer+1) = 0; % number of bits already transmitted
            service.MCS(service_pointer+1) = user_MCS; % Modulation and coding scheme of user
            service.APEP(service_pointer+1) = user_APEP; % APEP Length for the user is the number of payload bytes that is allowed. It is set in the MAC layer of the user. 
            queue(qii) = [];
    end
    % Generating the allocation table based on the total number of users

    total_servicing = length(service.user_name);
    if total_servicing < 10
       allocation_index = allocation_table(total_servicing);
       packet_label = [0,0,0,1,1,0,0,0]; %20 MHz
    elseif total_servicing < 18
       allocation_index = [allocation_table(9), allocation_table(total_servicing-9)]; 
       packet_label = [0,0,1,1,1,1,0,0]; %40 MHz
    elseif total_servicing == 18
       allocation_index = [allocation_table(9), allocation_table(9)]; 
       packet_label = [0,0,1,1,1,1,0,0]; %40 MHz
    elseif total_servicing < 27
       allocation_index = [allocation_table(9), allocation_table(9), allocation_table(total_servicing-18), 113];
        packet_label = [1,1,1,1,1,1,0,0]; %60 MHz
    elseif total_servicing == 27
       allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), 113];
       packet_label = [1,1,1,1,1,1,0,0]; %60 MHz
    elseif total_servicing < 36
       allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), allocation_table(total_servicing-27)];
       packet_label = [1,1,1,1,1,1,1,1]; %80 MHz
    else 
       allocation_index = [allocation_table(9), allocation_table(9), allocation_table(9), allocation_table(9)]; 
       packet_label = [1,1,1,1,1,1,1,1]; %80 MHz
    end
    
    if (~isempty(allocation_index))
    
        % Creating WLAN Transmission Object
        heMUCfg = wlanHEMUConfig(allocation_index, 'LowerCenter26ToneRU', 0, ...
            'NumTransmitAntennas', 1, ...
            'STBC', 0, ...
            'GuardInterval', GI , ...
            'HELTFType', 4, ...
            'SIGBCompression', 0, ...
            'SIGBMCS', 0, ...
            'SIGBDCM', 0, ...
            'UplinkIndication', 0, ...
            'BSSColor', 0, ...
            'SpatialReuse', 0, ...
            'TXOPDuration', 127, ...
            'HighDoppler', 0, ...
            'SIGBCompression', 0);

        % Configuring the transmission parameters for each user and extracting the
        % bits to be transmitted
        for ii=1:total_servicing
            heMUCfg.User{ii}.APEPLength = service.APEP(ii);
                    heMUCfg.User{ii}.MCS = service.MCS(ii);
                    heMUCfg.User{ii}.NumSpaceTimeStreams = 1;
                    heMUCfg.User{ii}.DCM = 0;
                    heMUCfg.User{ii}.ChannelCoding = 'LDPC';
                    heMUCfg.User{ii}.STAID = 0;
                    heMUCfg.User{ii}.NominalPacketPadding = 0;            
        end

        psduLength = getPSDULength(heMUCfg); % Length of data field to transmit in bytes
        TX_Length = 8*psduLength; % %Number of bits to be transmitted in the next packet  
        max_pdsu = max(TX_Length);


        % Creating the trasmission structure for the next packet
        for ii = 1:total_servicing

            data_file = service.user_file(ii); % copying file name
            rr_min = service.user_file_bits_tx(ii) + 1; % read range min value

            if ((rr_min+ TX_Length(ii)-1) >= service.user_file_bits(ii)) %checking if range needed is out of file bounds
                rr_max = service.user_file_bits(ii); %terminating data extraction at max bit
                data_extract = readmatrix(sprintf('%s%s',read_path,data_file),'Range',[rr_min 1 rr_max 1]); % Reading the required databits for the user
                TX_Psdu{ii}  = data_extract'; % Compiling data for one packet for one user
                service.user_file_bits_tx(ii) = rr_max; %Updating the number of bits transmitted
            else
                rr_max = rr_min+ TX_Length(ii)-1;
                data_extract = readmatrix(sprintf('%s%s',read_path,data_file),'Range',[rr_min 1 rr_max 1]); % Reading the required databits for the user
                TX_Psdu{ii}  = data_extract'; % Compiling data for one packet for one user
                service.user_file_bits_tx(ii) = rr_max; %Updating the number of bits transmitted
            end


        end

        full_waveform = wlanWaveformGenerator(TX_Psdu, heMUCfg, ...
            'NumPackets', numPackets, ...
            'IdleTime', 0, ...
            'ScramblerInitialization', 93, ...
            'WindowTransitionTime', 1e-07);

        % Extracting the data from the header 
        ind = wlanFieldIndices(heMUCfg); %Creates a strucutre with the index of different fields
        points = ind.HEData; %Location of data in the waveform

        waveform = full_waveform(points(1):points(2)); %This is the waveform without the header
        waveform = waveform.'; % Converting it to a row vector
        % Upsampling the data to the fs_max value 
        Fs = wlanSampleRate(heMUCfg); % Calculate the sample rate of waveform
        resample_rate = fs_max/Fs; % Calculating how many times one sample should be repeated
        if resample_rate>1
            up_m = repmat(waveform,[resample_rate,1]); % create copies of the waveform array
            upsampled_waveform = reshape(up_m,1,[]); % reshape to obtain upsampled signal
        else 
            upsampled_waveform = waveform;
        end


        upsampled_waveform = upsampled_waveform.'; %channel model later requires column vector

        % Creating and passing the signal through an indoor channel
        Tx_dist = TX_RX_distance_min + rand*(TX_RX_distance_max-TX_RX_distance_min); %A variable distance between trasmitter and receiver 
        tgax = wlanTGaxChannel('SampleRate',fs_max,'DelayProfile',channel_model,'ChannelBandwidth',ch_BW,...
            'CarrierFrequency',fc,'TransmitReceiveDistance',Tx_dist,...
        'LargeScaleFadingEffect',LS_fading,'NumPenetratedWalls',num_walls); % Defining an indoor WLAN channel
        

        % Adding awgn noise after calculating the signal power
        preChSigPwr_dB = 20*log10(mean(abs(upsampled_waveform))); %calculating the signal power before adding the channel
        sigPwr = 10^((preChSigPwr_dB-tgax.info.Pathloss)/10); % Refer to https://www.mathworks.com/help/wlan/gs/wlan-channel-models.html

        SNR_req = SNR_min +  rand* (SNR_max-SNR_min);
        chNoise = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)',...
            'SNR',SNR_req,'SignalPower', sigPwr);

        postCh_waveform = chNoise(tgax(upsampled_waveform)); % Passing the signal through the indoor channel and then AWGN channel

        % Reshaping the waveform in the format needed
        frame_nums = length(postCh_waveform )/Net_frame_length;
        reshaped_frame = (reshape(postCh_waveform,Net_frame_length,frame_nums)).'; %reshaping so that one OFDM frame and symbol is in one row

        data_frame = [data_frame;reshaped_frame(:,CP_frame_length+1:end)]; %This is the frame to be collected
        label_frame_packet = repmat(packet_label,frame_nums,1);
        label_frame = [label_frame; label_frame_packet]; %This is the label frame to be collected
        


        % Calculating full packet duration and removing users who have finished transmission

        t_packet = (1/Fs)*(length(full_waveform)-1); %Calculated using the original BW and original waveform
        time_calc = time_calc+t_packet 

        % Removing users who have been serviced 

        iteration_number = iteration_number+1; 
        service_track{iteration_number} = service.user_name;

    else

        time_comp = time_calc; %calculating the time that has been completed
        time_new_queue = ceil(time_comp)+10*(1/fs_max); %Going above the integer number when queue is serviced again
        time_del = time_new_queue-time_comp;
        zero_samples1 = ceil(time_del*fs_max); 
        zero_samples2 = OFDM_frame_length*ceil(zero_samples1/OFDM_frame_length); % getting a multiple of OFDM frame length
        frame_nums_blank = zero_samples2/OFDM_frame_length;
        blank_data = zeros(zero_samples2,1);
        sigPwr = 10^(mean((blank_data))/10);
        SNR_req = SNR_min +  rand* (SNR_max-SNR_min);
        chNoise = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)',...
            'SNR',SNR_req,'SignalPower', sigPwr);
        postCh_waveform = chNoise((blank_data)); %adding noise
              
        new_data_frame = reshape(postCh_waveform,frame_nums_blank,OFDM_frame_length); %reshaping the frame
        data_frame = [data_frame; new_data_frame];
        
        packet_label_blank = [0,0,0,0,0,0,0,0];
        label_frame_packet = repmat(packet_label_blank,frame_nums_blank,1);
        label_frame = [label_frame; label_frame_packet]; %This is the label frame to be collected
        
        
        t_packet = (1/fs_max)*(zero_samples2-1);
        time_calc = time_calc+t_packet 
        disp('Blank frame transmitted')


        iteration_number = iteration_number+1; 
        service_track{iteration_number} = 0;
    
       end
      
    
    while(sum(service.user_file_bits <= service.user_file_bits_tx)>0)
        for ii = 1:length(service.user_name)
            if (service.user_file_bits(ii) == service.user_file_bits_tx(ii))
                service.user_name(ii) = [];
                service.user_file(ii) = [];
                service.user_file_bits(ii) = [];
                service.user_file_bits_tx(ii) = [];
                service.MCS(ii) = [];
                service.APEP(ii) = [];
                break;
            end        
        end        
    end 
    
    TX_Psdu = {}; % Clear TX_Psdu after every iteration

end



%Prepare Data
data_frame_I = real(data_frame);
data_frame_Q = imag(data_frame);


% Save Data
save(sprintf('%s%s.mat',write_path,write_name), "data_frame_I", "data_frame_Q","label_frame",'-v7.3') 
