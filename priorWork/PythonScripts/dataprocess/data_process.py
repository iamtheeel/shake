# -*- coding: utf-8 -*-
"""
DataProcess
Contains post-processing methods to enhance the data
--------------------------------------
Contributors: Yohanna Mejia <mejiacru@sfsu.edu>, Juan Caicedo <caicedo@cec.sc.edu>, Charlie Vidal <cvidal@mail.sfsu.edu>, 
Last Modification: March, 2021
"""

#%% importing packages
from nidaqpy import Trial
from nidaqpy import Experiment

import dataprocess.storage as st
from dataprocess.spaceprior import SpacePrior

import numpy as np
import json
from scipy import signal
import matplotlib.pylab as plt
import matplotlib.mlab as mlab

import pandas as pd

import os

import peakutils

#%%
class TrialProcess(Trial):
    
    """
    Creates a trial for postprocessing
    
    Attributes
    ----------
    experiment : Experiment
        Experiment of this particular trial.
    ident : int
        ID of this particular trial.  Matches with hdf5 file
        
    Methods
    -------
    get_data():               Reads the data from the hdf5 file
        
        
    __resample_data__():          Resamples the trial records using a new sampling 
                              frequency (fs)
    
    __filter_data_lowpass__(): Filters the trial records using a low pass filter 
    
    center_data():           Centers the trial records for a total length of 
                             2*size
    
    split_data():            Splits the trial records using a reference sensor.
                             To be use in case of multiple "events" in the 
                             same record.
    
    tfestimate():            Calculates the transfer function of any input and 
                             output.
        
    tfes_data():             Calculates the transfer function of each record 
                             of the trial.
    
    
    """
    def __init__(self, experiment, ident = -1):
        Trial.__init__(self, experiment, ident)
        self.experiment = experiment
        self.tfe = ''
        self.freq = ''
        
        # import storage
        # container_parameter = {'fname':'abc.csv','columns':['a','b','c']}
        # self.container = storage.Container(ident,container_parameters)
        
    def get_data(self):
        '''
        Reads data from the hdf5 file or a temporary pickle file if the
        data has been modified before in the current session
        
        Returns:
        -------
        dataT : Data for the trial
        '''
        key = '%s'%self.id
        if key in self.experiment.tmp_data:
            return self.experiment.tmp_data[key]
        else:
            return Trial.get_data(self)
        
    def set_data(self,data):
        '''
        Sets the data for a particular trial

        Parameters
        ----------
        data : numpy array
            Data of the trial.  This should be an nxm array where n is the number
            of sensors and m is the number of data points.

        Returns
        -------
        None.

        '''
        self.experiment.tmp_data['%s'%self.id] = data
        
        
    def __resample_data__(self,fs):
        """
        Resamples the trial using a new sampling frequency
        WARNING: The sampling frequency is a property of the experiment, not
        the trial.  Therefore, it is highly recommended to use
        ExperimentTrial.resample_data(fs) instead of 
        ExperimentTrial.trials[0].resample_data(fs).  The second statement
        will not change the sampling frequency in the experiment and
        methods like TrialProcess.plot() will not work correctly.
        
        Parameters:
        --------  
        fs:     float 
                new sampling frequency
        
        Returns:
        --------
        data:    TrialProcess object
        """
        
        data = self.get_data()
        
        self.set_data(signal.resample(data,int(np.ceil(len(data)*(round(fs)/round(self.experiment.fs))))))
        self.fft_n = data.shape[0] 
        
    def __filter_data_lowpass__(self, fs, cutoff_freq, order):
        
        """
        Filter the data using a Impulse Response Filter for each trial.
        
        Please refer to this method in scipy:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
        
        1. Design an Nth-order digital or analog Butterworth filter and return 
        the filter coefficients.
        2. filters the data according to the filter coefficients
        
        Parameters:
        ----------
        order        :int 
                     The order of the filter
        
        fs          :int
                     The sampling frequency of the digital system.
                     
        cutoff_freq: int
                    For a Butterworth filter, this is the point at 
                    which the gain drops to 1/sqrt(2) that of the passband 
                    (the “-3 dB point”). Value of Frequency in Hz. All 
                    frequencies after this value are removed.
        
        Returns:
        ----------
        None
        
        """
        
        #Filter design function
        def butter_lowpass(cutoff, fs, order=order):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        
        #design current filter with input parameters
        b, a = butter_lowpass(cutoff=cutoff_freq, fs =fs, order=order)
        filtered_data = []
        
        for data_i in range(self.get_data().shape[1]):            
            tmp = signal.lfilter(b, a, self.get_data()[:,data_i])
            filtered_data.append(tmp.flatten())
        
        self.set_data(np.array(filtered_data).T)
      

    def filter_data_FIR(self, cutoff_freq, ripple=7, width=10, zero_phase=False):
        """
        Runs the trial data to a lowpass finite impulse response 
        filter.
        
        Parameters:
        --------
                              
        cutoff_freq     : integer 
                        Data above this frequency in Hz will be removed                
                        will be removed.
        
        ripple          : integer 
                        value in decibels, giving less or more attenuation 
                        of wanted signals.
        
        width           : integer 
                        Transition width, given in Hz.
                        
        zero_phase:     bool
                        If True:
                        Apply a digital filter forward and backwards to a signal.
                        This function applies a linear digital filter twice, 
                        once forward and once backwards. The combined filter 
                        has zero phase and a filter order twice that of the original.
                        
                        If False:
                        Filter data along one-dimension with an IIR or FIR filter.
                        
       filtfilt: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
       lfilter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                        
                       
        Returns:
        --------
        
        filtered_data   : TrialProcess object
    
        """  

        data = self.get_data()

        #Defining nyquist frequency    
        nyquist = self.experiment.fs/2.0
        #Setting transition width with respect to nyquist frequency
        width = width / nyquist
        #Getting order (number of taps) and beta value for kaiser window using the 
        #ripple and width parameters
        N , beta = signal.kaiserord(ripple,width)
        #Creating filter
        ntaps = signal.firwin(N, cutoff_freq / nyquist, window=('kaiser',beta))
        # Phase delay (to get the delay in sec we divide by fs)
        # phase_delay = 0.5 * (len(ntaps) -1) 
        #denominator in kaiser window, always 1.0
        a = 1.0
        
        if zero_phase == True:
            filtered_data = signal.filtfilt(ntaps, a, data,
                                         axis = 0, 
                                         padtype = 'odd', 
                                         method = 'pad', 
                                         irlen = None)
        else:
            filtered_data = signal.lfilter(ntaps, a, data)
        
        self.set_data(filtered_data)
       
    def center_data(self, size, ref_sensor): 
        """
        Centers the trial data according to the size specified.

        Parameters
        ----------
        size :        int
                      Desired amount of points left and right of the peak
        ref_sensor :  int
                      Reference sensor used to cut all the other
                      sensors records at the same location, 
                      usually, the force sensor 0 if available

        Returns
        -------
        None.

        """
        
        where = np.argmax(ref_sensor)
    
        trim1 = where-size
        trim2 = where+size
        
        data = self.get_data()
        
        if trim1 < 0:
            self.set_data(np.pad(data, (np.abs(trim1),0), 'constant'))
            trim2 = trim2-trim1
            trim1 = 0
        if trim2 > len(data):
            self.set_data(np.pad(data, (0, trim2-len(data)),
                                    'constant'))        
        
    
    def split_data(self, ref_sensor, window, droop_keys=[], plot=False):
        """
        Splits the data from a trial of a calibration experiment and returns
        multiple records of the same trial. 
        
        Parameters
        ----------
        ref_sensor:       float
                          The sensor to use as a reference when looking for the
                          peaks to perform the splitting. Usually,
                          in calibration records, the reference sensor is 
                          selected as the force sensor.
                          
        window:           float
                          Length of the window of each record in seconds
    
        plot:             bool
                          If True, it will plot a single record with the multiple
                          events and color bands of the splitting records. 
                          This can be used to check if the splitting was achived
                          as expected.
                          
        Returns
        -------
        None.             
        
        """
        
        #time_vector
        time_data = np.arange(0,self.experiment.record_length,1/self.experiment.fs)
        
        #window in time to separate peaks as independent records
        hit_window = int(window*self.experiment.fs/2)
        records_all_sensors = []
        # Looping through all sensors   
        dat_tmp = self.get_data()
        for sensor_i in range(dat_tmp.shape[1]):
          
          data_ref = dat_tmp[:,ref_sensor]
          peaks  = peakutils.indexes(data_ref, thres=0.2, min_dist=50)
        
          data_sensor = dat_tmp[:,sensor_i]
          
          if plot:
              plt.figure()
              if sensor_i==ref_sensor:
                  plt.plot(time_data,data_ref)
                  plt.plot(time_data[peaks], data_ref[peaks],'ro')
              else:
                  plt.plot(time_data,data_sensor)
          tmp = []
          for pk_i in peaks:
            
            # record_split += 1 
            
            if pk_i+hit_window > len(time_data):
                max_lim= len(time_data)-1
                min_lim= max_lim-hit_window*2
            elif pk_i-hit_window < 0:
                max_lim= pk_i+hit_window
                min_lim= 0
            else:
                max_lim = pk_i+hit_window
                min_lim =  pk_i-hit_window                        
             
            zero_padding = int(window*self.experiment.fs)
            val = zero_padding-len(data_sensor[min_lim:max_lim])
            if val>0:    
                data_accel = np.pad(data_sensor[min_lim:max_lim],val)
            else:
                data_accel = data_sensor[min_lim:max_lim]
            tmp.append(data_accel)
            
            #Plotting
            if plot:
                # ---
                color=list(np.random.choice(range(256), size=3)/256)
          
                plt.axvspan(xmin = time_data[min_lim], 
                            xmax = time_data[max_lim], 
                            alpha=0.5, facecolor = color, 
                            linewidth = 0.001)
                plt.vlines([time_data[min_lim]], 
                           ymin = -0.01,ymax = 0.01, 
                           color = color)
                plt.vlines([time_data[max_lim]], ymin = -0.01,
                           ymax = 0.01, color = color)
                plt.xlabel('time (s)')
                plt.ylabel('Amplitude')
                plt.title('Trial %s-sensor %s'%(self.id, sensor_i))
                plt.tight_layout()
                # ---
                
          tmp_ = np.vstack(tmp)
          records_all_sensors.append(tmp_)
          #Saving images 
          if plot:
            path_2 = os.path.join('data\\%s\\split-hit-data'%self.experiment.fname.split(sep='/')[-1].split('.')[0], 'Trial-%s'%self.id)
            if not os.path.exists(path_2):
                os.makedirs(path_2)                
            plt.savefig(os.path.join(path_2, 
                                     'Trial %s-sensor %s'%(self.id, sensor_i)))
            plt.close()
                        
        records = np.transpose(np.array(records_all_sensors),(1,0,2))
        
        return records
        
    def tfestimate(self,input_x, output_y, window_func=None, noverlap_perc=None, 
                   fft_n = '', fs = '', tf_method=3, filter_tf =False):
 
        """
        Estimation of the transfer function (TFE) using five different
        formulations including the power spectral density by Welch's average 
        periodogram method  (mlab.psd) https://kite.com/python/docs/matplotlib.mlab.psd
        and numpy fft.
        
        Parameters
        ----------
                
        input_x          :1D array-like
                         The input (force)
                         
        input_y          :1D array-like
                         The output (acceleration)
                         
        window_func:          A vector of length NFFT. 
                         Possible functions to use to create the array:
                         numpy.blackman, 
                         numpy.hamming, numpy.bartlett, scipy.signal, 
                         scipy.signal.get_window, etc. If a function 
                         is passed as the argument, it must take a data segment
                         as an argument and return the windowed version of the 
                         segment.
                         
        noverlap_per    :int
                        Percentage of points to overlap between segments.The 
                        default value is None (no overlap). 
                        
        fft_n           :int
                        Length of the FFT
                        
        fs              :float
                        Sampling frequency
                        
        tf_method:      float
                        With $P_{xy}$ as the cross power spectral density of 
                        the input force to the acceleration, $P_{xx}$ as the 
                        power spectral density of the input force, $P_{yx}$ as 
                        the cross power spectral density of the acceleration
                        to the input force, and $P_{yy}$ as the power spectral
                        density of the acceleration.
                        
                        If 1:
                        The TFE will be obtained using:                            
                        $tf_1 = \frac{P_xy}/{Pxx}$
                        
                        If 2:
                        The TFE will be obtained using:                            
                        $tf_2 = \frac{P_yy}/{Pyx}$
                        
                        If 3:
                        The TFE will be obtained using:                            
                        $tf_3 = \frac{tf1+tf2}/{2}$
                        
                        If 4:
                        The TFE will be obtained using:                            
                        $tf_4 = \frac{FFT(y)}/{FFT(x)}$
                        
                        If 5:
                        The TFE will be obtained using:                            
                        $tf_5 = \sqrt(tf1 \times tf2)$
        Returns
        -------
        Tfe:            1D array
                        Transfer Function Estimate

        freq:           1D array
                        Frequencies for PSD and FFT        

        """
        
        if noverlap_perc:
            noverlap =int((noverlap_perc/100)*fft_n) 
        else:
            noverlap=None
        # Cross spectral density calculations
        Pxx, freq = mlab.psd(x=input_x, Fs=fs,
                             NFFT=fft_n, 
                             window=window_func, 
                             noverlap=noverlap)
        
        Pxy, _ = mlab.csd(x=input_x, y=output_y, Fs=fs,
                          NFFT=fft_n, 
                          window=window_func, 
                          noverlap=noverlap)
        
        Pyy, _ = mlab.psd(x=output_y, Fs=fs,
                          NFFT=fft_n, 
                          window=window_func, 
                          noverlap=noverlap)
        
        Pyx, _ = mlab.csd(x=output_y, y=input_x, Fs=fs,
                          NFFT=fft_n, 
                          window=window_func, 
                          noverlap=noverlap)
        
        if tf_method ==1:
            # Method 1
            tfe = Pxy/Pxx
            
        elif tf_method ==2:
            # Method 2
            tfe = Pyy/Pyx
            
        elif tf_method ==3:
            # Method 3
            tf1 = Pxy/Pxx
            tf2 = Pyy/Pyx
            tfe = np.average(np.vstack((tf1,tf2)), axis = 0)
            
            
        elif tf_method ==4:
            #Method 4
            x_fft = np.fft.fft(input_x, n=fft_n, axis=0)
            y_fft = np.fft.fft(output_y,n=fft_n, axis=0)
            freq = np.fft.fftfreq(fft_n, 1/fs)
            tfe = (y_fft/x_fft)[0:int((fft_n/2) + 1)]
        
        elif tf_method ==5:
            # Method 5
            tfe = (tf1*tf2) ** 0.5
            
        elif tf_method==6:
            TF1_ = np.fft.rfft(input_x, n=fft_n, axis=0)
            TF2_ = np.fft.rfft(output_y,n=fft_n, axis=0)
            freq = np.fft.rfftfreq(fft_n, 1/fs)
            tfe = TF2_.T/TF1_
        
        
      
        if filter_tf: 
          tfe = np.exp(signal.savgol_filter(np.log(tfe), 51, 8))

        
        return tfe, freq
    
    def tfes_data(self, force_sensor, window_func=None, noverlap_perc=None, 
                tf_method=3, fft_n='', fs='', remove_sensors=None):
        
        """
        Calculates the transfer function estimate of each trial using the input
        force and the output accelerations. 
        
        Parameters
        ----------
        force_sensor:   float
                        indx of the force sensor  

        
                         
        window_func:         A vector of length NFFT. 
                         Possible functions to use to create the array:
                         numpy.blackman, 
                         numpy.hamming, numpy.bartlett, scipy.signal, 
                         scipy.signal.get_window, etc. If a function 
                         is passed as the argument, it must take a data segment
                         as an argument and return the windowed version of the 
                         segment.
                         
        noverlap_per    :int
                        Percentage of points to overlap between segments.The 
                        default value is 0 (no overlap). 
                        
        tf_method:      float
                        With $P_{xy}$ as the cross power spectral density of 
                        the input force to the acceleration, $P_{xx}$ as the 
                        power spectral density of the input force, $P_{yx}$ as 
                        the cross power spectral density of the acceleration
                        to the input force, and $P_{yy}$ as the power spectral
                        density of the acceleration.
                        
                        If 1:
                        The TFE will be obtained using:                            
                        $tf_1 = \frac{P_xy}/{Pxx}$
                        
                        If 2:
                        The TFE will be obtained using:                            
                        $tf_2 = \frac{P_yy}/{Pyx}$
                        
                        If 3:
                        The TFE will be obtained using:                            
                        $tf_3 = \frac{tf1+tf2}/{2}$
                        
                        If 4:
                        The TFE will be obtained using:                            
                        $tf_4 = \frac{FFT(y)}/{FFT(x)}$
                        
                        If 5:
                        The TFE will be obtained using:                            
                        $tf_5 = \sqrt(tf1 \times tf2)$
                        
        fft_n:           int
                        The number of data points used in each block for the 
                        FFT. A power 2 is most efficient. 
                        The default value is the length of the record. 
                        
                        
        fs              float
                        Sampling frequency
                        
        remove_sensors  :list
                        A list of sensors to ignore during the contruction of
                        the TFEs.
        """
        
        data = self.get_data()
        
        if not fft_n:
            fft_n = data.shape[0]            
            
        self.experiment.fft_n = fft_n #Adding the attribute of the FFT length to the experiment
        
        #force signal using the reference sensor
        trial_force = data[:,force_sensor]
            
        #extracting only acceleration sensors
        sensors = list(range(min(data.shape)))
        if remove_sensors:
            for sen_i in remove_sensors:
                sensors.remove(sen_i)
        sensors.remove(force_sensor)
        #calulating tfs for all sensors in each trial
        tmp_sensors = []
        for sensor_i in sensors:
            trial_accel = data[:,sensor_i]
            tmp, freq = self.tfestimate(input_x=trial_force,
                                    output_y=trial_accel,
                                    window_func=window_func,
                                    noverlap_perc=noverlap_perc,
                                    fft_n = fft_n,
                                    fs = self.experiment.fs,
                                    tf_method=tf_method)
            
            tmp_sensors.append(tmp)
        #TFs all sensors    
        tmp_sen = np.array(tmp_sensors).T       
        #Adding the information of the sensors used in tfe calculations
        tmp=[]
        for sen_i in sensors:
            tmp.append(self.experiment.sensors[sen_i])
        self.experiment.tfe_sensors = list(tmp)
        self.tfe = tmp_sen
        self.freq = freq
        
    def get_noise_threshold(self):
        
        """
        Calculates the noise level of the combined signals between sensors
                
        Parameters
        ---------- 
        None
        
        """
        signal = self.get_data().mean(axis=1)
        mean = signal.mean(axis=0)
        sd = signal.std(axis=0, ddof=0)
        thres = mean + 4*sd
        
        self.combined_signal = signal
        self.thres = thres
              
    def get_fft(self, n_points=None):
        
        """
        Compute the one-dimensional discrete Fourier Transform for real input.
        This function computes the one-dimensional n-point discrete Fourier 
        Transform (DFT) of a real-valued array by means of an efficient 
        algorithm called the Fast Fourier Transform (FFT).

        Parameters:
        ----------  
        n_points: int, optional
        Number of points along transformation axis in the input to use. 
        If n is smaller than the length of the input, the input is cropped. 
        If it is larger, the input is padded with zeros. If n is not given, 
        the length of the input along the axis specified by axis is used. If n
        is even, the length of the transformed axis is (n/2)+1. If n is odd, 
        the length is (n+1)/2
        
        Returns:
        ----------
        None

        """
        data = self.get_data()
        if not n_points:
            n_points = data.shape[0]
            
        self.fft = np.fft.rfft(data, n=n_points, axis=0)
        self.freq = np.fft.rfftfreq(n=n_points, d=1/self.fs)

    def wiener_filter(self):
        '''
        Applies a Wiener filter on the data

        Returns
        -------
        filtered_data : numpy.array
            Filtered data

        '''
        
        data=self.get_data()
        filtered_data=np.zeros_like(data) #initialize with zeros matching size
        for i in range(np.shape(data)[1]):        
                filtered_data[:,i]=signal.wiener(data[:,i],mysize=201) #Wiener filter

        return filtered_data
        
    def id_peaks(self,window_t = 0.3, verbose = True):
        """
        Calculate peaks of the acceleration record.  The function returns the
        id of the record as well as the magnitude.  The magnitude is either
        positive or negative depending if the peak is in the positive
        or negative side of the signal.

        Parameters
        ----------
        window_t : float, optional
            The window used to calculate peaks. The default is 0.3.
        verbose : bool, optional
            If True (default), "Calculating Peaks ..." is shown if the peaks
            are being calculated and not obtained from the database

        Returns
        -------
        all_ids : TYPE
            IDs of the peaks.
        all_mag : TYPE
            Magnitude of the peaks.  This includes the sign (positive or negative).

        """
        
        # if not self.container.exists('peaks'):
        
        # Checking if the data is in storage
        sto_out = self.experiment.__sto__.read('peaks',col = 'id', value = self.id)
        if sto_out == []:
            # If the data is not in storage
            if verbose:
                print('Calculating Peaks...')
            fs=self.experiment.fs
            
            record_raw=self.wiener_filter() #retrieves filtered data
            record=np.abs(record_raw)
            all_ids=[]
            all_mag = []
            for i in range(np.shape(record)[1]):
                data=record[:,i]
                window_length = int(np.round(window_t*fs))
                top_ids = np.array([])
                for point_id in range(np.shape(data)[0]- window_length):
                    top_id = np.where(data[point_id:point_id + window_length] == max(data[point_id:point_id + window_length]))[0][0] + point_id
                    if top_id not in top_ids:
                        # Remove any other top_ids in this window
                        top_ids = top_ids[top_ids<point_id]
                        # Add the new identified top_id
                        top_ids=np.append(top_ids,top_id)
                top_ids_forward = top_ids
                #% Backward
                top_ids = np.array([])
                for point_id in range(np.shape(data)[0]-1,window_length,-1):
                    top_id = np.where(data[point_id-window_length:point_id] == max(data[point_id-window_length:point_id]))[0][0] + point_id - window_length
                    if top_id not in top_ids:
                        # Remove any other top_ids in this window
                        top_ids = top_ids[top_ids>point_id-1]
                        # Add the new identified top_id
                        top_ids=np.append(top_ids,top_id)
            
                top_ids_backward = top_ids 
                top_ids = list(set(top_ids_forward).intersection(top_ids_backward))
                top_ids.sort()
                top_ids = list(map(int, top_ids))
                all_ids.append(top_ids)
                all_mag.append(record_raw[top_ids,i].tolist())
                # Saving data in storage
                for top_id in top_ids:
                    row = {'id':self.id,'Channel':i, 'Location':top_id, 'Magnitude':record_raw[top_id,i]}
                    self.experiment.__sto__.add('peaks',row)
        else:
            # Changing the format to list of peaks per channel
            # Creating the structure of the list of lists
            all_ids = []
            all_mag = []
            for counter in range(len(self.experiment.sensors)-self.experiment.__ncam__): # Number of channels
                all_ids.append([])
                all_mag.append([])
            # Get each peak to the appropriate list
            for peak in sto_out:
                all_ids[peak[1]].append(peak[2])
                all_mag[peak[1]].append(peak[3])
        return all_ids, all_mag
    
    def plot(self):
            import matplotlib.pylab as plt

            data = self.wiener_filter()
            
            peaks = self.id_peaks()
            
            t = np.arange(0,np.shape(data)[0]) / self.experiment.fs
        
            legends = []
            for sensor, counter in zip(self.experiment.sensors,range(len(self.experiment.sensors)-self.experiment.__ncam__)):
                legends.append(sensor['serial'])
            plot = plt.plot(t,data)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend(legends)
            # plt.gca().set_prop_cycle(None)       
            for (ids,amp) in zip(peaks[0], peaks[1]):
                plt.plot(np.array(ids)/(self.experiment.fs),amp,'.')   
            
            return plot
#-----------------------------CNN-----------------------------------------=
    def calculate_cwt(self, data, wavelet='morl', scales =np.arange(1, 128), dt=1):
        """
        Calculate the Continuous Wavelet Transform (CWT) of the input data.
        This is meant ot be use with the CNN classifer.
    
        Args:
        - data (array_like): Input data.
        - wavelet (str or callable): The wavelet function. Can be a string (e.g., 'morl') or a callable function.
        - scales (array_like): 1-D array of scales to use.
    
        Returns:
        - coefficients (ndarray): 2-D array of wavelet coefficients.
        """
        from scipy.signal import cwt, morlet
    
        if isinstance(wavelet, str):
            wavelet = morlet
        coefficients = cwt(data, wavelet, scales)
        frequencies = np.fft.fftfreq(scales.size, dt)
    
        return coefficients, frequencies

    def create_image_with_cwt(self):    
        
        """
        Creates a temporal image of the data after calculating the coefficients
        using the Continuous Wavelet Transform (CWT).
    
        Parameters
        ----------
        data_trial : array_like
           inpit data
        
        Returns
        -------
        path_tmp_image : TYPE
            path to image
    
        """        
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        import os
        import matplotlib.pylab as plt
        import numpy as np
        
        data_trial = self.get_data()
        
        time = np.linspace(0, data_trial.shape[0]/self.experiment.fs, data_trial.shape[0])
        # Normalize data
        numeric_columns =list(range(data_trial.shape[1]))
        
        data_to_normalize = pd.DataFrame(data_trial, columns=numeric_columns)
        
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        
        # Fit and transform the data
        normalized_data = scaler.fit_transform(data_to_normalize)
        
        # Replace the original numeric columns with the normalized values
        data_to_normalize[numeric_columns] = normalized_data    
        
        #data normalized turned to array again
        data_to_feed = data_to_normalize.to_numpy()
        
        integ_data =np.sum(data_to_feed,axis=1)
    
        coefficients, frequencies = self.calculate_cwt(data=integ_data, dt=1/self.experiment.fs)
        
        current_dir = os.getcwd()
        path_tmp_image = os.path.join(current_dir, 'cnn_tmp_image.png')
    
        plt.figure(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(coefficients), extent=[0, len(time)/self.experiment.fs, frequencies[-1], frequencies[0]],
               aspect='auto', cmap='magma', interpolation='nearest', norm=plt.Normalize(vmin=0, vmax=np.max(np.abs(coefficients))))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path_tmp_image,bbox_inches='tight',pad_inches=0)       
        plt.close() 
         

    def cnn_clean_gait_classify(self,model='',
                                control_probability=0.9, 
                                drop_table=0,
                                verbose = True):
        
        """ 
        Using a CNN classifier, this method establishes if the trial data
        contains gait information.
        
        Parameters:
        -----------
        model:  Keras CNN model object
                The model trianed for classification
                
        control_probability: float
                 probability lower limit to accept categorical event. E.g 
                 if cnn calssification probability is lower than 
                 control_probability, the record is not considered a gait 
                 record.
        drop_table: int
                    Decision to overwrite the classification in the database 
                    table or to use the classification from the table,
                    if available. 1: Yes, overwrite classification in table. 
                    0: Do not overwrite classification in table.
          
        Returns:
        -----------

        decision (boolean): Decision if there is gait information
        
        """
        from tensorflow.keras.preprocessing import image
        import numpy as np
        import gc
        import os
                        
        # Checking if the classification data is in storage, if not calculate it and retrieve it
        C_sto_out = self.experiment.__sto__.read('cnn_classification',col = 'id', value = self.id)   
        self.predictions = list(C_sto_out[0][1:])
        if drop_table:
            print("Removing table from database...")
            self.experiment.__sto__.drop_table('cnn_classification') 
            self.experiment.__sto__.check('cnn_classification','id integer, decision bool, probability float')
            C_sto_out = self.experiment.__sto__.read('cnn_classification',col = 'id', value = self.id)   

        # If the database is empty for classification, start classification
        if C_sto_out == []:  
            
            # CNN classification
            if model=='':
                model = self.experiment.load_cnn_model()
            

            img_height, img_width = 432, 720
            # Preprocess the new image 
            self.create_image_with_cwt()
            current_dir = os.getcwd()
            path_to_image = os.path.join(current_dir, 'cnn_tmp_image.png')
            
            img = image.load_img(path_to_image, target_size=(img_height, img_width))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

            # Make predictions
            self.predictions = model.predict(img_array, verbose=0)
            
            # Get the predicted class label
            if self.predictions[0][1] >= control_probability:
                decision = True
            else:
                decision = False
            
            row = {'id':self.id, 'decision':decision, 'probability':self.predictions[0][1]}
            self.experiment.__sto__.add('cnn_classification',row)
            gc.collect()
            
            #remove image
            os.remove(os.path.join(current_dir, 'cnn_tmp_image.png'))            
            
        else:
            decision=bool(self.experiment.__sto__.read('cnn_classification',col = 'id', value = self.id)[0])
        
        return decision
    
    
    def clean_gait_classify(self,ampR1=0.0005, ampR2=0.05, meanR1=-5, meanR2=35, stdR1=0.3, stdR2=8,MSER=0.099, micptR1=0.3, micptR2=3, verbose = True):
        """ Checks for clean gait records
        
            Analyzes all channels available
    
            Args:
                to complete
                ampR1 - ampR2
                meanR1 - meanR2
                            #    classify=1                  # Sequence of means
            #    ;    # Amplitude range
            #    meanR1=-5;meanR2=35         # Mean range
            #    stdR1=0.3; stdR2=8           # Std dev range
            #    
            #    micptR1=0.3; micptR2=3           # MICPT range
            
            #    classify=1                  # Sequence of means
            #    ampR1=0.0005; ampR2=0.01   # Amplitude range
            #    meanR1=-5;meanR2=35         # Mean range
            #    stdR1=1; stdR2=8           # Std dev range
            #    MSER=0.061
            #    micptR1=1; micptR2=3           # MICPT range
                ...
                
                Returns:
                decision (boolean): Decision on clean_gait
        """
        # if not self.container.exists('peaks'):
        import numpy as np
        import pandas as pd
        from scipy import optimize
        
        def envelope_function(x, amplitude, mean, stddev):
            return amplitude*(np.exp((-1.0/2.0)*(((x-mean)/stddev)**2)))   
        
        # Checking if the classification data is in storage, if not calculate it and retrieve it
        C_sto_out = self.experiment.__sto__.read('global_classification',col = 'id', value = self.id)        
        
        if C_sto_out == []:
            peaks=self.id_peaks(verbose = verbose)
            
            abs_peaks=[]; abs_time=[]
            for i in range(len(peaks[0])):
                abs_time.append([j/self.experiment.fs for j in peaks[0][i]]) # Index to time (maybe not needed)
#                x=t[sto_out[0]]
                abs_peaks.append(np.abs(peaks[1][i]).tolist()) # Transforming to Absolute peaks
                
            #TODO: create classification plots? another method?

            
            #initial parameters for envelope fitting
            amp1 = 0.01
            mean1 = 10
            stddev1 = 30
            
            parameters=[]
            for i in range(len(peaks[0])):          
                try:
                    popt, _ = optimize.curve_fit(envelope_function, abs_time[i], abs_peaks[i], p0=[amp1, mean1, stddev1])
                    parameters.append(popt)    
                    fit=True
                except:
                    parameters.append(np.array([0,0,0]))
                    fit=False
            
            #% create dataframe for seaborn plot
            amplitude1=[]; amplitude2=[];amplitude3=[];amplitude4=[]
            std1=[];std2=[];std3=[];std4=[]
            mean1=[];mean2=[];mean3=[];mean4=[];
        
                
            amplitude1.append(parameters[0][0])
            mean1.append(parameters[0][1])
            std1.append(parameters[0][2])
        
            amplitude2.append(parameters[1][0])
            mean2.append(parameters[1][1])
            std2.append(parameters[1][2])
        
            amplitude3.append(parameters[2][0])
            mean3.append(parameters[2][1])
            std3.append(parameters[2][2])
            
            amplitude4.append(parameters[3][0])
            mean4.append(parameters[3][1])
            std4.append(parameters[3][2])    
                
                
            columns=['amplitude1','amplitude2','amplitude3','amplitude4'
                     ,'mean1','mean2','mean3','mean4'
                     ,'std_dev1','std_dev2','std_dev3','std_dev4'
                    ]
               
            arraypersons=np.array([amplitude1,amplitude2,amplitude3,amplitude4,
                                   mean1,mean2,mean3,mean4,
                                   np.abs(std1),np.abs(std2),np.abs(std3),np.abs(std4),
                                   ]).T   
            df2 = pd.DataFrame(arraypersons,columns=columns)   
           
            
        
            df=df2
            df2['dif1']=df2['mean1']-df2['mean2']
            df2['dif2']=df2['mean1']-df2['mean3']
            df2['dif3']=df2['mean1']-df2['mean4']
            
            df2['Ndif3']=df2['dif2']/df2['dif3']
            df2['Ndif2']=df2['dif1']/df2['dif3']
            
            df2['MSE']=(((df2['Ndif3']-0.66)**2)+((df2['Ndif2']-0.33)**2))/2
            
            df2['ampRch1']=(df['amplitude1']>ampR1) & (df['amplitude1']<ampR2)              #amplitude range
            df2['ampRch2']=(df['amplitude2']>ampR1) & (df['amplitude2']<ampR2)              #amplitude range
            df2['ampRch3']=(df['amplitude3']>ampR1) & (df['amplitude3']<ampR2)              #amplitude range
            df2['ampRch4']=(df['amplitude4']>ampR1) & (df['amplitude4']<ampR2)              #amplitude range
            
            df2['meanRch1']=(df['mean1']>meanR1) & (df['mean1']<meanR2)                     #mean range      
            df2['meanRch2']=(df['mean2']>meanR1) & (df['mean2']<meanR2)                     #mean range      
            df2['meanRch3']=(df['mean3']>meanR1) & (df['mean3']<meanR2)                     #mean range      
            df2['meanRch4']=(df['mean4']>meanR1) & (df['mean4']<meanR2)                     #mean range      
            
            df2['stdRch1']=(df['std_dev1']>stdR1) & (df['std_dev1']<stdR2)                  #stdmean range      
            df2['stdRch2']=(df['std_dev2']>stdR1) & (df['std_dev2']<stdR2)                  #stdmean range      
            df2['stdRch3']=(df['std_dev3']>stdR1) & (df['std_dev3']<stdR2)                  #stdmean range      
            df2['stdRch4']=(df['std_dev4']>stdR1) & (df['std_dev4']<stdR2)                  #stdmean range      
        
            df2['ICPT1']=np.abs(df2['mean1']-df2['mean2'])
            df2['ICPT2']=np.abs(df2['mean2']-df2['mean3'])
            df2['ICPT3']=np.abs(df2['mean3']-df2['mean4'])
            df2['MICPT']=df2[['ICPT1', 'ICPT2', 'ICPT3']].mean(axis=1)
           
            df2['micptR']=(df2['MICPT']>micptR1) & (df2['MICPT']<micptR2)
        
            df2['MSER']=(df2['MSE']<MSER)
            df2['CGR']=df2['ampRch1'] & df2['ampRch2'] & df2['ampRch3'] & df2['ampRch4'] \
                     & df2['meanRch1'] & df2['meanRch2'] & df2['meanRch3'] & df2['meanRch4'] \
                     & df2['stdRch1'] & df2['stdRch2'] & df2['stdRch3'] & df2['stdRch4'] & df2['MSER'] & df2['micptR']
            
            #temp solution to nan values not stored in storage
            if fit==False:
                df2['Ndif3']=0
                df2['Ndif2']=0
                df2['MSE']=100 #has to be high, zero is the perfect value
            decision=df2['CGR'].bool()

            
            row = {'id':self.id}
            row.update(df2.to_dict('index')[0]) #adds the dataframe to the dict
            self.experiment.__sto__.add('global_classification',row)

        else:
            decision=bool(self.experiment.__sto__.read('global_classification',col = 'id', value = self.id)[0][37])
        
        return decision
    
    def plotCGR(self,width = 11, height = 15):
        '''
        Plots the signals of each channel in a sub-plot and shows the fitting of
        the Gaussian function on each record.  The plot shows the parameters
        of the fitting function
        
        Parameters
        ----------
        width : float, optional
            The width of the figure in inches (default 11)
        height : float, optional
            The height of the figure in inches (default 15)

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure handle
        '''
        import matplotlib.pylab as plt
        import numpy as np
        ampR1=0.0005; ampR2=0.05; meanR1=-5; meanR2=35; stdR1=0.3; stdR2=8;MSER=0.099; micptR1=0.3; micptR2=3 #TODO retrieve parameters from classify
        
        
        # TODO: This needs to change - the parameters should be an output of clan_gait_classify()
        Clean = self.clean_gait_classify()
        C_sto_out = self.experiment.__sto__.read('global_classification',col = 'id', value = self.id) 
            
        def gaussian(x, amplitude, mean, stddev):
            return amplitude*(np.exp((-1.0/2.0)*(((x-mean)/stddev)**2))) 
        
        record = self.wiener_filter()
        recordName=self.get_specific_parameter()
        
        t = np.arange(0,np.shape(record)[0]) / self.experiment.fs
        peaks=self.id_peaks()
        
        nchan = record.shape[1]
        fig, ax = plt.subplots(nchan)
        fig.set_size_inches(width,height)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(nchan):  # For each channel
            s = np.abs(record[:,i])
            top_ids = peaks[0][i]
            popt = [C_sto_out[0][i+1],C_sto_out[0][i+1+nchan],C_sto_out[0][i+1+2*nchan]]
            
            ax[i].plot(t,record[:,i],colors[i],alpha=1,label="%s"%self.experiment.sensors[i]['serial'])
            ax[i].plot(t[top_ids], s[top_ids], markerfacecolor='None', markeredgecolor='black',marker='o',linestyle = 'None')
            ax[i].fill_between(t,0 , gaussian(t, *popt),color=colors[i],alpha=0.15)
            ax[i].legend(loc="lower left")

            ax[i].text(0.2, 0.01, 'A='+'%.4f'%popt[0]+' $t_{max}$='+'%.2f'%popt[1]+' $s$='+'%.2f'%abs(popt[2]), transform=ax[i].transAxes)
            
            ampRch=True if popt[0]>ampR1 and popt[0]<ampR2  else False             #amplitude range
            meanRch=True if popt[1]>meanR1 and popt[1]<meanR2 else False           #mean range 
            stdRch =True if abs(popt[2])>stdR1  and abs(popt[2])<stdR2  else False #stdmean range     
            
            ax[i].text(0.6, 0.01, 'Ar='+str(ampRch)+' $t_{max}$r='+str(meanRch)+' $s$r='+str(stdRch), transform=ax[i].transAxes)

        if Clean:
            ax[0].text(0.05, 0.95, 'Clean Gait Record='+str(Clean), transform=ax[0].transAxes,bbox={'facecolor':'green', 'alpha':0.5, 'pad':2})
        else:
            ax[0].text(0.05, 0.95, 'Clean Gait Record='+str(Clean), transform=ax[0].transAxes,bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        # ax1.text(0.05, 0.90, 'MSE range='+str(MSET))     
        # MICPTR=C_sto_out[0][35]
        # ax[0].text(0.05, 0.88, 'micptR='+str(MICPTR), transform=ax[0].transAxes) 
      
                    
        fig.tight_layout()
        fig.suptitle('Acceleration Record '+ str(recordName)[2:-1] + ' idx'+ str(self.id))
        return fig
        
    def zone_peaks(self, scale=1):
        '''
        Retrieves the peaks around the calculated fitting function (Time:t_{max} and s:spread)
        
        Parameters
        ----------
        scale : float, optional
            Multiplier that shrinks or enlarges the region to retrieve the peaks

        Returns
        -------
        peaks : array
            Peaks contained on the zone defined
        '''
        
        # TODO: This needs to change - the parameters should be an output of clan_gait_classify()
        Clean = self.clean_gait_classify()
        C_sto_out = self.experiment.__sto__.read('global_classification',col = 'id', value = self.id) 

        peaks=self.id_peaks()
        
        nchan = len(peaks[0])
        zone_peaks=[]
        for i in range(nchan):  # For each channel
            top_ids = peaks[0][i]
            popt = [C_sto_out[0][i+1],C_sto_out[0][i+1+nchan],C_sto_out[0][i+1+2*nchan]]          
            t_max_idx=int(popt[1]*self.experiment.fs)
            spread_idx=int(popt[2]*self.experiment.fs)
            zone_peaks.append([value for index,value in enumerate(top_ids) if t_max_idx-spread_idx*scale <= value <= t_max_idx+spread_idx*scale])
        return zone_peaks
            
            
#%%
class TrialShaker(TrialProcess):
    
    """
    Creates a trial for the split data of a shaker test
    """
    
    def __init__(self, experiment, ident = -1, sloc=None,seloc=None):
        TrialProcess.__init__(self, experiment, ident)
        self.sloc = sloc
        self.seloc = seloc
        
    def get_specific_parameter(self, parameter='Node-ID'):
        
        """
        Gives the specific parameters of the trial
        
        Parameter
        ---------
        Name of the specific parameter
        
        Return
        ---------
        Value of the specific parameters for the specific trial
        """

        if parameter=='Node-ID':
            value = self.seloc
        elif parameter == 'shaker location':
            value = self.sloc

        return value
        
            
#%%
class ExperimentProcess(Experiment):
    """
    This class contains:
        1. Post processing methods to enhace the signals of the experiment. 
        2. Methods to calculate the transfer function estimates using input and output signals from the same
        record.
        3. A method to built the frequency response function matrix 
        4. A method to perform systemID (frequency, modes and damping) using pyEMA module
    
    Attributes
    ----------
    Inherited attributes from Experiment class: 
        
        fs : float
            Sampling frequency in Hz (Default 1652 Hz)
        record_length : float
            Record length in seconds (Default 30 seconds)
        pre_trigger : float
            Time before trigger (Default 10 seconds)
        sensors : list
            List of dictionaries with information about each sensor.  Each
            dictionary has the following fields
        
        +-------------------+------------------------------------------+
        | model             | Sensor's model (str)                     |
        +-------------------+------------------------------------------+
        | serial            | Sensor's serial number (str)             |
        +-------------------+------------------------------------------+
        | sensitivity       | Sensor's sensitivity (float)             |
        +-------------------+------------------------------------------+
        | sensitivity_units | Sensitivity units (e.g. 'mV/g') (str)    |
        +-------------------+------------------------------------------+
        | location          | Location of the sensor. List of 3 floats |
        +-------------------+------------------------------------------+
        | location_units    | Location units (e.g. 'm')                |
        +-------------------+------------------------------------------+
        | direction         | Direction.  List of 3 floats             |
        +-------------------+------------------------------------------+
        | trigger           | Level that will trigger data collection  |
        +-------------------+------------------------------------------+
        
        parameters : dict
            List of parameters to be added to the HDF5.  The dictionary has two
            keys: 'general' and 'specific'.  General parameters are those that 
            describe the whole experiment such as a title for the experiment.
            Specific parameters are specific for each record.  This is useful when
            each record needs to have specific variables.  For example, when
            collecting data from human excitation and the age or weight of the
            person is to be collected and documented.
        title : str
            Experiment's title
        fname : str
            String containing the path to the hdf5 file
    
    Methods
    -------
    load_setup(fname='setup.json')
        Load configuration of experiment stored in file 'fname'
        
    resample(fs)
        Resamples all trials from experiment using the new fs
        
    filtered(cutoff_freq, ripple, width, zero_phase)
        Filters the data from the experiment using a low pass filter
    
    center_data(size, ref_sensor)
        Center the data of the experiment for a record length equal to 2*size
    
    split_data(ref_sensor, window, drop_keys, plot)
        Splits all trials in the experiment data into individual records
    
    load_data()
        Creates the record attribute for each trial in the experiment data
        
    tfes_data(self, force_sensor, fft_n, window, noverlap_perc, tf_method)
        Calculates the tranfer function for all the trials in the experiment 
        data.
        
    frf_matrix_for_systemID()
        Builts the frequency response function matriz using the experiment
        data with the multiple input x outputs.
        
    system_id_using_openmodal()
        Use the experiment data for system identification (natural frequencies,
       mode shapes and damping ratios) using the pyEMA module.
    """
      
    def __init__(self,title = '', fs = 1652, record_length = 30, pre_trigger = 10, sensors = [], parameters = {}):
        # Importing shelve for temporary files
        import shelve 
        import tempfile
        import os
        
        Experiment.__init__(self, title, fs, record_length, pre_trigger, sensors, parameters)
        
        # Open temporary file to store data
        file_exists = True
        while file_exists:
            tmp_fname = tempfile._get_default_tempdir() + '\\'  + next(tempfile._get_candidate_names())
            file_exists = os.path.exists(tmp_fname)
            
        self.tmp_data = shelve.open(tmp_fname)
        
    def __del__(self):
        # Close temporary file
        self.tmp_data.close()

    def load_setup(self,fname='setup.json',data_folder = 'data',populate_trials = True):
        '''
        Opens the JSON file containing the setup parameters for the experiment.
        This function is overloaded from Experiment to load the trials as
        TrialProcess functions
        
        Parameters
        ----------
        fname : str
            File that the parameters for the experiment were saved into (JSON file)
            
        data_folder : str
            Path to the folder that contains the hdf5 file.  The default value
            is 'data'.  Do not use trailing slash in the path
            
        populate_trials : boolean
            If True Experiment.trials will be populate with the trials in the file.
            By default this is true

        '''
        
        Experiment.load_setup(self,fname = fname, data_folder = data_folder, populate_trials = False)
        
        self.__sto__ = st.Storage('%s\\%s.db'%(data_folder,self.title))
        # Checking on the peaks table
        self.__sto__.check('peaks','id integer, Channel integer, Location integer, Magnitude double') #Table for peaks
        self.__sto__.check('global_classification', \
                           'id integer, \
                           amplitude1 integer, amplitude2 integer, amplitude3 integer, amplitude4 integer, \
                           mean1 integer, mean2 integer, mean3 integer, mean4 integer, \
                           std_dev1 integer, std_dev2 integer, std_dev3 integer, std_dev4 integer, \
                           dif1 integer, dif2 integer, dif3 integer, \
                           Ndif3 integer, Ndif2 integer,\
                           MSE integer, \
                           ampRch1 bool, ampRch2 bool,ampRch3 bool,ampRch4 bool, \
                           meanRch1 bool, meanRch2 bool, meanRch3 bool, meanRch4 bool, \
                           stdRch1 bool, stdRch2 bool, stdRch3 bool, stdRch4 bool, \
                           ICPT1 integer, ICPT2 integer, ICPT3 integer, \
                           MICPT integer, \
                           micptR bool, MSER bool, CGR bool') #Table global parameters   
        self.__sto__.check('cnn_classification','id integer, decision bool, probability float')

        
        if populate_trials:
            import os
            import tables
            
            self.trials = []
            if os.path.isfile(self.fname):
                self.__efile__ = tables.open_file(self.fname, mode="r")
                data = self.__efile__.get_node('/experiment/data')
                for counter in range(data.shape[0]):
                    self.trials.append(TrialProcess(self,ident = counter))
        
        # Create SpacePrior
        # Loading file to get sprior data (if available)
        with open(fname, 'r') as setup_file:
            setup_data = json.load(setup_file)
        
        if 'sprior' in setup_data.keys():
            # If sprior exists in the setup file
            self.sprior = SpacePrior(parent = self, vertex = setup_data['sprior']['domain']['vertex'],
                                     units = setup_data['sprior']['domain']['units'])
            # If polygons are defined in the file
            if 'polygon' in setup_data['sprior'].keys():
                for polygon in setup_data['sprior']['polygon']:
                    self.sprior.addpolygon(polygon['vertex'], polygon['logp'], polygon['units'],label = polygon['label'])
            
        else:
            # If prior does not exist, add 10 meters around the sensors as domain
            coords = np.array([sensor['location'] for sensor in self.sensors if sensor['sensor_type'] == 'Accelerometer'])
            maxc = np.max(coords,axis = 0)
            minc = np.min(coords,axis = 0)
            if self.sensors[0]['location_units'][0:2] == 'in':
                # Adding 10 meters box
                maxc = maxc + 10*39.37
                minc = minc - 10*39.97
            else:
                raise Exception('Only inches are used for sensor locations at this time')
            vertex = [(minc[0],minc[1]),(maxc[0],minc[1]),(maxc[0],maxc[1]),(minc[0],maxc[1]),(minc[0],minc[1])]
            self.sprior = SpacePrior(parent = self, vertex = vertex,units = self.sensors[0]['location_units'])

    def status(self):
        '''
        Returns the status of the Experiment in a pandas
        dataframe.
            - Trials: Number of trials
            - Peaks: Number of trials with peaks calculated
            - CNN Classify: Number of trials that have been
                classified using cnn_clean_gait_classify()

        Returns
        -------
        df : pandad.DataFrame
            Status of the experiment.  One row with the following
            columns:
            - Trials: Number of trials
            - Peaks: Number of trials with peaks calculated
            - CNN Classify: Number of trials that have been
                classified using cnn_clean_gait_classify()

        '''
        queries = {}
        out = {}
        out['Experiment'] = self.title
        out['Trials'] = [len(self.trials)]
        queries['Peaks'] = 'select count(distinct id) from peaks'
        queries['CNN Classify'] = 'select count(distinct id) from cnn_classification'
        for key in queries.keys():
            cursor = self.__sto__.__cursor__.execute(queries[key])
            out[key] = [cursor.fetchall()[0][0]]

        df = pd.DataFrame(out)        
        df.set_index('Experiment')
        return df

    def load_shaker_data(self,shaker_accel_sensor_idx, remove_locations=['NA1','NA2']):
        
        """
        This function loads the data from a shaker test with multiple sensors 
        at different locations. To use this function, the specific parameters
        of the dataset should include:
            sensor locatio : location of the sensors in each of the trials.
            
        Parameters
        ----------
        shaker_accel_sensor_idx: float
                                The index of the accelerometer, if any, located
                                on top of the shaker. This accelerometer
                                is considered the force sensor.
        remove_locations:       list
                                List of location names that should not be considered
                                in the dataset. These locations can be related to
                                damage data or erroneous trials.
                            
            
        Returns
        ----------
        None
        """
        
        
        from tqdm import tqdm
        import numpy as np
        import gc
        
        #Extracting the sensors that are not output signals
        sensors = pd.DataFrame(self.sensors).drop(shaker_accel_sensor_idx)
        sensors = list(sensors[sensors['serial'] != 'SMO'].index)
        
        data_df = {'shake-location':[], 'sensor':[], 'data':[]}
        force_signals = []
        trials_id = []
        for trial_i in tqdm(self.trials):
            #get specific parameter
            tmp = trial_i.get_specific_parameter(parameter='sensors locatio')
            if tmp != 'Eliminar':
                tmp_sl = trial_i.get_specific_parameter(parameter='shaker location')

                locs =tmp.split('[')[1].split(']')[0].split(',')
                #add ifnormation to dictionary
                for sen_i in range(len(sensors)):
                    data_df['shake-location'].append(tmp_sl)
                    data_df['sensor'].append(locs[sen_i])
                    data_df['data'].append(trial_i.data[:,sensors[sen_i]])
                    force_signals.append(trial_i.data[:,shaker_accel_sensor_idx])
                    trials_id.append(trial_i.id)
                    
                    
        #create dataframe
        data_DF = pd.DataFrame(data_df)    
            
        #unique locations list         
        unique_locs = list(data_DF['sensor'].unique())  
        
        #remove specified locations form unique locations list
        for removi in remove_locations:
            unique_locs.remove(removi)
            
        #filter data by location  and add the array to a new set of trials
        self.trials = []
        count=0
        for uni_loc_i in tqdm(unique_locs):
            data_tmp = np.array(list(data_DF[data_DF['sensor']==uni_loc_i]['data'].to_numpy()))
            data_all = np.insert(data_tmp, 0, force_signals[count], axis=0)
            
            sloc = data_DF[data_DF['sensor']==uni_loc_i]['shake-location'].unique().tolist()
            seloc = uni_loc_i
            
            new_trial = TrialShaker(self,ident=trials_id[count], sloc=sloc, 
                                    seloc=seloc)
            new_trial.data = data_all.T
            count += 1            
            self.trials.append(new_trial)
            gc.collect()

    def resample_data(self,fs):
        """
        Resample all the trials of the experiment using a new
        sampling frequency
        
        Parameters
        ----------
        fs:     int  
                new sampling frequency
        Returns:
        --------        
        None
        """
        from tqdm import tqdm
        #Resampling each trial
        for trial in tqdm(self.trials):
            trial.__resample_data__(fs)
            
        self.fs = fs # data new sampling frequency
        
    def filter_data_lowpass(self, fs, cutoff_freq, order=5):
        
        """
        Filter the data using a Impulse Response Filter for all trials in
        the experiment.
        
        Please refer to this method in scipy:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
        
        1. Design an Nth-order digital or analog Butterworth filter and return 
        the filter coefficients.
        2. filters the data according to the filter coefficients
        
        Parameters:
        ----------
        order        :int 
                     The order of the filter
        
        fs          :int
                     The sampling frequency of the digital system.
                     
        cutoff_freq: int
                    For a Butterworth filter, this is the point at 
                    which the gain drops to 1/sqrt(2) that of the passband 
                    (the “-3 dB point”). Value of Frequency in Hz. All 
                    frequencies after this value are removed.
        
        Returns:
        ----------
        None
        
        """    
        
        from tqdm import tqdm
        #Filtering each trial
        for trial in tqdm(self.trials):
            trial.__filter_data_lowpass__(fs=fs, cutoff_freq=cutoff_freq , order=order)
            
    def filter_data_FIR(self, cutoff_freq, ripple=10, width=10, zero_phase=False): 
        """
        Runs all the trials from the experiment to a lowpass finite impulse
        response filter.
        
        Parameters
        ----------
        cutoff_freq     : integer 
                        Data above this frequency in Hz will be removed                 will be removed.
        
        ripple          : integer 
                        value in decibels, giving less or more attenuation 
                        of wanted signals.
        
        width           : integer 
                        Transition width, given in Hz.
                        
        zero_phase:     bool
                        If True:
                        Apply a digital filter forward and backwards to a signal.
                        This function applies a linear digital filter twice, 
                        once forward and once backwards. The combined filter 
                        has zero phase and a filter order twice that of the original.
                        
                        If False:
                        Filter data along one-dimension with an IIR or FIR filter.
                        
        filtfilt: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        lfilter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                        
             
        Returns:
        --------        
        None
        """
        from tqdm import tqdm
        #Filtering each trial
        for trial in tqdm(self.trials):
            trial.filter_data_FIR(cutoff_freq, ripple, width, zero_phase=False)
    
    def center_data(self, size, ref_sensor):
        """
        Centers all the trials of the experiment in the time domain. To fit 
        the size, the signal is padded with zeros symmetrically at the beginning 
        and at the end. All sensors from the same trial are centered using the 
        ref sensor.
        
        Parameters
        ----------
        size            :int
                        Desired amount of points left and right 
                        of impact hammer peak
                        
        ref_sensor      : int
                        Reference sensor used to cut all the other
                        sensors records at the same location, 
                        usually, the force sensor 0 if available

        Returns
        -------
        None.

        """
        from tqdm import tqdm
        #Filtering each trial
        for trial in tqdm(self.trials):
            trial.center_data(size, ref_sensor)
    

    def split_data(self, ref_sensor, window, droop_keys, plot=False) :
        
        """
        For a calibration test with multiple impacts in the same record,this
        function splits each trial in multiple records.        

p.tf
        plot:             bool
                          If True, it will plot a single record with the multiple
                          events and color bands of the splitting records. 
                          This can be used to check if the splitting was achived
                          as expected.
        
        Returns
        -------
        None.

        """ 
        from tqdm import tqdm
        #Resampling each trial
        tmp_trials = []
        cont = 0
        for trial in tqdm(self.trials):
            
            if not trial.get_specific_parameter('Node-ID') in droop_keys:
                trial_records = trial.split_data(ref_sensor,window, plot)
                for trial_rec in trial_records:
                    new_trial = TrialProcess(self,ident =trial.id)
                    new_trial.data = trial_rec.T
                    tmp_trials.append(new_trial)
            cont = cont+1
            
        self.trials = tmp_trials

    def tfes_data(self, force_sensor, window_func=None, 
                  noverlap_perc =None, tf_method=3, fft_n='', fs='', 
                  remove_sensors=None):
        
        """
        Gets the Transfer function for all the trials in the experiment data
        
        Parameters
        -------
        force_sensor:   float
                        indx of the force sensor 
                        
        fft_n:           int
                        The number of data points used in each block for the 
                        FFT. A power 2 is most efficient. 
                        The default value is the length of the record.          
        
        window_func:          A vector of length NFFT. 
                         Possible functions to use to create the array:
                         numpy.blackman, 
                         numpy.hamming, numpy.bartlett, scipy.signal, 
                         scipy.signal.get_window, etc. If a function 
                         is passed as the argument, it must take a data segment
                         as an argument and return the windowed version of the 
                         segment.
                         
        noverlap_per    :int
                        Percentage of points to overlap between segments.The 
                        default value is None (no overlap). 
                        
        tf_method:      float
                        check tfes_data in trial class for details on the 
                        different methods.
                        
                        
        remove_sensors  :list
                        A list of sensors to ignore during the contruction of
                        the TFEs.              
        Returns
        -------
        None.

        """
        from tqdm import tqdm
        import gc

        #Resampling each trial
        for trial in tqdm(self.trials):
            trial.tfes_data(force_sensor, window_func=window_func, 
                          noverlap_perc=noverlap_perc, tf_method=tf_method,
                          fft_n=fft_n, fs=fs, remove_sensors=remove_sensors)
            
            gc.collect()
            
    def tfes_data_plot(self, limits, loc_parameter_name):
        
        """
        Creates a folder with the TFs plots of the experiment trials.
        
        Parameters
        -------
        limits:  list
                A list conatining the minimum and maximum x values on the plot
        loc_parameter_name: str
                            Name of the parameter in the specific parameters 
                            that belongs to the location of the force.
        None 
              
        Returns
        -------
        None.

        """
        
        import os
        import matplotlib.pylab as plt
        import numpy as np
        
        path = '%s/TFE_plots'%self.fname.split('.')[0]
        if not os.path.exists(path):
            os.makedirs(path)
        
        from tqdm import tqdm
        #Resampling each trial
        count = 0
        for trial in tqdm(self.trials):
            location = trial.get_specific_parameter(loc_parameter_name)
            plt.figure()
            for sen_i in range(len(self.tfe_sensors)):
                plt.plot(trial.freq, np.log(trial.tfe[:,sen_i]), label=self.tfe_sensors[sen_i]['serial'])
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Log(TFE)')
                plt.title('Force location: %s'%location)
                plt.xlim(limits)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(path, 'TFE-%s-%s.pdf'%(location,count)))
            plt.close()
            count +=1
    
    def frf_matrix_for_systemID(self):
        """
        Builts the matrix of frequency response functions. The format of the 
        matrix is required when using the "open-modal" module for system 
        identification. Inside this function also the arrays with the x and y cooridnates
        of the experimental impact locations and the sensors (accelerometers) locations are
        extracted.
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        import numpy as np
        from tqdm import tqdm
        import ast
        
        
        #get list of unique locations
        locations = []
        XY_TF_loc_i =[]
        for trial_i in tqdm(range(len(self.trials))):
            locations.append(self.trials[trial_i].get_specific_parameter('Node-ID'))            
            
            value = ast.literal_eval(self.trials[trial_i].get_specific_parameter('coordinates'))
            XY_TF_loc_i.append(value) 
        
        indexes = np.unique(locations, return_index=True)[1]
        loc = [locations[index] for index in sorted(indexes)]
        
        
        hits = len(loc)
        sens = self.trials[0].tfe.shape[1] 
        FRF_data = np.ndarray((hits,sens,int(self.trials[0].tfe.shape[0])), dtype=complex)
        
        XY_TF_loc_i =[]
        
        for idx in range(len(loc)):
            loc_i = loc[idx]
            loc_indx = np.where(np.array(locations)==loc_i)[0]
            tmp=0
            for indx in loc_indx :
                tmp += self.trials[indx].tfe
            
            TF_loc_i = np.divide(tmp, len(loc_indx))
            value = ast.literal_eval(self.trials[indx].get_specific_parameter('coordinates'))
            
            XY_TF_loc_i.append(value)
            
            for sen_idx in range(TF_loc_i.shape[1]):
                
                FRF_data[idx,sen_idx,:] = TF_loc_i[:,sen_idx]

        self.FRF_matrix = FRF_data
        self.freq = self.trials[0].freq
        
        #Extracting impacts location x-y
        self.XY_locations = np.array(XY_TF_loc_i);
        self.unique_locations = loc
        
        #Extracting sensors locations x-y
        XY_sensor_i = []
        for sensor_i in range(len(self.tfe_sensors)):
            sen_x = self.tfe_sensors[sensor_i]['location'][0]
            sen_y = self.tfe_sensors[sensor_i]['location'][1]
            XY_sensor_i.append([sen_x, sen_y])
            
        self.XY_sensors = np.array(XY_sensor_i)
        
    def system_id_using_openmodal(self, lower=0.01, upper=200, 
                                  pol_order_high=60):
            """
            System identification using the module OpenModal 
            https://pypi.org/project/pyEMA/
            This module provides estimations of the natural frequencies, damping 
            rations and mode shapes of the system.
            
            Parameters
            -------
            lower:     float
                       Lower limit for pole determination [Hz]
                       
            upper:     float
                       Upper limit for pole determination [Hz]
                       
            pol_order_high: float
                      Highest order of the polynomial
        
            Returns
            -------
            None
    
            """
           
            import pyEMA
      
            selected_response = 0
            FRF =  self.FRF_matrix[:,selected_response,:]
            
            acc = pyEMA.Model(frf=FRF, 
                     freq= self.freq,
                     lower=lower, 
                     upper=upper,
                     pol_order_high=pol_order_high)
            acc.get_poles()        
            acc.select_poles() 
                   
            natural_freq = acc.nat_freq
            mode_shapes = acc.normal_mode()
            damping = acc.nat_xi
            
            self.natural_freq = natural_freq
            self.mode_shapes = mode_shapes
            self.damping = damping
            
    def gaussian_regression_processor(self, xy, z, x_pred, y_pred):
        
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
            from itertools import product           
            import numpy as np
        
            
            # Instantiate a Gaussian Process model
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            
            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(xy, z)
            
            
            if len(x_pred)==1:
                x1y1 = np.array([x_pred, y_pred]).reshape(1,-1)                
            
            else:             
                x1y1 = np.array(list(product(x_pred, y_pred)))

            
            mean,std = gp.predict(x1y1, return_std=True) 
            
            return mean, std, x1y1

    def get_noise_threshold(self):
        """
        Calculates the noise level of the combined signals between sensors
        for all trials in the experiment class
                
        Parameters
        ---------- 
        None

        Returns:
        --------        
        None
        """
        from tqdm import tqdm
        #Resampling each trial
        for trial in tqdm(self.trials):
            trial.get_noise_threshold()
            
            
    def clean_gait_classify(self, verbose = False):
        """
        Classify the records as clean gait records or not

        Returns
        -------
        classification : List of booleans with the classification

        """
        classification = []
        from tqdm import tqdm
        for trial in tqdm(self.trials):
            classification.append(trial.clean_gait_classify(verbose = verbose))
        
        return classification
    
    def load_cnn_model(self):
        
        import tensorflow as tf
        # Load the trained model
        model = tf.keras.models.load_model('image_classifier_model_4.h5')
        
        return model
    
    def cnn_clean_gait_classify(self, drop_table=0, control_probability=0.8, verbose = False):
        """
        Classify acceleration signals to estabish if there is any gait information.
        The classification is made using CNN classifier.
        
        Parameters:
        -------        
                
        control_probability: float, default value 80%
                 probability lower limit to accept categorical event. E.g 
                 if cnn calssification probability is lower than 
                 control_probability, the record is not considered a gait 
                 record.
        
        
        Returns
        -------
        classification : List of booleans with the classification

        """
        classification = []
        from tqdm import tqdm
        import gc

        classification_data = self.__sto__.read_all('cnn_classification')
        classification_data = {t[0]: t[1] for t in classification_data}
        # Loading model
        model = self.load_cnn_model()
        
        for trial in tqdm(self.trials):
            tmp = classification_data.get(trial.id)            
            # If not found in the database, classify it
            if tmp is None:
                tmp = trial.cnn_clean_gait_classify(model=model, drop_table=drop_table,
                                              control_probability=control_probability,
                                              verbose = verbose)
                gc.collect()

            classification.append(tmp)
                    
        return classification