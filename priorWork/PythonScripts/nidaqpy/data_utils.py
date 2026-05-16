# -*- coding: utf-8 -*-
"""
Data utils
"""
import numpy as np

import numpy
#from nidaqmx import *
from scipy.signal import chirp#, spectrogram
from scipy import signal as scisig
import tables as tb
import matplotlib.pylab as plt
import pandas as pd
import matplotlib.mlab as mlab
import warnings
warnings.filterwarnings("ignore")
#%% Other
def comp_std(data):
    """ Compare std. dev. for 2000 data points at beginning and at the end
        it only analyzes the first channel available
    
        Args:
        data (numpy array): acceleration data
        
        Returns:
        decision (boolean): Decision.
    """
    b_std=np.std(data[:2000,1])
    e_std=np.std(data[-2000:,1])
    
    if e_std>b_std*1.2:
        decision=False
        print('Dismissed by comp_std criteria')
    elif b_std>e_std*1.2: #backwards comparison
        decision=False    
        print('Dismissed by comp_std criteria')
    else:
        decision=True
    
    return decision

def noise_dev(data):
    """ Calculates the std deviation and compares to given value inside
        this function
        it only analyzes the first channel available
    
        Args:
        data (numpy array): acceleration data
        value (double): Std. Dev. Threshold
        Returns:
        decision (boolean): Decision.
    """
    n_std=np.std(data[:,3])
    value=0.0023
    if n_std>value:
        decision=False
        print('Dismissed by noise std. dev. criteria')
    else:
        decision=True
    
    return decision

def noise_dev3(data):
    """ Calculates the std deviation on 3 sections of the current record
        and compares to given value inside, if all 3 values are in range
        the record gets a False decision
        
        it only analyzes the first channel available
    
        Args:
        data (numpy array): acceleration data
        value (double): Std. Dev. Threshold
        Returns:
        decision (boolean): Decision.
    """
    
    n_std1=np.std(data[:2000,3])
    n_std2=np.std(data[int((len(data[:,3])/2)-1000):int((len(data[:,3])/2)+1000),3])
    n_std3=np.std(data[-2000:,3])
    
    value=0.00105
    if n_std1>value and n_std2>value and n_std3>value:
        decision=False
        print('Dismissed by 3 point std. dev. criteria')
    else:
        decision=True
    
    
    return decision    

def rem50(data):
    """ Sorts the record from smallest to largest and removes 50% of the data
        
        it only analyzes the first channel available
    
        Args:
        data (numpy array): acceleration data
        Returns:
        decision (boolean): Decision.
    """
    channel=0
    sort=sorted(data[:,channel])
    n_std1=np.std(sort[int(len(sort)/4):int(len(sort)*3/4)])
    
    
    
    
    value=11.788079e-05 #SortedData and 50 compressor OFF.png 2+
    if n_std1>value:
        decision=False
        message='Dismissed by rem50 criteria'
        print(message)
    else:
        decision=True
    
    
    return decision    

class buffer:
    """
    Creates a buffer to be used in data collection
    #TODO: Complete help
    """
    pretrig = 0   # Number of points in the buffer for pre-trigger
    trigger = 0   # Amplitude used for trigger
    __buffer__ = []  # Buffer itself
    __position__ = 0 # Current position to add to the buffer
    
    def __init__(self,buffer_size):
        """This class is an object to serve as a buffer for data collection.
        
        Args:
            buffer_size (int): Number of points for the buffer. Corresponds to the
                number of rows
                
        Attributes:
            buffer_size (int): Number of points for the buffer.
        
        """
        self.__buffer__ =  [0] * buffer_size
        self.buffer_size = buffer_size
        
    def add(self,data):
        """Adds data to a buffer object.
        
        Args:
            data (list): List to add to the buffer.  The length data should
            be n corresponding to the number of points being added to the
            buffer.  Data can be a list of lists if more than one sensor
            will be used.
        """
        
        # Adding to the buffer
        if self.__position__ + len(data) > len(self.__buffer__):
            # If the data is bigger than the remaining of the buffer
            points_end = len(self.__buffer__) - self.__position__
            points_beginning = len(data) - points_end
            self.__buffer__[self.__position__:] = data[0:points_end]
            self.__buffer__[0:points_beginning] = data[points_end:]
            self.__position__ = points_beginning
        else:    
            # If the data is smaller than the remainder of the buffer
            self.__buffer__[self.__position__:self.__position__+len(data)] = data
            self.__position__ += len(data)

    def read(self):
        """ Reads the buffer
        
        Returns:
            data (list): Data in the correct order.
        """
        data = self.__buffer__[self.__position__:]
        for item in self.__buffer__[0:self.__position__]:
            data.append(item)
            
        return data   

#%% Kaisser_window
def Kaisser_window(signal,fs, beta=28, t_window=30, end=None):
    
    """
    The Kaiser window is a taper formed by using a Bessel function. 
    The function was rewritten to smooth the edges of an output signal when 
    sent to a shaker.
        
    Input
    ------
    signal:     np.array

    fs:         np.array
                sampling rate. Usually an atribute of the class expriment
                
    beta:       float
                Kaiser window beta parameter: Shape parameter, determines 
                trade-off between main-lobe width and side lobe level. 
                As beta gets large, the window narrows            
    Output
    ------
    signal      np.array
                Smoothed signal on the edges
    """
    
    
    window = scisig.kaiser(t_window*fs, beta=beta)#scisig.windows.hann(t_window*e.fs, False)
    limit = int(len(window)/2)
    signal_=signal
    
    if end=="both":
        signal_[0:limit] = signal_[0:limit]*window[0:limit]
        signal_[len(signal_)-limit:] = signal_[len(signal_)-limit:]*window[limit:]
    else:
        signal_[0:limit] = signal_[0:limit]*window[0:limit]
        window2 = scisig.kaiser(int((t_window/2)*fs), beta=beta)
        limit2 = int(len(window2)/2)
                
        signal_[len(signal_)-limit2:] = signal_[len(signal_)-limit2:]*window2[limit2:]        
    
    return signal_
#%% Sine wave
def create_sine(amplitude_peak_voltage, sine_freq_npi_hz, signal_width, fs, 
                pre_trigger, kaisser_beta=28, t_window=30,end="both"):
    """
    This function creates a sine wave signal in which the amplitude units are peak 
    Voltages. The function if meant to be used during shaker tests.
    
    Input
    ------
    amplitude_peak_voltage: float
                            Maximum peak voltage  of the sine wave
                            
    sine_freq_npi_hz:       float
                            Frequency in Hz of the sine wave signal
                
    signal_width:           float
                            length of the signal in seconds, usually the specified
                            record length minus two times the pre-trigger
                        
    fs:                     float
                            sampling frequency, usually an atribute of the class 
                            experiment.
                        
    pre_trigger:            float
                            time before initializing recording. Usually this parameter
                            is specify during setup.
                            
    kaiser_beta:            float
                            Kaiser window beta parameter: Shape parameter, determines 
                            trade-off between main-lobe width and side lobe level. 
                            As beta gets large, the window narrows                   
    Output
    ------
    
    send_signal:            np.array
                            output signal in peak voltage
    
    
    """
    t=  np.linspace(0, signal_width, signal_width*fs)
    send_signal = amplitude_peak_voltage*np.sin(2*np.pi*sine_freq_npi_hz*t)
    send_signal= np.pad(send_signal, int(pre_trigger*fs), mode='constant')
    send_signal = Kaisser_window(send_signal,fs, kaisser_beta,t_window=t_window,
                                 end=end)
    return send_signal

#%% band_limited_noise
def create_band_limited_noise(min_freq, max_freq, amplitude, record_length, 
                              pre_trigger, fs, kaisser_beta=28,t_window = 30,
                              multiple_windows=True,
                              window_time=10, end="both"):
    
    """
    This function creates a band limited noise signal with a range of frequencies
    between min_freq and max_freq and with the maximum amplitude in peak voltage 
    equal to amplitude. The function if meant to be used during shaker tests.

    Input
    ------
    
    min_freq:       float
                    Minimum frequency
                
    max_freq:       float
                    Maximum frequency
                    
    amplitude:      float
                    Maximum value of peak Voltage desired
    
    record_length:  float
                    length of the signal in seconds, usually the specified
                    record length
                    
    fs:             float
                    sampling frequency, usually an atribute of the class 
                    experiment.
                        
    pre_trigger:    float
                    time before initializing recording. Usually this parameter
                    is specify during setup.  
    
    multiple_windows:bool
                    If True, the window of the signal of size window_time will 
                    be repeated until completion of the record length. This 
                    option is included to allow for continuous excitation with
                    intervals of no excitation. If False, the window of the 
                    signal will be equal to the record_length.
                    
    window_time:    float
                    The length of the signal in seconds that will be repeated 
                    until completion of record length.
                    
    kaiser_beta:            float
                            Kaiser window beta parameter: Shape parameter, determines 
                            trade-off between main-lobe width and side lobe level. 
                            As beta gets large, the window narrows              
                    
    Output
    ------
    
    send_signal:            np.array
                            output signal in peak voltage     
                    
                    
    """    
    def fftnoise(f):
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = (np.cos(phases) + 1j * np.sin(phases))
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        return np.fft.ifft(f).real
    
    def band_limited_noise(min_freq, max_freq, samples, samplerate):
        freqs = np.abs(np.fft.fftfreq(samples, 1.0/samplerate))
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
        f[idx] = 1
        return np.array(fftnoise(f),  order='C')       
    
    if multiple_windows:
        total_windows = int((record_length-pre_trigger)/(window_time+(pre_trigger/2)))
        record_length_ = window_time#record_length/total_windows        

    if multiple_windows:
        n_sec = record_length_
        samples = int(fs*n_sec)
        signal = band_limited_noise(min_freq, max_freq, samples, int(fs))
        #signal with pre-trigger/2 seconds rest (1/4 + 1/4)
        send_signal_ = np.pad(signal, pad_width=int(fs*pre_trigger/4), mode='constant') #assemble output signal to match record length
        send_signal_ = Kaisser_window(send_signal_,fs, kaisser_beta, t_window=t_window,
                                      end=end)
        multiple_signal = np.tile(send_signal_,total_windows)
        
        padding = int(record_length*fs-(len(multiple_signal)+int(fs*pre_trigger)))
        output_signal_ = np.pad(multiple_signal,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
        output_signal = (amplitude/max(output_signal_))*output_signal_ 

    else:
        n_sec= record_length-pre_trigger
        samples = int(fs*n_sec)
        signal = band_limited_noise(min_freq, max_freq, samples, int(fs))
        send_signal = Kaisser_window(signal,fs, kaisser_beta, t_window=t_window,
                                      end=end)
    
        padding = int(record_length*fs-(len(send_signal)+int(fs*pre_trigger)))
        output_signal_ = np.pad(send_signal,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
        output_signal = (amplitude/max(output_signal_))*output_signal_ 
#    t=np.arange(0,len(output_signal)/fs,1/fs)
    
    return output_signal
def create_sine_sweep(fs, f0, f1, amplitude, record_length, pre_trigger, 
                      method='linear', kaisser_beta=28, t_window=30,
                      multiple_windows=False,
                              window_time=10, end=None):

    """
    This function creates a chirp signal in which the frequency increases or 
    decreases with time. 
    
    Input
    ------
    fs:             float
                    sampling rate, usually an atribute of the class experiment
    
    f0:             float
                    Frequency (e.g. Hz) at time t=0.

    t1:             float
                    Time at which f1 is specified.

    f1:             float
                    Frequency (e.g. Hz) of the waveform at time t1.
                    
    amplitude:      float
                    Maximum value of peak Voltage desired          
                    
    record_length:  float
                    length of the signal in seconds, usually the specified
                    record length
                    
    pre_trigger:    float
                    time before initializing recording. Usually this parameter
                    is specify during setup.        
                    
    method:         string
                    {‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}, optional
                    Kind of frequency sweep. If not given, linear is assumed. 
                    
    kaiser_beta:    float
                    Kaiser window beta parameter: Shape parameter, determines 
                    trade-off between main-lobe width and side lobe level. 
                    As beta gets large, the window narrows               
    multiple_windows:bool
                    If True, the window of the signal of size window_time will 
                    be repeated until completion of the record length. This 
                    option is included to allow for continuous excitation with
                    intervals of no excitation. If False, the window of the 
                    signal will be equal to the record_length.
                    
    window_time:    float
                    The length of the signal in seconds that will be repeated 
                    until completion of record length.
    The input values of this function are to be used on scipy,signal.chirp function
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
    
    Output
    ------
    
    send_signal:            np.array
                            output signal in peak voltage
    
    """
    
    if multiple_windows:
        total_windows = int((record_length-pre_trigger)/(window_time+(pre_trigger/2)))
        record_length_ = window_time#record_length/total_windows        
    
    if multiple_windows:
        n_sec= record_length_
        t = np.linspace(0, n_sec, n_sec*fs)
        send_signal = amplitude*chirp(t, f0=f0, t1=n_sec, f1=f1,method=method)
        #signal with pre-trigger/2 seconds rest (1/4 + 1/4)
        send_signal_ = np.pad(send_signal, pad_width=int(fs*pre_trigger/4), mode='constant') #assemble output signal to match record length
        send_signal_ = Kaisser_window(send_signal_,fs, kaisser_beta, t_window=t_window, 
                                      end=end, method=method)
        multiple_signal = np.tile(send_signal_,total_windows)
        
        padding = int(record_length*fs-(len(multiple_signal)+int(fs*pre_trigger)))
        output_signal_ = np.pad(multiple_signal,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
        output_signal = (amplitude/max(output_signal_))*output_signal_ 
    
    else:
        n_sec= record_length-pre_trigger
        t = np.linspace(0, n_sec, n_sec*fs)
        send_signal = amplitude*chirp(t, f0=f0, t1=n_sec, f1=f1,method=method)
        send_signal = Kaisser_window(send_signal,fs, kaisser_beta, t_window=t_window, 
                                      end=end, method=method)
        padding = record_length*fs-(len(send_signal)+int(fs*pre_trigger))
        output_signal_ = np.pad(send_signal,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
        output_signal = (amplitude/max(output_signal_))*output_signal_ 

    return output_signal
#%% Special sine sweep
def create_special_sine_sweep(fs, f0, f1, amplitude, record_length, pre_trigger, 
                      method='linear', kaisser_beta=28, t_window_spect = 90, end=None):
                    
    
    """
    
    This function creates a special chirp signal in which the frequency increases or 
    decreases with time. It returns a spectrogram so the user can select the
    time where the signal should reach maximum amplitude. This is developed to avoid
    the mass of the shaker hitting the floor in the lower frequency range.
    Input
    ------
    fs:             float
                    sampling rate, usually an atribute of the class experiment
    
    f0:             float
                    Frequency (e.g. Hz) at time t=0.

    t1:             float
                    Time at which f1 is specified.

    f1:             float
                    Frequency (e.g. Hz) of the waveform at time t1.
                    
    amplitude:      float
                    Maximum value of peak Voltage desired          
                    
    record_length:  float
                    length of the signal in seconds, usually the specified
                    record length
                    
    pre_trigger:    float
                    time before initializing recording. Usually this parameter
                    is specify during setup.        
                    
    method:         string
                    {‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}, optional
                    Kind of frequency sweep. If not given, linear is assumed. 
                    
    kaiser_beta:    float
                    Kaiser window beta parameter: Shape parameter, determines 
                    trade-off between main-lobe width and side lobe level. 
                    As beta gets large, the window narrows               
    multiple_windows:bool
                    If True, the window of the signal of size window_time will 
                    be repeated until completion of the record length. This 
                    option is included to allow for continuous excitation with
                    intervals of no excitation. If False, the window of the 
                    signal will be equal to the record_length.
                    
    window_time:    float
                    The length of the signal in seconds that will be repeated 
                    until completion of record length.
    The input values of this function are to be used on scipy,signal.chirp function
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
    
    Output
    ------
    
    send_signal:            np.array
                            output signal in peak voltage
    
    """

    
    n_sec= record_length-pre_trigger
    t = np.linspace(0, n_sec, n_sec*fs)
    send_signal = amplitude*chirp(t, f0=f0, t1=n_sec, f1=f1,method=method)
    
    # find time where frequecny is lower that f_thres
    f,ts,Sxx = scisig.spectrogram(send_signal,fs) #using spectrogram
    
    #kaiser window in the change of frequency according to the threshold
    send_signal = Kaisser_window(send_signal,fs, kaisser_beta, t_window=t_window_spect,
                                 end=end)
    
    padding = record_length*fs-(len(send_signal)+int(fs*pre_trigger))
    output_signal_ = np.pad(send_signal,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
    output_signal = (amplitude/max(output_signal_))*output_signal_ 
    
    return output_signal
#%%
def create_particular_shaker_signal_(fs, f0, f1, amplitude, record_length, pre_trigger, 
                      method='linear'):
    
    """
    Special signal for the shaker to avoid high amplitudes in the 
    low frequency range.
    """              
                 
    
    n_sec= record_length-pre_trigger
    t = np.linspace(0, n_sec, n_sec*fs)
    send_signal = amplitude*chirp(t, f0=f0, t1=n_sec, f1=f1,method=method)
    
    # Linear vector voltage
    linear_vector = []
    all_vector=[]
    #limits
    # Ti=[0,10, 20, 30, 40, record_length-40, record_length-20,record_length-10]
    Ti=3*np.array([0,3.0,10,12,20,50,60, 65,68,70, record_length-40, record_length-20,record_length-10])    
    Vi=np.array([0,1.0,2.0,1.5,1.8,3.0,3.5,3.5,3.6,4.0, 5.0,2.0, 0.05])
    
    for i in range(1,len(Ti)):
        time_change_to_one = Ti[i]
        V = Vi[i]/max(Vi)
        Vi_1 = Vi[i-1]/max(Vi)
        t_magic = np.where(t<=time_change_to_one)[0][ -1]
        t_magic_1 = np.where(t<=Ti[i-1])[0][ -1]
       
        linear_vector = np.concatenate((linear_vector,np.linspace(Vi_1, V,len(send_signal[t_magic_1:t_magic]))))
        
    constant_vector = np.zeros(len(send_signal[t_magic:]))
    all_vector =np.concatenate((linear_vector,constant_vector))
    plt.plot(t,all_vector)
    
    send_signal_ = send_signal*all_vector 
    
    padding = record_length*fs-len(send_signal_)#+int(fs*pre_trigger))
    output_signal_ = np.pad(send_signal_,pad_width=(int(fs*pre_trigger),padding), mode='constant') 
    output_signal = (amplitude/max(output_signal_))*output_signal_ 

    return output_signal
#%%
def create_particular_shaker_signal(fs, f0, f1, amplitude, record_length, pre_trigger,method='linear'):

    """
    Special signal for the shaker to avoid high amplitudes in the
    low frequency range.
    """
    
    n_sec= record_length-pre_trigger
    t = np.linspace(0, n_sec, n_sec*fs)
    send_signal = amplitude*chirp(t, f0=f0, t1=n_sec, f1=f1,method=method)
    
    # Linear vector voltage
    linear_vector = []
    all_vector=[]
    #limits
    # Ti=[0,10, 20, 30, 40, record_length-40, record_length-20,record_length-10]
    Ti=[         0, 10, 78, 95,165,182,230,324,344,352,357,360,593,599,600]
    Vi=np.array([0,1.5,1.5,1.5,1.5,1.5,1.8,3.0,3.5,3.5,3.6,4.0,5.0,5.0,0])
    
    for i in range(1,len(Ti)):
        time_change_to_one = Ti[i]
        V = Vi[i]/max(Vi)
        Vi_1 = Vi[i-1]/max(Vi)
        t_magic = np.where(t<=time_change_to_one)[0][ -1]
        t_magic_1 = np.where(t<=Ti[i-1])[0][ -1]
    
        linear_vector = np.concatenate((linear_vector,np.linspace(Vi_1, V,len(send_signal[t_magic_1:t_magic]))))
    
    constant_vector = np.zeros(len(send_signal[t_magic:]))
    all_vector =np.concatenate((linear_vector,constant_vector))
    
    send_signal_ = send_signal*all_vector
    
    padding = record_length*fs-len(send_signal_)#+int(fs*pre_trigger))
    output_signal_ = np.pad(send_signal_,pad_width=(int(fs*pre_trigger),padding), mode='constant')
    output_signal = (amplitude/max(output_signal_))*output_signal_
    


    return output_signal

#%% align
def align(input_signal, output_signal, record_length, fs, t):
    """
    This function aligns two signals in the time domain by using the 
    cross-correlation values between the two signals. 
    
    Input
    ------
    
    input_signal:       np.array
                        Signal A
                        
    output_signal       np.array
                        Signal B
                        
    record_length:      float
                        length of the signal in seconds, usually the specified
                        record length
    
    fs:                 float
                        sampling rate, usually an atribute of the class experiment
                        
    t:                  np.array
                        Time vector  
                        
    Output
    ------
    record_tau:         np.array
                        Signal A shifted
                        
    signal_tau:         np.array
                        Signal B shifted
                        
    cross_val_norm:     np.array
                        Pearson-cross crorrelation value    
                        
    """
    
    
    cross_val = scisig.correlate(input_signal, output_signal)
    max_corr_idx = np.argmax(cross_val)
    
    lag_time = record_length-t[max_corr_idx]
    lag_points = int(lag_time*fs)
    
    record_tau = [0]*lag_points + list(input_signal)
    signal_tau = list(output_signal) + [0]*lag_points
    t_tau = np.linspace(0, record_length+lag_time, 
                    len(record_tau))
    
    cross_val_norm = np.corrcoef(record_tau,signal_tau)[0][1]
    return record_tau, signal_tau, t_tau, cross_val_norm
#%% show data
    
def show_data(experiment, sensors_to_plot='all', all_sen_one_fig =True, idx=0):
    
    """
    This function allows visualizing the data directly from the experiment class. 
    
    Input
    ------
    experiment :      Object (__main__.Experiment)
                      The experiment of this particular trial.
                    
    sensors_to_plot   string/list
                        The sensors that want to be plotted. The possible values
                        of this parameter are:
                            
                        "all": (String) All sensors from experiment.sensors will be included
                        in the plots
                        
                        [0,1,2,3]:  (list) A list of the sensors that will be included
                        in the plots. The numerical values represent the 
                        position of the sensor in the experiment.sensors list.
    
    all_sen_one_fig:   bool
                        If True, all sensors' data will be plotted in the same 
                        figure independent of the units. If False, the data 
                        is separated by units on different figures
    idx:              int
                      Index of the trials you want to plot
                        
    
    Output
    ------
    Figures containing the data collected for each sensor. The figures are 
    separated by sensor's units.
   
    
    """
    
    print(pd.DataFrame(experiment.sensors)[['channel', 'serial', 'units']])
    
    plt.close() # Close all open figures
    
    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    
    h5f = tb.open_file(data_file_name,mode='a') 
    data =  h5f.get_node('/experiment/data')[:][idx] #access the data table from file
    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    fs = experiment.fs
    t = np.linspace(0, experiment.record_length, int(experiment.record_length*fs))
    
    if all_sen_one_fig:
    
        if sensors_to_plot =='all':  #when all sensors' data is requiere
            # check if we have input and output signals to remove delay in time
            sensor_type_str = np.array([s['serial'].decode('UTF-8') for s in sensors])
            plt.figure()
            for sensor_i in sensors:
                if sensor_i['serial'].decode('UTF-8')  != 'SMI' and sensor_i['serial'].decode('UTF-8')  != 'SMO':
                    sensor_type = sensors[sensors['serial']==sensor_i['serial']] #select sensors based on the units
                    data_type = data[sensors['serial']==sensor_i['serial']][0] #select data from sensors with units in common
                    plt.plot(t, data_type, label = sensor_i['serial'].decode('UTF-8'))
                    
            if 'SMI' in sensor_type_str and 'SMO' in sensor_type_str:
                input_signal = data[np.where(sensor_type_str=='SMI')[0][0]]
                output_signal = data[np.where(sensor_type_str=='SMO')[0][0]]
                
                #using align function to remove the time shift
                record_tau, signal_tau, t_tau, cross_val_norm = align(input_signal, 
                                                                      output_signal, 
                                                                      experiment.record_length, 
                                                                      experiment.fs, t)
                
                plt.title(r'$\rho_{ij} = %.2f$'%cross_val_norm)
                plt.plot(t_tau, record_tau, label = 'SMI')
                plt.plot(t_tau, signal_tau, label = 'SMO')
                
            if not 'SMI' in sensor_type_str and 'SMO' in sensor_type_str:
                output_signal = data[np.where(sensor_type_str=='SMO')[0][0]]
                
                plt.plot(t,  output_signal, label = 'SMO')
            
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel('Amplitude')
            
        if sensors_to_plot !='all': #when specific sensors' data is requiere     
            sensor_plot = sensors[sensors_to_plot]
            data_plot = data[sensors_to_plot]            
            # check if we have input and output signals to remove delay in time
            sensor_type_str = np.array([s['serial'].decode('UTF-8') for s in sensor_plot])            
            plt.figure()
            for sensor_i in sensor_plot:
                if sensor_i['serial'].decode('UTF-8')  != 'SMI' and sensor_i['serial'].decode('UTF-8')  != 'SMO':
                    sensor_type = sensor_plot[sensor_plot['serial']==sensor_i['serial']] #select sensors based on the units
                    data_type = data_plot[sensor_plot['serial']==sensor_i['serial']][0] #select data from sensors with units in common
                    plt.plot(t, data_type, label = sensor_i['serial'].decode('UTF-8'))
                #Some element in the list may not inlcude both input an output shaker module signals 
                #the following will allow still show one of those without shifting in time
                elif sensor_i['serial'].decode('UTF-8')  == 'SMI' or sensor_i['serial'].decode('UTF-8')  == 'SMO':
                    sensor_type = sensor_plot[sensor_plot['serial']==sensor_i['serial']] #select sensors based on the units
                    data_type = data_plot[sensor_plot['serial']==sensor_i['serial']][0] #select data from sensors with units in common
                    plt.plot(t, data_type, label = sensor_i['serial'].decode('UTF-8'))
                    
            if 'SMI' in sensor_type_str and 'SMO' in sensor_type_str:
                
                input_signal = data_plot[np.where(sensor_type_str=='SMI')[0][0]]
                output_signal = data_plot[np.where(sensor_type_str=='SMO')[0][0]]
                
                #using align function to remove the time shift
                record_tau, signal_tau, t_tau, cross_val_norm = align(input_signal, 
                                                                      output_signal, 
                                                                      experiment.record_length, 
                                                                      experiment.fs, t)
                
                plt.title(r'$\rho_{ij} = %.2f$'%cross_val_norm)
                plt.plot(t_tau, record_tau, label = 'SMI')
                plt.plot(t_tau, signal_tau, label = 'SMO')
                        
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel('Amplitude')         
    else:
        if sensors_to_plot =='all':  #when all sensors' data is requiere
            unit_types = np.unique(sensors['units'])  #extract the unique units of all sensors 
            for unit_type_i in unit_types:
                sensor_type = sensors[sensors['units']==unit_type_i] #select sensors based on the units
                data_type = data[sensors['units']==unit_type_i] #select data from sensors with units in common
                
                # check if we have input and output signals to remove delay in time
                sensor_type_str = np.array([s['serial'].decode('UTF-8') for s in sensor_type])
                if  list(sensor_type_str) ==['SMI', 'SMO'] or  list(sensor_type_str) ==['SMO', 'SMI']:
                    input_signal = data_type[np.where(sensor_type_str=='SMI')[0][0]]
                    output_signal = data_type[np.where(sensor_type_str=='SMO')[0][0]]
                    
                    #using align function to remove the time shift
                    record_tau, signal_tau, t_tau, cross_val_norm = align(input_signal, 
                                                                          output_signal, 
                                                                          experiment.record_length, 
                                                                          experiment.fs, t)
                    plt.figure()
                    plt.title(r'$\rho_{ij} = %.2f$'%cross_val_norm)
                    plt.plot(t_tau, record_tau, label = 'SMI')
                    plt.plot(t_tau, signal_tau, label = 'SMO')
                    plt.legend()
                    plt.xlabel('time (s)')
                    plt.ylabel('%s'%unit_type_i.decode('UTF-8'))
                else:
                    plt.figure()
                    for sensor_i in range(len(sensor_type)):
                        plt.plot(t, data_type[sensor_i], label = sensor_type[sensor_i]['serial'].decode('UTF-8'))
                    plt.legend()
                    plt.xlabel('time (s)')
                    plt.ylabel('%s'%unit_type_i.decode('UTF-8'))
                
        if sensors_to_plot !='all': #when specific sensors' data is requiere     
            unit_types = np.unique(sensors['units'])   
            sensor_plot = sensors[sensors_to_plot]
            data_plot = data[sensors_to_plot]
            for unit_type_i in unit_types:
                sensor_type = sensor_plot[sensor_plot['units']==unit_type_i]
                data_type = data_plot[sensor_plot['units']==unit_type_i]
                
                # check if we have input and output signals to remove delay in time
                sensor_type_str = np.array([s['serial'].decode('UTF-8') for s in sensor_type])
                if  list(sensor_type_str) ==['SMI', 'SMO'] or  list(sensor_type_str) ==['SMO', 'SMI']:
                    input_signal = data_type[np.where(sensor_type_str=='SMI')[0][0]]
                    output_signal = data_type[np.where(sensor_type_str=='SMO')[0][0]]
    
                    #using align function to remove the time shift
                    record_tau, signal_tau, t_tau, cross_val_norm = align(input_signal, 
                                                                          output_signal, 
                                                                          experiment.record_length, 
                                                                          experiment.fs, t)
                    plt.figure()
                    plt.plot(t_tau, record_tau, label = 'SMI')
                    plt.plot(t_tau, signal_tau, label = 'SMO')
                    plt.legend()
                    plt.xlabel('time (s)')
                    plt.ylabel('%s'%unit_type_i.decode('UTF-8'))
                    
                else:   
                    plt.figure()
                    for sensor_i in range(len(sensor_type)):
                        plt.plot(t, data_type[sensor_i], label = sensor_type[sensor_i]['serial'].decode('UTF-8'))
                    plt.legend()
                    plt.xlabel('time (s)')
                    plt.ylabel('%s'%unit_type_i.decode('UTF-8'))
            
#%% check psd
                    
def check_psd(experiment, sensors):   
    """
    Plot the power spectral density.
    
    Input
    ------
    experiment :        Object (__main__.Experiment)
                        The experiment of this particular trial.
                  
    sensors:            list
                        [0,1,2,3]:  (list) A list of the sensors that will be included
                        in the plots. The numerical values represent the 
                        position of the sensor in the experiment.sensors list.
    Output
    ------
    Figure with PSD vs Frequency 
    
    
    """
    print(pd.DataFrame(experiment.sensors)[['channel', 'serial', 'units']])
    
    plt.close() # Close all open figures
    
    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    
    h5f = tb.open_file(data_file_name,mode='r') 
    data =  h5f.get_node('/experiment/data')[:][0] #access the data table from file
#    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    fs = experiment.fs
#    t = np.linspace(0, experiment.record_length, experiment.record_length*fs)
    plt.figure()
    for sensor_i in sensors:              
        plt.title(r'PSD')
        plt.psd(data[sensor_i], Fs=fs, label=experiment.sensors[sensor_i]['serial'])
        
    plt.legend()
#%% create filter
def create_filter(oldfs, trans_hz=10, ripple_db=160, cutoff_hz=256):
    
    from scipy.signal import firwin, kaiserord
    # nyquist rate of signal
    nyq_rate = oldfs / 2.0

    # transition width from pass to stop relative to nyquist rate
    width = trans_hz / nyq_rate

    # determine order (number of taps) and kasier beta parameter for FIR filter
    N, beta = kaiserord(ripple_db, width)

    # generate a lowpass fir filter
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    # denominator is always equal to 1 for kaiser windows
    a = 1.0

    return taps, a

def filter_signal(data, taps, a):
    
    from scipy.signal import lfilter
    # phase delay
    delay = 0.5 * (len(taps) - 1)

    # filtered data
    datafil = lfilter(taps, a, data)

    return datafil, delay
    
#%% resample
def resample(data, oldfs, newfs, trans_hz=10, ripple_db=160, cutoff_hz=256):
    """
    Ref
    ----------
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.resample.html    
    """
    from scipy import signal
    from numpy import ceil
    
    # filtering using FIR (Finite impulse response)
    taps, a = create_filter(oldfs, trans_hz, ripple_db, cutoff_hz)
       
    fdata, delay = filter_signal(data, taps, a)

    # resampling
    rdata = signal.resample(x=fdata,
                            num=int(ceil(len(fdata) * (round(newfs) / round(oldfs)))))

    # adjusting for phase delay and returning the good data
    delay = round(delay * (newfs / oldfs))

    return rdata[delay:]

#%% tfes

def tfes(experiment, x, y, method, newfs=None, noverlap=None, filter_TF=False): 
    
    from scipy.signal import savgol_filter

    if newfs:
        x_resampled = resample(x, oldfs=experiment.fs,
                              newfs =newfs, 
                              trans_hz=10, 
                              ripple_db=160, 
                              cutoff_hz=256)
       
        x =x_resampled        
        y_resampled = resample(y, oldfs=experiment.fs,
                              newfs =newfs, 
                              trans_hz=10, 
                              ripple_db=160, 
                              cutoff_hz=256)
       
        y =y_resampled
        
    window = np.hamming(len(x))
    if noverlap:
        noverlap = int(noverlap*len(x)/100)

    fft_n = len(x)
    
    # Cross spectral density calculations
    Pxx, freq = mlab.psd(x=x, NFFT=fft_n,Fs=experiment.fs, 
                         window=window, noverlap=noverlap)
    Pxy, _ = mlab.csd(x=x, y=y, NFFT=fft_n,Fs=experiment.fs, 
                      window=window, noverlap=noverlap)
    
    Pyy, _ = mlab.psd(x=y, NFFT=fft_n, Fs=experiment.fs,
                      window=window, noverlap=noverlap)
    Pyx, _ = mlab.csd(x=y, y=x, NFFT=fft_n, Fs=experiment.fs, 
                      window=window, noverlap=noverlap)
    
    if method ==1:
        # Method 1
        tfe = Pxy/Pxx
        
    elif method ==2:
        # Method 2
        tfe = Pyy/Pyx
        
    elif method ==3:
        # Method 3
        tf1 = Pxy/Pxx
        tf2 = Pyy/Pyx
        tfe = np.average(np.vstack((tf1,tf2)), axis = 0)
        
    elif method ==4:
        #Method 4
        x_fft = np.fft.fft(x, n=experiment.fft_n, axis=0)
        y_fft = np.fft.fft(y,n=experiment.fft_n, axis=0)
        freq = np.fft.fftfreq(experiment.fft_n, 1/experiment.fs)
        tfe = (y_fft/x_fft)[0:int((experiment.fft_n/2) + 1)]
    
    elif method ==5:
        # Method 5
        tfe = (tf1*tf2) ** 0.5 
    
    if filter_TF:
      tfe = np.exp(savgol_filter(np.log(tfe), 51, 8))
    
    return tfe, freq

def show_tfs(experiment, window_sec=20, shaker_sensor_idx=0, noverlap=None,
             method=3, newfs=None, plot_per_sensor=True, trial_index = -1, 
             limits = [0,200], filter_TF = False):
    
    import matplotlib
    import matplotlib.pylab as plt
    
    font = {'family' : 'Times New Roman',
              'size'   : 15}
      
    matplotlib.rc('font', **font)
    
    print(pd.DataFrame(experiment.sensors)[['channel', 'serial', 'units']])

    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    
    h5f = tb.open_file(data_file_name,mode='a') 
    data =  h5f.get_node('/experiment/data')[:][ trial_index] #access the data table from file
    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    h5f.close()
    
    sensors_info = pd.DataFrame(sensors)
    sensors_accel = sensors_info[sensors_info['sensor_type']==b'Accelerometer']
    accel_idxs = list(sensors_accel.index);
    accel_idxs = np.where(np.array(accel_idxs) != shaker_sensor_idx)[0]
    force = data[shaker_sensor_idx]
    
    split_number = int(experiment.record_length/window_sec)
    
    TF_general=[]
    for accel_idx in accel_idxs:
        accel = data[accel_idx]
        
        accel_split = np.split(accel, split_number)
        force_split =  np.split(force, split_number)
        tfi_window = []
        for window_i in range(len(accel_split)):
            
            accel_window = accel_split[window_i] 
            force_window = force_split[window_i]
            
            tf_i, freq = tfes(experiment, force_window, accel_window, method=3,
                              newfs=newfs, noverlap=noverlap, filter_TF=filter_TF)
            tfi_window.append(tf_i)
        tf_average_all_windows = np.mean(np.array(tfi_window, dtype=complex), axis=0)
        if plot_per_sensor:
            plt.figure()
            plt.plot(freq, np.real(20*np.log10(tf_average_all_windows)),
                      label='%s'%sensors_info['serial'][accel_idx].decode('UTF-8'))
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('TF [dB]')
            plt.xlim(limits)
            #plt.title('TF average of %s windows per sensor'%split_number)

        TF_general.append(tf_average_all_windows)

    if not plot_per_sensor:
        tfi_average = np.mean(TF_general, axis=0)
        plt.plot(freq, np.real(20*np.log10(tfi_average)),label='TF [dB]')
        plt.xlim(limits)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log(TF)')
        plt.title('TF average of %s windows between all sensors'%split_number)

#%% show_tfs_imag
def show_tfs_imag(experiment, window_sec=20, shaker_sensor_idx=0, noverlap=None,
             method=3, newfs=None, plot_per_sensor=True, trial_index = -1, 
             limits = [0,200]):
    
    import matplotlib
    font = {'family' : 'Times New Roman',
              'size'   : 15}
      
    matplotlib.rc('font', **font)
    
    print(pd.DataFrame(experiment.sensors)[['channel', 'serial', 'units']])

    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    
    h5f = tb.open_file(data_file_name,mode='a') 
    data =  h5f.get_node('/experiment/data')[:][ trial_index] #access the data table from file
    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    h5f.close()
    
    sensors_info = pd.DataFrame(sensors)
    sensors_accel = sensors_info[sensors_info['sensor_type']==b'Accelerometer']
    accel_idxs = list(sensors_accel.index);
    force = data[shaker_sensor_idx]
    
    split_number = int(experiment.record_length/window_sec)
    
    TF_general=[]
    for accel_idx in accel_idxs:
        accel = data[accel_idx]
        
        accel_split = np.split(accel, split_number)
        force_split =  np.split(force, split_number)
        tfi_window = []
        for window_i in range(len(accel_split)):
            
            accel_window = accel_split[window_i] 
            force_window = force_split[window_i]
            
            tf_i, freq = tfes(experiment, force_window, accel_window, method=3,
                              newfs=newfs, noverlap=noverlap)
            tfi_window.append(tf_i)
        tf_average_all_windows = np.mean(np.array(tfi_window, dtype=complex), axis=0)
        if plot_per_sensor:
            plt.figure()
            plt.plot(freq, np.imag(tf_average_all_windows),
                      label='%s'%sensors_info['serial'][accel_idx].decode('UTF-8'))
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Imag')
            plt.xlim(limits)
            #plt.title('TF average of %s windows per sensor'%split_number)

        TF_general.append(tf_average_all_windows)

    if not plot_per_sensor:
        tfi_average = np.mean(TF_general, axis=0)
        plt.plot(freq, np.imag(tfi_average),label='TF [dB]')
        plt.xlim(limits)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Imag')
        plt.title('TF average of %s windows between all sensors'%split_number)
#%% coherence
def coherence(experiment, window_sec=20, shaker_sensor_idx=1, plot_per_sensor=True,
              idx = 0):
    from scipy import signal
    
    print(pd.DataFrame(experiment.sensors)[['channel', 'serial', 'units']])

    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    
    h5f = tb.open_file(data_file_name,mode='r+') 
    data_ =  h5f.get_node('/experiment/data')[:][idx] #access the data table from file
    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    
    # initial_cut = int(experiment.pre_trigger*experiment.fs*1.5)
    # data = np.array([d.tolist()[initial_cut:len(d)-initial_cut] for d in data_])
    data = data_
    
    sensors_info = pd.DataFrame(sensors)
    sensors_accel = sensors_info[sensors_info['sensor_type']==b'Accelerometer']
    accel_idxs = list(sensors_accel.index);
    #accel_idxs.remove(shaker_sensor_idx);
    force = data[shaker_sensor_idx]
    
    split_number = int(experiment.record_length/window_sec)
    
    Cxy_general=[]
    for accel_idx in accel_idxs:
        accel = data[accel_idx]
        
        accel_split = np.split(accel, split_number)
        force_split =  np.split(force, split_number)
        Cxy_window = []
        for window_i in range(len(accel_split)):
            
            accel_window = accel_split[window_i] 
            force_window = force_split[window_i]
            
            # tf_i, freq = tfes(experiment, force_window, accel_window, method=3)
            # tfi_window.append(tf_i)
            freq, Cxy = signal.coherence(force_window, accel_window, experiment.fs)
            Cxy_window.append(Cxy)
            
        Cxy_average_all_windows = np.mean(np.array(Cxy_window), axis=0)
        if plot_per_sensor:
            plt.semilogy(freq, Cxy_average_all_windows,
                      label='%s'%sensors_info['serial'][accel_idx].decode('UTF-8'))
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Log(TF)')
            plt.title('Coherence average of %s windows per sensor'%split_number)
            plt.xlim([0,300])
        Cxy_general.append(Cxy_average_all_windows)

    if not plot_per_sensor:
        Cxy_average = np.mean(Cxy_general, axis=0)
        plt.semilogy(freq, np.log(Cxy_average),label='TF')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log(TF)')
        plt.title('Coherence average of %s windows between all sensors'%split_number)
        plt.xlim([0,50])
#%% show_tfs_average
def show_tfs_average(experiment, window_sec=20, shaker_sensor_idx=0, noverlap=None,
             method=3, newfs=None, plot_per_sensor=True):
    
    import matplotlib
    import numpy as np
    import os
    
    font = {'family' : 'Times New Roman',
              'size'   : 15}
      
    matplotlib.rc('font', **font)
    
    #sensors info
    data_file_name = experiment.fname     # load the hdf5 file of the experiment  
    h5f = tb.open_file(data_file_name,mode='a') 
    sensors = h5f.get_node('/experiment/sensors')[:] # access the sensors table from file
    h5f.close()
    
    sensors_info = pd.DataFrame(sensors)
    sensors_accel = sensors_info[sensors_info['sensor_type']==b'Accelerometer']
    accel_idxs = list(sensors_accel.index);
    
    #getting specific locations from trials   
    locations = []
    for trial_i in experiment.trials:
        location_i = trial_i.get_specific_parameter("Node-ID")
        locations.append(location_i)
    
    unique_loc = np.unique(locations)
    #empty dictionary
    loc_dic = {}
    for loc in unique_loc:
        loc_dic["%s"%loc] = []
        
    split_number = int(experiment.record_length/window_sec)
    
       
    #get tfes and append per location
    for trial_i in experiment.trials:
        data_i = experiment.trials[0].get_data()
        force = data_i[:,shaker_sensor_idx]
        
        TF_general=[]
        for accel_idx in accel_idxs:
            accel = data_i[:,accel_idx]
            
            accel_split = np.split(accel, split_number)
            force_split =  np.split(force, split_number)
            tfi_window = []
            for window_i in range(len(accel_split)):
                
                accel_window = accel_split[window_i] 
                force_window = force_split[window_i]
                
                tf_i, freq = tfes(experiment, force_window, accel_window, method=3,
                                  newfs=newfs, noverlap=noverlap)
                tfi_window.append(tf_i)
            tf_average_all_windows = np.mean(np.array(tfi_window, dtype=complex), axis=0)
       
            TF_general.append(tf_average_all_windows)
        
        loc_trial_i = trial_i.get_specific_parameter("Node-ID")
        
        loc_dic["%s"%loc_trial_i].append(np.array(TF_general))
    
    loc_dic_final ={}
    for loc in unique_loc:
        loc_dic_final["%s"%loc] = np.average(np.array(loc_dic["%s"%loc]), axis=0)
    
    if plot_per_sensor:
        path_new = os.path.join(experiment.fname.split(".")[0],"tfes_figures")
        
        if not os.path.exists(path_new):
            os.makedirs(path_new)
        
        for loc in unique_loc:
            plt.figure("Impact location Node-%s"%loc)
            plt.title("Impact location Node-%s"%loc)
            for sensor_i in accel_idxs:
                plt.plot(freq, np.real(20*np.log10(loc_dic_final["%s"%loc][sensor_i-1])),
                          label='%s'%sensors_info['serial'][sensor_i].decode('UTF-8'))
                plt.legend()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('$\\log(TF)$ [dB]')
                plt.xlim([0,200])
                plt.grid(True)
            plt.savefig(os.path.join(path_new, "loc -%s.pdf"%loc))
            plt.close()