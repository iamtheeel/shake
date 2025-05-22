"""
Collects all data pertaining to the experiment and will eventually store it in 
an HDF5 file.
"""

import os
import tables
import numpy as np
import matplotlib.pyplot as plt
import nidaqpy.recorder as rec
from nidaqpy.data_utils import buffer
import nidaqpy.data_utils as dt
import time
import datetime


class Trial():
    """
    Creates a trial for the data collection
    
    Attributes
    ----------
    experiment : Experiment
        Experiment of this particular trial.
    ident : int
        ID of this particular trial.  Matches with hdf5 file
        
    Methods
    -------
    get_data()
        Reads the data from the hdf5 file
    plot()
        Plots the experimental data
        
    """
    def __init__(self, experiment, ident = -1):
        self.experiment = experiment
        self.id = ident

    def peak_detector(self, max_level = 0.0005, sensor = 0):
        """
        Identifies if the trial contains a single peak
        
        Parameters:
        -----------
        max_level : float
            Level that triggers the peak_detector
            
        sensor : int
            Sensor to use for the peak detection
            
        Returns:
        --------
        
        flag : Float indicating if a peak is detected
        """
        record_acceleration = self.get_data()[sensor]
        accel_arr = np.abs(np.array(record_acceleration)) 
        accel_index = np.where(accel_arr > max_level)
        number_above = np.shape(accel_index)[1]
        if number_above >=1 and number_above <= 10:
            return True
        else:
            return False
        
    def get_specific_parameter(self,parameter = 'Date'):
        """
        Reads and retrieves the specific parameter stored on the hdf5 file for a specific trial.

        Definitions:
        ----------
        specific parameter :  different for each iteration
        general parameter :   same for all trials of the experiment


        Parameters:
        ----------
        Parameter : string
            a specific parameter that may be retrieved by this method (default = Date)
            

        Returns:
        -------
        value : a specific parameter retrieved for that trial

        """

        table= self.experiment.__efile__.get_node('/experiment/specific_parameters')
        return next(record['value'] for record in table.where("(parameter == '%s') & (id == %i)"%(parameter, self.id)))

        
    def get_data(self):
        '''
        Reads data from the hdf5 file for the appropriate trial
        
        Returns:
        -------
        dataT : Data for the trial
        '''

        # Get the record
        data = self.experiment.__efile__.get_node('/experiment/data')
        dataT = data.read(self.id)[0].transpose()

        return dataT
        
    
    def plot(self):
        """
        Generates a singular plot of the amplitude of N number of sensors (i.e. 1,2,3,...N) 
        with respect to time. Each sensors data is a different color 
        and this is denoted in the plots legend.
        
        Parameters:
        ------------
        None.

        Returns:
        -------
        None.

        """
        data = self.get_data()
        t = np.arange(0,np.shape(data)[0]) / self.experiment.fs
        
        legends = []
        for sensor, counter in zip(self.experiment.sensors,range(len(self.experiment.sensors)-self.experiment.__ncam__)): #TODO: same, stop looking into cameras
            legends.append(sensor['serial'])
        plot = plt.plot(t,data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(legends)
        
        return plot

    def video_file_name(self):
        r""" 
            Returns the complete video filename based on the specific parameter
        
            Parameters:
            -----------

            
            Returns:.
            ---------
            fname (str): Full address to video file name for the current trial
            
          
        """  
        import os
        
        
        date=self.get_specific_parameter() 
        fname=os.getcwd()+'\\data\\'+ self.experiment.title + '_video\\' + date[0:10] + '\\' + date + '.avi' #date[0:10]=folder date
        
        return fname
    

class Experiment():
    """
    Collects and stores the parameters of the experiment and runs a DAQ system. 
    
    This class allows the user to setup a National Instruments DAQ system
    via a serie of questions.  This class was originally developed for
    experiments dealing with accelerometers.  Other type of sensors might need
    modifications to the code.  The class generates an HDF5 file with the
    appropriate metadata.
    
    Attributes
    ----------
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
    __data_folder__ : str
        String containing the folder where data will be storerd
    
        
    Methods
    -------
    collect_data(fname='data.hdf5')
        Collects data and saves the data and the metadata in the file 'fname'
    load_setup(fname='setup.json')
        Load configuration of experiment stored in file 'fname'
    save_setup(fname='setup.json')
        Saves the setup of the experiment and associated metadata in the file
        'fname'.
    setup()
        Asks the user a serie of questions to setup the DAQ
    
    
    """
    def __init__(self,title = '', fs = 1652, record_length = 30, pre_trigger = 10, sensors = [], parameters = {}, voltage_check_module=False): #This voltage check should be inside the setup JSON file
        # Parameters
        self.fs = fs  # sampling frequency in Hz
        self.record_length = record_length  # Record length in seconds
        self.pre_trigger = pre_trigger      # Pre-trigger length in seconds
        self.sensors = sensors
        self.parameters = parameters
        self.title = title
        self.__b__ = []                     # data buffer
        self.__vb__ = []                    # video buffer
        self.__vision__ = False             # Cameras availability
        self.__ncam__=0                     # Number of cameras (auto detected from JSON)
        self.trials = []
        self.fname = ''
        self.voltage_check_module =voltage_check_module
        self.__efile__ = ''
        
    def continuous_recording(self,t_rec,web_n=False,filter_functions=[],end_time=[]):
        """
        Sets unatended acquisition until target records (t_rec) is collected
        Webhook url is needed for online updates to slack
        web_n boolean to decide web message posting, by default is False
        end_time= list with 3 values [Days,Hour,Minute] (24hr format)
                    Days: amount of days to collect
                    Hour, Minute: Time of ending the data collection after [Days] amount
        
        TODO: Complete help
        """
        
        import requests
        import json
        webhook_url = 'https://hooks.slack.com/services/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'
        
        Emessage="------------ DATA ACQ STARTED --------------"    
        data = {'text': Emessage,'username': 'HAL','icon_emoji': ':robot_face:'}
        if web_n:
            response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        if end_time==[]:
            timed_adq=False
        else:
            target = datetime.datetime.combine(datetime.date.today(), datetime.time(hour=end_time[1],minute=end_time[2]))
            target = target + datetime.timedelta(days = end_time[0]) #Adds the amount of days on end_time[0]
            timed_adq=True

        for i in range(t_rec):
            try:
                self.collect_data(filter_functions=filter_functions)
                
                now = datetime.datetime.now()
                
                if timed_adq:
                    if target<now:
                        
                        print("Finished continued acquisition due to timeframe")
                        break
                                        
            except:
              
                data = {'text': '----------SOMETHING HAPPENED----------','username': 'HAL','icon_emoji': ':robot_face:'}
                if web_n:
                    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
             
                print(data['text'])
                self.__task__.stop()
                self.__task__.close()
                pass
        
            message="--------- Finished Acquisition and Restarting, Record #"+str(i+1)+" out of "+str(t_rec)+"---------"
            print(message)
            
            data = {'text': message,'username': 'HAL','icon_emoji': ':robot_face:'}
            if web_n:
                response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

        Emessage="------------ All done! --------------"    
        print(Emessage)

        data = {
                'text': Emessage,
                'username': 'SDII DAQ',
                'icon_emoji': ':robot_face:'
                }
        if web_n:
            response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

        

    def setup_daq(self):
        """
        Setup NI DAQ system and returns the task
        TODO: Complete help
        """
        # Importing packages
        import nidaqmx
        
        if self.__vision__ == True:
            cid=0
            import cv2
            for sensor in self.sensors:
                if sensor['sensor_type'] == "Camera": #search for cameras and creates and setup each VideoCapture object
                    print("Setting up camera "+str(cid))
                    exec(f'cap{cid} = cv2.VideoCapture(cid,cv2.CAP_DSHOW)') #dshow mode to access full sensor
                    exec(f'cap{cid}.set(cv2.CAP_PROP_AUTOFOCUS, 0)') #disable autofocus
                    exec(f'cap{cid}.set(3, sensor["resolution"][1])') #Set h_resolution
                    exec(f'cap{cid}.set(4, sensor["resolution"][2])') #Set v_resolution
                    exec(f'self.__cap{cid}__=cap{cid}') #Set video capture global self.__cap1__ =cap1
                    print("Camera setup completed")
                    cid=cid+1
                    
                # font 

        
        # Create the task
        task = nidaqmx.Task()
        task_o = False
        #% Create the channels
        for sensor in self.sensors:
            if sensor['sensor_type'] == "Accelerometer":
                task.ai_channels.add_ai_accel_chan(physical_channel = sensor['channel'], sensitivity = sensor['sensitivity'], min_val = sensor['min_val'], max_val = sensor['max_val'])
            elif sensor['sensor_type'] == "Hammer" or sensor['sensor_type'] == "Force":
                task.ai_channels.add_ai_force_iepe_chan(physical_channel = sensor['channel'], sensitivity = sensor['sensitivity'], min_val = sensor['min_val'], max_val = sensor['max_val'])
            elif sensor['sensor_type'] == "Microphone" or sensor['sensor_type'] == "Sound":
                task.ai_channels.add_ai_force_iepe_chan(physical_channel = sensor['channel'], sensitivity = sensor['sensitivity'], min_val = sensor['min_val'], max_val = sensor['max_val'])
            elif sensor['sensor_type'] == "Module-input":
                task.ai_channels.add_ai_voltage_chan(physical_channel = sensor['channel'], min_val = sensor['min_val'], max_val = sensor['max_val'])
            elif sensor['sensor_type'] == "Module-output":
                task_o = nidaqmx.Task()
                task_o.ao_channels.add_ao_voltage_chan(physical_channel = sensor['channel'],  min_val = sensor['min_val'], max_val = sensor['max_val'])
                task_o.timing.cfg_samp_clk_timing(self.fs,sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=int(self.fs/10))
                
                #Writing samples of the ouput task
                #---------------------------------------------------------------
                # osignal = dt.create_sine(amplitude_peak_voltage=2,
                                        # sine_freq_npi_hz=10,
                                        # signal_width=self.record_length-(self.pre_trigger*2), 
                                        # fs=self.fs, 
                                        # pre_trigger=self.pre_trigger, kaisser_beta=28,
                                        # t_window=30, end="both")
                #---------------------------------------------------------------

                # osignal = dt.create_band_limited_noise(min_freq=0, max_freq=200,
                #                                         amplitude=2,
                #                                         record_length=self.record_length,
                #                                         pre_trigger=self.pre_trigger,
                #                                         fs=self.fs, kaisser_beta=10,
                #                                         t_window = 90,
                #                                         multiple_windows=False,
                #                                         window_time=20)                        
                #---------------------------------------------------------------
                # osignal = dt.create_sine_sweep(fs=self.fs, f0=1, f1=50, 
                #                               amplitude=1, 
                #                               record_length=self.record_length,
                #                               pre_trigger=self.pre_trigger,
                #                               method='linear', kaisser_beta=28,
                #                               t_window=30,
                #                               multiple_windows=False,
                #                               window_time=30)   
                #---------------------------------------------------------------
                # osignal = dt.create_special_sine_sweep(fs= self.fs, f0 =200, 
                #                                         f1=0.01, amplitude=5, 
                #                                         record_length=self.record_length,
                #                                         pre_trigger=self.pre_trigger,
                #                                         method="logarithmic", 
                #                                         kaisser_beta=30, 
                #                                         t_window_spect =50,
                #                                         end=None) #None means not symmetric kaisers [w, w/2]

                osignal = dt.create_particular_shaker_signal(fs=self.fs, f0=1,
                                                             f1 =200, amplitude =4.8,
                                                             record_length=self.record_length, 
                                                             pre_trigger=self.pre_trigger,
                                                             method="logarithmic")
                
                test_Writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(task_o.out_stream, auto_start=True)    
                test_Writer.write_many_sample(osignal, timeout=nidaqmx.constants.WAIT_INFINITELY)#nidaqmx.constants.WAIT_INFINITELY) #start output signal
                self.output_signal =osignal #saving the signal as an atribute of the experiment
            
        # Setting sampling rate and samples per channel
        print(f"Init clock: {self.fs}Hz")
        task.timing.cfg_samp_clk_timing(rate=self.fs,sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=int(self.fs/10))
        print(f"Actual rate: {task.timing.samp_clk_rate}")
        self.fs = task.timing.samp_clk_rate #MJB: Set what we want, then ask what we got


        self.__task__ = task
        self.__task_o__= task_o
        
        
    def get_frame(self):
        """
        Reads frames from 2 cameras, writes timestamps, append both frames
        TODO: Complete help and make it work for __ncam__ without sacrificing performance
        """
        import cv2
        frame = np.empty([int(self.__cap0__.get(4)*2),int(self.__cap0__.get(3)),3],dtype=np.uint8)
        ret, frame[0:int(self.__cap0__.get(4)),:,:] = self.__cap1__.read()
        ret, frame[int(self.__cap0__.get(4)):,:,:] = self.__cap0__.read() 
        frame = cv2.putText(frame, str(datetime.datetime.now()), (100,100), cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0), 3, cv2.LINE_AA)
        """
        "str(datetime.datetime.now()), (100,100)", overlays the current date and time (obtained from the datetime module) 
        at the position (100, 100) on the frame), cv2.FONT_HERSHEY_SIMPLEX (Font type),2(Font Scale) ,
        (255, 0, 0)Specifies the text color in BGR format (blue)), 3(thickness of the text), 
        cv2.LINE_AA(Antialiasing type,used in computer graphics and typography to reduce the visual distortion 
        or "jagged" appearance that can occur when rendering images, text, or other graphical elements at 
        lower resolutions or when they are displayed at angles)
        """

                
        return frame
    
    def check_folder(self,mydir):
        import os
        # If folder doesn't exist, then create it.
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
            print("Created folder : ", mydir)
    
    def save_video(self,name):
        """
        common
    
        """

        import numpy as np
        import cv2
        
        
        
        mydir = 'data' + '/' + self.title + "_Video"+ '/' + name[:10]
    
        # If folder doesn't exist, then create it.
        self.check_folder(mydir)

   
    # fname= mydir +'/'+ self.title+'.hdf5'

        for sensor in self.sensors:
            if sensor['sensor_type'] == "Camera":
                vr=int(sensor['resolution'][1])
                hr=int(sensor['resolution'][2])
                
        
    # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(mydir + '/' + name + '.avi',fourcc, 10, (hr*2,vr))
        tmp2=self.__vb__.read()
        
        
        print('rotating and saving frames...')
        for i in range(len(tmp2)):
            
            if np.isscalar(tmp2[i]): #case of having 0 in the list (buffer not filled)
                frame=np.zeros((hr*2,vr,3)) #case of not having ndarray in the list
            else:
                frame=tmp2[i]
            
            frame1=cv2.rotate(frame[:hr,:],cv2.ROTATE_90_CLOCKWISE)
            frame2=cv2.rotate(frame[hr:,:],cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame=np.append(frame2,frame1,axis=1)
            if i==int(self.pre_trigger*10): #Trigger location image retrieve
                thumb=cv2.resize(frame,None,fx=0.3,fy=0.3)
                cv2.imwrite(mydir + '/' + name + '.jpg', thumb)
            
            out.write(frame)
            # cv2.imshow('frame',frames[:,:,:,i])
            i=i+1    
        out.release()
        cv2.destroyAllWindows()
#%%
            
    def collect_data(self, filter_functions=[]):
        """
        Collects data
        
        Parameters
        ----------
        fname : str
            File that the data collected will be saved in (HDF5 file)
        
        """
        # Check if directory exists, if not, create it

        mydir = "data"
    
        # If folder doesn't exist, then create it.
        self.check_folder(mydir)
   
        fname= mydir +'/'+ self.title+'.hdf5'
        
        #%% Constants
        nsensors = len(self.sensors)-self.__ncam__ #TODO: find a way to not count the cameras


        #%% Closing the hdf5 as read-only file
        if self.__efile__ != '':
            self.__efile__.close()
        
        #%% Opening hdf5 file in append mode
        efile = tables.open_file(fname, mode="a", title=self.title)
        
        # If group exists, opens existing group and adds to tables
        if "/experiment" not in efile:
            exp = efile.create_group("/", 'experiment', 'Experimental data')
        else:
            exp = efile.get_node('/experiment')
            
        if "/experiment/general_parameters" in efile:
            # opens table for experiment class (general parameters)
            gpar_table = efile.get_node('/experiment/general_parameters')
            gpar = gpar_table.row
        else:
            # creates table for experiment class (general parameters)
            gpar_table = efile.create_table(exp, 'general_parameters', rec.TrialTable, "General Parameters Table")
            gpar = gpar_table.row
            gpar['parameter'] = 'fs'
            gpar['value'] = self.fs
            gpar['units'] = 'Hz'
            gpar.append()
            
            gpar['parameter'] = 'record_length'
            gpar['value'] = self.record_length
            gpar['units'] = 's'
            gpar.append()
            
            gpar['parameter'] = 'pre_trigger'
            gpar['value'] = self.pre_trigger
            gpar['units'] = 's'
            gpar.append()
            
            #%% Instead of adding parameters when not existing, the code should check if the parameters exist already in the table
            for parameter in self.parameters['general']:
                gpar['parameter'] = parameter['parameter']
                gpar['value'] = parameter['value']
                gpar['units'] = parameter['units']
                gpar.append()


        if "/experiment/sensors" in efile:                        
            # opens table for sensors class (sensor data)
            sensors_table = efile.get_node('/experiment/sensors')
            d_sensors = sensors_table.row        
        else:
            # creates table for sensors class (sensor data)
            sensors_table = efile.create_table(exp, 'sensors', rec.SensorsTable, "Sensors Table")
            d_sensors = sensors_table.row
            for sensor in self.sensors:
                if sensor['sensor_type']=='Accelerometer' or sensor['sensor_type']=='Hammer' or sensor['sensor_type']=='Module-output' or sensor['sensor_type']=='Module-input':
                    d_sensors['model'] = sensor['model']
                    d_sensors['serial'] = sensor['serial']
                    d_sensors['sensitivity'] = int(sensor['sensitivity'])
                    d_sensors['sensitivity_units'] = sensor['sensitivity_units']
                    d_sensors['location_x'] = sensor['location'][0]
                    d_sensors['location_y'] = sensor['location'][1]
                    d_sensors['location_z'] = sensor['location'][2]
                    d_sensors['location_units'] = sensor['location_units']
                    d_sensors['direction_x'] = sensor['direction'][0]
                    d_sensors['direction_y'] = sensor['direction'][1]
                    d_sensors['direction_z'] = sensor['direction'][2]
                    d_sensors['channel'] = sensor['channel']
                    d_sensors['trigger'] = sensor['trigger']
                    if sensor['trigger']:
                        d_sensors['trigger_value'] = sensor['trigger_value']
                    else:
                        d_sensors['trigger_value'] = 0
                    d_sensors['max_val'] = sensor['max_val']
                    d_sensors['min_val'] = sensor['min_val']
                    d_sensors['sensor_type'] = sensor['sensor_type']
                    d_sensors['units'] = sensor['units']
                    d_sensors.append()

        if "/experiment/specific_parameters" in efile:
            spar_table = efile.get_node('/experiment/specific_parameters')
            spar = spar_table.row      
        else:             
            spar_table = efile.create_table(exp, 'specific_parameters', rec.RecordParameters, "Specific Parameters Table")
            spar = spar_table.row
        
        if '/experiment/data' in efile:
            hdf5_data = efile.get_node('/experiment/data')
        else:
            hdf5_data = efile.create_earray(exp,'data',tables.Float64Atom(), shape=(0,nsensors,int(self.record_length*self.fs)))
        #Create the table to save the moments bewteen input and oupt signal when there is a voltage type of test
        if self.voltage_check_module:
            if '/experiment/OI_moments' in efile:
                io_moments_table = efile.get_node('/experiment/OI_moments')
                io_mom = io_moments_table.row      
            else:
                io_moments_table = efile.create_table(exp,'OI_moments',rec.IO_moments, "Input-Output Moments")
                io_mom = io_moments_table.row
        efile.flush()

        #%% Settup data acquisition and collect data
        print (f'Setting up data acquisition system. Sample Rate: {self.fs}')
        self.setup_daq()
        
        # Defining the buffer
        b_size=int(self.record_length*self.fs) # MJB: the whole thing to int, not just the record len
        self.__b__ = buffer(b_size)
        if self.__vision__ == True:
            self.__vb__= buffer(int(self.record_length*10))
    
        # Determining what channels should be compared for trigger
        #This line only considers the triggers of the sensors different of type output
        input_sensors = [sensor for sensor in self.sensors if sensor['sensor_type'] != 'Module-output']
        out_sensors = [sensor for sensor in self.sensors if sensor['sensor_type'] == 'Module-output']

        trigger_channs = [sensor['trigger'] for sensor in input_sensors]
        trigger_vals = [sensor['trigger_value'] for indx, sensor in zip(trigger_channs, input_sensors)]
        if self.__vision__ == True:
            trigger_channs = trigger_channs[:-self.__ncam__] #TODO: removes cameras from triggering
            trigger_vals = trigger_vals[:-self.__ncam__] #TODO: removes cameras from triggering since triggering is looking in all sensors self.sensors

        # Collecting data and adding it to the buffer
        triggered = False   # Bool to determine if the system was triggered in the past
        collect = True      # Bool to determine if the system should continue collecting
        buffer_filled = False # Bool to determine if the buffer is filled
        extra_frames = 0    # Number of extra frames collected after trigger
        # extra_frames_needed = int(np.ceil((self.record_length - self.pre_trigger)))+1int()
        extra_frames_needed = int(np.ceil((self.record_length - self.pre_trigger)))
        
        #%% Create input channel
        
        print ('Collecting data and filling up the buffer ...')
        # for counter in range(int(np.ceil((self.record_length+2)/1)*10)):
        if self.voltage_check_module: 
            rang = range(int(np.ceil((self.pre_trigger)*10)))
        else:
            rang = range(int(np.ceil((self.pre_trigger+1)*10)))
        for counter in rang: 
            dataA = self.__task__.read(number_of_samples_per_channel=int(self.fs/10))  # Collecting 1/10 second            
            dataA = np.array(dataA).transpose()
            self.__b__.add(dataA.tolist())
            if self.__vision__ == True:
                self.__vb__.add([self.get_frame()])
        
        print ('Buffer filled, waiting for trigger...') # the system will be triggered  by the shaker movement
        while collect:
            dataA = self.__task__.read(number_of_samples_per_channel=int(self.fs/10))  # Collecting 1/10 second
            dataA = np.array(dataA).transpose()
            self.__b__.add(dataA.tolist())
            if self.__vision__ == True:
                self.__vb__.add([self.get_frame()])

            
            # Testing if the signal is higher than the trigger
            maxvals = np.max(np.abs(dataA),axis = 0); 
            if nsensors == 1 or self.voltage_check_module:
                trigger = (maxvals > trigger_vals)[0]
            else:
                trigger = (maxvals[trigger_channs] > np.array(trigger_vals)[trigger_channs]).any()
            
            
            if trigger and not triggered:
            # if trigger and not triggered:    
                print ('Trigger detected, completting buffer ...')
                # sys.stdout.flush()
                triggered = True

            if triggered:
                extra_frames += 1/10
                # print(str(extra_frames))
                # bar.update()
                if extra_frames >= extra_frames_needed:
                    collect = False
                    print('Finished')
                    # bar.close
        print('waiting 3 seconds before stoping the task...')
        time.sleep(3)        
        # Stop and close the task
        if self.__task_o__:
            self.__task_o__.stop()
        self.__task__.stop()
        

        print('waiting 3 seconds before closing the task...')
        time.sleep(3)  
        if self.__task_o__:
            self.__task_o__.close()
        self.__task__.close()
        

        if self.__vision__ == True:
            self.__cap0__.release()
            self.__cap1__.release()
        
        # if input('plot? (y/n) ') == 'y':
        #%% Cuttine the record to the appropriate length
   
        listA=self.__b__.read()
        if len(self.sensors)==1: #converts the record to a list of lists per value
            for i in range(len(listA)):
                listA[i]=[listA[i]]
            record = np.array(listA)
            
        if self.voltage_check_module: #appends input and output signals to save to file
            for i in range(len(listA)):
                listA[i]=[listA[i]]
            record = np.array(listA)
            record = np.hstack((record,self.output_signal.reshape(record.shape)))
            print(record.shape)
        else:
            for i in range(len(listA)):
                if np.isscalar(listA[i]):
                    listA[i]=[0]*(len(input_sensors)-self.__ncam__) # the list is filled with zeros in the same format #TODO: remove looking in cameras
                
            # record = np.array(self.b.read())
            record = np.array(listA)
            if len(np.where(np.array([sensor['sensor_type'] for sensor in self.sensors])=='Module-output')[0])!=0:
                #locate the column position in the same location as in sensors dictionary
                loc = np.where(np.array([sensor['sensor_type'] for sensor in self.sensors])=='Module-output')[0][0]
                #append the vector of the output signal to the data
                record = np.concatenate((record[:,:loc],self.output_signal.reshape((len(self.output_signal),1)),record[:,loc:]), axis = 1)

        if len(filter_functions)==0: # if no filters loaded, continue with saving the record and video
            decision=True
         
        else:
            decision_list=[0]*len(filter_functions)
            for i in range(len(filter_functions)):
                decision_list[i]=filter_functions[i](record)
            if all(decision_list)==True:
                decision=True
            else:
                decision=False
        
        if decision == False:
            print('dismissing current record...')
         #%% Plotting record for debugging pruposes only
            now = datetime.datetime.now()  
            name=str(now.strftime("%Y_%m_%d-%H-%M-%S"))
           
            mydir = 'data'+'/'+ self.title + "_DismissedPlots" + '/' + name[:10]
          
            # If folder doesn't exist, then create it.
            self.check_folder(mydir)
            
            t = np.arange(0,self.record_length,1/self.fs)
            # plt.clf()
            # print('before plottin')
            plt.figure()
            for sensor, counter in zip(input_sensors,range(len(input_sensors)-self.__ncam__)): #TODO: same, stop looking into cameras
                plt.plot(t,record[:,counter], label = '%s'%sensor['serial'])
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(name)
            plt.legend()
            plt.pause(0.1)
            plt.savefig(mydir + '/' + name + 'DISMISSED.jpg', dpi=150)
            plt.close()
        
        #MJB: Put this here so we don't do it twice.
        # Make our t based on how many enterys we actualy have
        n_samples = int(record.shape[0])
        t = np.linspace(0, (n_samples - 1) / self.fs, n_samples)
        
        if decision == True:     
            #%% Plotting record
            print('Plotting and saving process started')
            now = datetime.datetime.now()  
            name=str(now.strftime("%Y_%m_%d-%H-%M-%S"))
           
            mydir = 'data'+'/'+ self.title + "_Plots" + '/' + name[:10]
                
            # If folder doesn't exist, then create it.
            self.check_folder(mydir)
            
            #t = np.arange(0,self.record_length,1/self.fs) #MJB: sameple rate is not an int walking in the door


            print(f"record length: {self.record_length} seconds")
            print(f"record: {record.shape[0]}")

            plt.figure(1)
            if not self.voltage_check_module and out_sensors: #No voltage but there is an output channel
                for sensor, counter in zip(self.sensors,range(len(self.sensors)-self.__ncam__)): #TODO: same, stop looking into cameras
                    if sensor['sensor_type'] == 'Module-input':
                        plt.figure(2)
                        #Retriving delay in both signals
                        record_tau, signal_tau, t_tau, cross_val_norm = dt.align(record[:,counter], 
                                                                 self.output_signal, 
                                                                 self.record_length,
                                                                 self.fs, t)                        
                        plt.title(r'Time delay removed (Visual purpose only), $\rho_{ij} = %.2f$'%cross_val_norm)
                        plt.plot(t_tau,record_tau, label = '%s'%sensor['serial'])
                        plt.plot(t_tau,signal_tau, label='Sent signal')
                        plt.xlabel('Time (s)')
                        plt.ylabel('peak Voltage')
                        plt.legend()
                        plt.pause(0.1)
                        plt.savefig(mydir + '/' + name + '_%s'%sensor['serial'] + '.jpg', dpi=150)
                        plt.close(2)
                    if sensor['sensor_type'] != 'Module-input':
                        plt.plot(t,record[:,counter], label = '%s'%sensor['serial'])
                        
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.pause(0.1)
                plt.savefig(mydir + '/' + name + '.jpg', dpi=150)            

                plt.close(1) 
                
            elif not self.voltage_check_module and not out_sensors: #no voltage tests and no output channel
                plt.figure(3)
                for sensor, counter in zip(self.sensors,range(len(self.sensors)-self.__ncam__)): #TODO: same, stop looking into cameras
                    plt.plot(t,record[:,counter], label = '%s'%sensor['serial'])
        
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.pause(0.1)
                plt.savefig(mydir + '/' + name + '.jpg', dpi=150) 
                plt.close(3)
                
            elif self.voltage_check_module: #voltage test, there is an output channel 
                mean_val = [];std_val=[]
                plt.figure(4)
                for sensor, counter in zip(self.sensors,range(len(self.sensors)-self.__ncam__)):
                    if sensor['serial'] == 'SMI':
                        record_input = record[:,counter]
                    elif sensor['serial'] == 'SMO':
                        record_output = record[:,counter]
                        
                    control_noise_mean = np.mean(record[:,counter][0:int(self.pre_trigger*self.fs)])
                    mean_val.append(np.mean(record[:,counter], axis=0)-control_noise_mean)
                    std_val.append(np.std(record[:,counter]))
                        
                #Retriving delay in both signals
                record_tau, signal_tau, t_tau, cross_val_norm = dt.align(record_input, 
                                                         record_output, 
                                                         self.record_length,
                                                         self.fs, t)                   
                    
                
                plt.title(r'Time delay removed (Visual purpose only), $\rho_{ij} = %.2f$'%cross_val_norm)
                plt.plot(t_tau,record_tau, label = 'SMI')
                plt.plot(t_tau,signal_tau, label='SMO')
                plt.xlabel('Time (s)')
                plt.ylabel('peak Voltage')
                plt.legend()
                plt.pause(0.1)
                plt.savefig(mydir + '/' + name + '_%s'%sensor['serial'] + '.jpg', dpi=150)
                plt.close(4)  
            plt.close(1)


          
            
            #t = np.arange(0,self.record_length,1/self.fs)
            # plt.clf()
            # print('before plottin')
            plt.figure()
            for sensor, counter in zip(self.sensors,range(len(self.sensors)-self.__ncam__)): #TODO: same, stop looking into cameras
                plt.plot(t,record[:,counter], label = '%s'%sensor['serial'])
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(name)
            plt.legend()
            plt.pause(0.1)
            plt.savefig(mydir + '/' + name + '.jpg', dpi=150)
            plt.close()

            #%% Ask if the record should be saved or not
            # if input('Save this record? (y/n) ') == 'y':
                # Save Video file TODO: input name file and add date
            
    
            if self.__vision__ == True:
                self.save_video(name) # retrieves video filename to add into the hdf5 specific Date field
            # Save the record in the hdf5 file
            print(f"appending data: hdf5_data:{hdf5_data.shape}, {record.shape}")
            hdf5_data.append(np.expand_dims(record.transpose(), axis = 0))
            data_id = hdf5_data.shape[0]-1
            for param in self.parameters['specific']:
                if self.parameters['specific']==['Date']:
                    spar['id'] = data_id
                    spar['parameter'] = param
                    spar['value'] = name
                    spar['units'] = 'date'
                    spar.append()            
                else:          
                    spar['id'] = data_id
                    spar['parameter'] = param
                    spar['value'] = input('What is the value for %s? '%(param))
                    spar['units'] = input('Units for %s? '%(param))
                    spar.append()
            #Adding moments to table
            if self.voltage_check_module:
#                print('mean =%s'%mean)
#                print('std =%s'%std)
                for sen_i in range(len(self.sensors)):
                    io_mom['mean_val'] = mean_val[sen_i]
                    io_mom['std_val'] = std_val[sen_i] 
                    io_mom.append()                

            
        # Clossing HDF5 file in append mode
        efile.flush()
        efile.close()
        
        # Re-loading the experiment to load the new records
        self.load_setup(self.__conf_file__)
        


    def load_setup(self,fname='setup.json',data_folder = 'data',populate_trials = True):
        """
        Opens the JSON file containing the setup parameters for the experiment.
        
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

        """
        import json
        
        with open(fname, 'r') as setup_file:
            setup_data = json.load(setup_file)
        self.title = setup_data['title']
        self.__conf_file__ = fname
        self.fs = setup_data['sampling frequency']
        self.record_length = setup_data['record length']
        self.sensors = setup_data['sensors']
        self.parameters = setup_data['parameters']
        self.pre_trigger = setup_data['pre-trigger']
        # self.voltage_check_module =setup_data['voltage_check'] #TODO
        # checks for presence of cameras and changes vision identifier
        for sensor in self.sensors:
            if sensor['sensor_type'] == "Camera":
                self.__vision__ = True #True for cameras, False without cameras 
                self.__ncam__=self.__ncam__+1
                
        # Getting the path to the file name
        # If folder doesn't exist, then create it.
        self.check_folder(data_folder)
        # Setting data folder property for the experiment
        self.__data_folder__ = data_folder
            
        self.fname= data_folder + '/' + self.title+'.hdf5'
        
        # If the file exists, populate the trials list
        if populate_trials:
            self.trials = []
            if os.path.isfile(self.fname):
                self.__efile__ = tables.open_file(self.fname, mode="r")
                data = self.__efile__.get_node('/experiment/data')
                for counter in range(data.shape[0]):
                    self.trials.append(Trial(self,ident = counter))

    def __del__(self):
        # Close hdf5 file
        self.__efile__.close()

    def save_setup(self):
        """
        Saves the configuration of an experiment to a json file
        
        Parameters
        ----------
        fname : str
            File that the parameters for the experiment will be stored into (JSON file)

        """
        import json
        # organizes all of the data collected
        
        datastore = {
            "title": self.title,
            "sampling frequency": self.fs,
            "record length": self.record_length,
            "pre-trigger": self.pre_trigger,
            "sensors": self.sensors,
            "parameters": self.parameters,
            "voltage_check": self.voltage_check_module
        }

        mydir = ("setup")
        
        # If folder doesn't exist, then create it.
        self.check_folder(mydir)
   
        fname= mydir +'/'+ self.title+'.json'
        print(f"Saving Setup: {fname} | fs: {self.fs}")

        
        with open(fname, 'w') as outfile:
            json.dump(datastore, outfile)
    
    def setup(self):
        """
        Asks a series of questions to acquire information about the parameters 
        specific to the experiment. Then uses the save_setup definition to 
        save the parameters in a JSON file.

        """
        
        # Importing packages
        import json
        
        # Defining local functions
        def identify_channel():
            """
            Identifies the channel of the sensor that is tapped.
            
            This function collects data from all the channels available in the DAQ
            connected to the computer.  The sensor channel is identified by calculating
            the maximum amplitude of the channel.  
            
            Returns
            -------
            channel : str
                The channel in the data acquisition system
                
            Notes
            -----
            The sensitivity is set to 1 for all channels.  Therefore, the incorrect
            channel can be identified if sensors of different sensitivity are connected.
            Data is collected at 1652Hz.
            """
            import nidaqmx
            import numpy as np
            
            system = nidaqmx.system.System.local()
            
            devices = []
            channels = []
            ranges = []
            fs = 413 #1652 413
            
            with nidaqmx.Task() as task:
                
                for device in system.devices:
                    d_name = device.name
                    devices.append(d_name)
                    for chan in device.ai_physical_chans:
                        c_name = chan.name
                        channels.append(c_name)
                        
                        task.ai_channels.add_ai_accel_chan(physical_channel = c_name, sensitivity = 1, min_val = -10, max_val = 10)
                        
                task.timing.cfg_samp_clk_timing(fs, samps_per_chan=fs*5,)
                data = task.read(number_of_samples_per_channel=fs*5)
                for item in data:
                    r = np.max(item) - np.min(item)
                    ranges.append(r)
                
            # Identify the channel with the biggest range
            index = ranges.index(max(ranges))
            return channels[index]


        # brings the information from the json file containing the constants for the sensors
        with open('sensors.json', 'r') as accel:
            acc = json.load(accel)

        # Ask the experiment's title
        self.title = input('Title for this experiment? ')
        
        # Ask about the sampling frequency
        fs = input('What is the sampling frequency in Hz? (Default 1652) ')
        if fs: # If the string is not empty
            fs = int(fs)
        else:
            fs = 1652
        self.fs = fs
        
        # Asking about the pre-trigger
        self.pre_trigger = float(input('Pre_trigger length in seconds? '))
        
        # Asking if the test is a voltage checking
        self.voltage_check_module = bool(input('Are you checking the voltage between input and output channels (True/False)? '))
        
        # Ask about the record length
        flag = True
        while flag:
            self.record_length = float(input('Record length in seconds? '))
            if self.record_length >= self.pre_trigger:
                flag = False
            else:
                print('Record length needs to be longer than pre_trigger length. Please re-enter record length. ')   
                    
        # collects sensor data (model, serial number, sensitivity, location, direction, and channel)
        ns = input('How many sensors? ')
        for x in range(1, int(ns)+1):
            sensor={}
            flag = True
            while flag:
                try:        
                    sensor['model'] = input('What is the model of sensor %i? '%(x))
                    sensor['serial'] = input('What is the serial number of sensor %i? '%(x))
                    if acc[sensor['model']][sensor['serial']]['type'] == "Accelerometer" or acc[sensor['model']][sensor['serial']]['type'] == "Force" or acc[sensor['model']][sensor['serial']]['type'] == "Sound":
                        sensor['sensitivity'] = acc[sensor['model']][sensor['serial']]['sensitivity']
                        sensor['sensitivity_units'] = acc[sensor['model']][sensor['serial']]['sensitivity_units']
                        sensor['sensor_type'] = acc[sensor['model']][sensor['serial']]['type']
                        sensor['units'] = acc[sensor['model']][sensor['serial']]['units']
                    elif acc[sensor['model']][sensor['serial']]['type'] == "Hammer":
                        sensor['extender'] = input('no_extender or steel_extender? ')
                        sensor['sensitivity'] = acc[sensor['model']][sensor['serial']][sensor['extender']]['sensitivity']
                        sensor['sensitivity_units'] = acc[sensor['model']][sensor['serial']][sensor['extender']]['sensitivity_units']
                        sensor['sensor_type'] = acc[sensor['model']][sensor['serial']]['type']
                        sensor['units'] = acc[sensor['model']][sensor['serial']]['units']
                    elif acc[sensor['model']][sensor['serial']]['type'] == "Camera":
                        r = int(input('resolution? [0] 1080p, [1] 720p] ')) #TODO: retrieve available resolutions
                        sensor['resolution'] = acc[sensor['model']][sensor['serial']]['resolution'][r]
                        sensor['resolution_units'] = acc[sensor['model']][sensor['serial']]['resolution_units']
                        sensor['sensor_type'] = acc[sensor['model']][sensor['serial']]['type']
                        sensor['intrinsic_parameters'] = acc[sensor['model']][sensor['serial']]['intrinsic_parameters']
                        sensor['intrinsic_values'] = acc[sensor['model']][sensor['serial']]['intrinsic_values']
                    elif acc[sensor['model']][sensor['serial']]['type'] == "Module-input":
                        sensor['sensitivity'] = acc[sensor['model']][sensor['serial']]['sensitivity']
                        sensor['sensitivity_units'] = acc[sensor['model']][sensor['serial']]['sensitivity_units']
                        sensor['sensor_type'] = acc[sensor['model']][sensor['serial']]['type']
                        sensor['units'] = acc[sensor['model']][sensor['serial']]['units']
                    elif acc[sensor['model']][sensor['serial']]['type'] == "Module-output":
                        sensor['sensitivity'] = acc[sensor['model']][sensor['serial']]['sensitivity']
                        sensor['sensitivity_units'] = acc[sensor['model']][sensor['serial']]['sensitivity_units']
                        sensor['sensor_type'] = acc[sensor['model']][sensor['serial']]['type'] 
                        sensor['units'] = acc[sensor['model']][sensor['serial']]['units']
                    else:
                        raise NameError('Unknown sensor type')
                    flag = False
                except KeyError:
                    print('Model and serial number do not match records. Please re-enter model and serial number.')
        # collect sensor position information
            loc_x = input('What is the x position of sensor %i? '%(x))
            loc_y = input('What is the y position of sensor %i? '%(x))
            loc_z = input('What is the z position of sensor %i? '%(x))
            loc_u = input('What are the location units of sensor %i? '%(x))
            loc = [float(loc_x), float(loc_y), float(loc_z)]    
            sensor['location'] = loc
            sensor['location_units'] = loc_u
        # collect sensor direction information
            dir_x = input('What is the x direction of sensor %i? '%(x))
            dir_y = input('What is the y direction of sensor %i? '%(x))
            dir_z = input('What is the z direction of sensor %i? '%(x))
            dire = [float(dir_x), float(dir_y), float(dir_z)]        
            sensor['direction'] = dire
        # Trigger level
           #TODO: the trigger levels are required for all sensors in order for the triggering occur
            if acc[sensor['model']][sensor['serial']]['type'] == "Accelerometer" or acc[sensor['model']][sensor['serial']]['type'] == "Force" or acc[sensor['model']][sensor['serial']]['type'] == "Sound" or acc[sensor['model']][sensor['serial']]['type'] == "Module-input" or acc[sensor['model']][sensor['serial']]['type'] == "Module-output":
                tmp = input('Trigger level for this sensor in %s? (Empty for no-trigger) '%sensor['units'])
                if tmp:
                    sensor['trigger'] = True
                    sensor['trigger_value'] = float(tmp) 
                else:
                    sensor['trigger'] = False
                # Maximum and minimum value to record
                sensor['max_val'] = float(input('Maximum value to record in %s: '%sensor['units']))
                sensor['min_val'] = float(input('Minimum value to record in %s: '%sensor['units']))
            
            
            sensor['comments'] = input('Comments for this sensor? ')

        # channel information
            if acc[sensor['model']][sensor['serial']]['type'] == "Accelerometer" or acc[sensor['model']][sensor['serial']]['type'] == "Force" or acc[sensor['model']][sensor['serial']]['type'] == "Sound":
          
                yn = 'n'
                while not(yn == 'y'):
                    tmp = input('Tap sensor %i (you have 5 seconds after pressing enter)'%x)
                    chan = identify_channel()
                    yn = input('Is sensor %i (SN: %s) in %s? (y/n) '%(x,sensor['serial'],chan))
                sensor['channel'] = chan
            
            if acc[sensor['model']][sensor['serial']]['type'] == "Module-input" or acc[sensor['model']][sensor['serial']]['type'] == "Module-output":
                chan = input('To what channel is sensor %i (SN: %s) connected? (E.g CDAQ1Mod1/ai0)'%(x,sensor['serial']))
                sensor['channel'] = chan
            self.sensors.append(sensor)
            
        # gathers information about the general parameters of the experiment
        gen_par = input('How many general parameters will there be? ')
        general = []
        for x in range(1, int(gen_par)+1):
            gpa = input('What is general parameter %i? '%(x))
            gen={}
            gen['parameter'] = gpa
            gen['value'] = input('What is the value for ' + gpa + '? ')
            gen['units'] = input('What are the units for the ' + gpa + '? ')
            general.append(gen)
        
        # gathers information about the information specific to the data
        spe_par = input('How many specific parameters will there be? ')
        specific = []
        for x in range(1, int(spe_par)+1):
            spa = input('What is specific parameter %i? '%(x))
            specific.append(spa)
        
        # oraganizes all of the data collected
        self.parameters['general'] = general
        self.parameters['specific'] = specific
        # checks for presence of cameras and changes vision identifier
        for sensor in self.sensors:
            if sensor['sensor_type'] == "Camera":
                self.__vision__ = True #1 for cameras, 0 without cameras 
                self.__ncam__=self.__ncam__+1
        self.save_setup()
