# -*- coding: utf-8 -*-
"""
Utilities to deal with the videos
"""

import experiment as ex
import os
import tables
       
                
def read_skels(input_dir=''):
    r""" 
    Reads existing keypoints JSON file
    
    Parameters:
    -----------
   openpose_dir  (str): openpose root folder containing temp_dir for estimated skeletons from openpose
    
    Returns:
    --------
    skels     (list): List of skeletons per frame
        
    """
    import json
    import re
    import glob

    skels=[]
    #reading the output JSON files to extract number of persons
    fileList = glob.glob(input_dir + '//*.json')
    #Organizing file list into alphanum key
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    fileList=sorted(fileList, key = alphanum_key)
    
    
    for file in fileList:
        
        with open(file) as f:
            data = json.load(f)
            skels.append(data)
    return skels

class VideoExperiment(ex.Experiment):
    
    def __init__(self,title = '', fs = 1652, record_length = 30, pre_trigger = 10, sensors = [], parameters = {}):
        super().__init__(title = title, fs = fs, record_length = record_length, pre_trigger = pre_trigger, sensors = sensors, parameters = parameters)
        self.skels_dir = r'' #defines the skeletons directory location
        self.openpose_dir = r'' #defines the root instalation of open pose backend
    
    def load_setup(self,fname='setup.json'):
        ex.Experiment.load_setup(self,fname = fname)
        
         # Recreating the list of trials as VideoTrials
        self.trials = []
        if os.path.isfile(self.fname):
            efile = tables.open_file(self.fname, mode="r")
            data = efile.get_node('/experiment/data')
            for counter in range(data.shape[0]):
                self.trials.append(VideoTrial(self,ident = counter))
        


class VideoTrial(ex.Trial):
    

    
    def sequential_frame_grab(self,output_dir = [] ,sp = 10,scale=1):
        r""" 
            Extracts every sp frames and writes those as images on a
            folder located output_dir
            if an empty folder is suplied, the frames will return to memory 
            as a list
        
            Parameters:
            -----------
                sp          (int): frame step
                output__dir (str): output folder to write the extracted frames
                scale       (int): Value from 0 to 1
            Returns:
            --------  
                temp_list  (list): frames
                
        """
        import cv2
        import numpy as np
            # If folder doesn't exist, then create it.
        cap = cv2.VideoCapture(self.video_file_name()) #Creates the video object
        amount_of_frames = cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT) #Total frames
        frames=np.arange(0,int(amount_of_frames),sp) #specified frames to extract
        if output_dir==[]:
            temp_list=[0]*np.size(frames)
        
        for i in range(len(frames)):
            cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()  
            frame=cv2.resize(frame,dsize=(0,0),fx=scale,fy=scale)
            if output_dir==[]:
                temp_list[i]=frame
                
            else:
                filename = output_dir + '//'+self.video_file_name()[-23:-4]+'_frame'+str(frames[i])+'.jpg' # outputs frames to output_dir, adds video_file_name
                cv2.imwrite(filename, frame)
        
        if output_dir==[]:        
            return temp_list
    
    def skeletons(self, sp=10):
        r""" 
        Performs skeleton detection using OpenPose for current trial
        outputs keypoints JSON files to OpenPose output folder
        
        Performs video filename retrieving
        Extracts frames at sp spacing between video frames
        Calls OpenPose to work on the extracted frames
        
        Saves output variable to JSON file
        
        Parameters:
        -----------
        openpose_dir (str): Root directory for OpenPose
        skels_dir    (str): Directory for saving skeletons as json file
        sp           (int): Frame step for skeleton estimation
        
        Returns:
        --------
        skels     (list): List with body keypoints per frame
          
        """
        import json
        import os
        import tempfile
    
        with tempfile.TemporaryDirectory() as tmpdir:
                    
            #del_frames_skels(openpose_dir)                                     # Deletes previous files
            self.sequential_frame_grab(output_dir = tmpdir,sp=sp)     # Extracts frames
            # Starts OpenPose skeleton Detection
            os.system(r'cmd /c "cd '+self.experiment.openpose_dir+' & bin\OpenPoseDemo.exe --image_dir %s/ --write_json %s/"'%(tmpdir, tmpdir))
            
            skels=read_skels(input_dir = tmpdir)
    
            print('Finished Skeleton Detection')
            
            # Defining JSON aoutput name to coincide with video file name
            date=self.get_specific_parameter() 
            fname=self.experiment.skels_dir + '//' + date + '.json' #date[0:10]=folder date
            
            # Serializing json  
            json_object = json.dumps(skels, indent = 4)
            
            with open(fname, "w") as outfile: 
                outfile.write(json_object)
            
            return skels


    def person_classifier(self):
        r""" 
        Performs skeleton detection using OpenPose for current trial
        outputs keypoints JSON files to OpenPose output folder
        and retrieves number of skeletons inside JSON files using the mean
        of the total skeletons in all frames for classifying the number of persons
        
        Performs video filename retrieving                  |
        Extracts frames at sp spacing between video frames  |
        Calls OpenPose to work on the extracted frames      | self.skeletons(sp)
        
        
        
        Parameters:
        -----------
        sp    (int): frame step for skeleton calculation
        
        Returns:.
        ---------
        pflag (int): [0] for no person detected
                     [1] One person detected
                     [2] Multiple persons detected
        
        Example:
        --------
        e.trials[924].person_classifier(sp=20) #multiple persons
        e.trials[0].person_classifier(sp=20)   #No persons
        e.trials[778].person_classifier(sp=20)   #1 person
            
            
        """
        import numpy as np
        import json
        import os.path
                
        date=self.get_specific_parameter()
        fname=self.experiment.skels_dir + '//' + date + '.json'
        
        
        if os.path.exists(fname):
            with open(fname) as f:
                skels = json.load(f)
            
            people=[]
            
            for i in range(len(skels)):                 #extracts number of persons per frame into a people list
                people.append(len(skels[i]['people']))
                            
            if np.mean(people)>1.5:
                pflag=2                                 #more than one person classifies as 2
            elif np.mean(people)<1.5 and np.mean(people)>=0.29:
                pflag=1                 
            elif np.mean(people)<0.29:
                pflag=0
        else:
            pflag=[]            
        return pflag
        
    def time_ocr(self,frames_dir = r'C:\Users\FRANCOLJ\openpose\examples\frames',sp=40, p1=[2035,95], p2=[2109,1100]):
        r""" 
        Performs Optical Character Recognition for available frames in 
        frames dir
        
        
        Parameters:
        -----------
        frames_dir   (str): Directory for reading video frames, default for OpenPose folder
        sp           (int): Frame step for retrieving timings
        p1          (list): First corner of text    [x1,y1]
        p2          (list): Opposing corner from p1 [x2,y2]
        
        Returns:
        --------
        frame_timing (list): List with frames filename and associated timings [2 by n]
        
        Example:
        --------
        frame_timing=e.trials[924].time_ocr(frames_dir = r'C:\Users\FRANCOLJ\openpose\examples\frames', sp=20)
   
        frame_timing=e.trials[924].time_ocr(frames_dir = r'C:\Users\FRANCOLJ\openpose\examples\frames', sp=40)
            
        """
        import pytesseract
        import cv2
        # Mention the installed location of Tesseract-OCR in your system 
        # requires download and pip installation
#        pip install pytesseract
#        https://github.com/UB-Mannheim/tesseract/wiki
        # example installation folder C:\Users\FRANCOLJ\AppData\Local\Tesseract-OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\FRANCOLJ\AppData\Local\Tesseract-OCR\tesseract.exe'
        
        
        self.del_frames_skels()                                     # Deletes previous files
        self.sequential_frame_grab(frames_dir=frames_dir,sp=sp)     # Extracts frames
        frame_timing=[]
        for dirName, subDirList, fileList in os.walk(frames_dir):
                    for file in fileList:
                        img = cv2.imread(frames_dir+'\\'+file)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                        img_t=gray[p1[1]:p2[1],p1[0]:p2[0]]
                        img_t = cv2.rotate(img_t, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#                        text = pytesseract.image_to_string(img_t)
                        text = pytesseract.image_to_string(img_t, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.')
#                        config = ("-l eng --oem 1 --psm 7")
#                        text = pytesseract.image_to_string(img_t, config=config)
#                        (English language, LSTM neural network, and single-line of text).
                        frame_timing.append([text,file])

        print('Finished OCR Detection')

        return frame_timing