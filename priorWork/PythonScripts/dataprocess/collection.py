# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:59:27 2022

@author: francolj

Super Experiment

"""

from dataprocess import ExperimentProcess as Experiment
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class Collection():

    def __init__(self):
        '''
        Init of the co

        Returns
        -------
        None.

        '''
        self.experiments= []
        self.trials = []
        
    def load_setup(self,fnames,data_folder = 'data',populate_trials = True, title = 'Collection'):
        
        self.title = title # Setting up the title
        self.__trial_dates__ = [] # Setting up internal variable to cache dates
        self.__gait_index__ = [] # indexes of trials that are clean date
        
        for fname in fnames:
            E = Experiment()
            E.load_setup(fname,data_folder,populate_trials)
            self.experiments.append(E)
            for trial in E.trials:
                self.trials.append(trial)
        
    def status(self):
        '''
        Returns the status of the information cached in the sql database
        from different processing activities such as peak detection.

        Returns
        -------
        status : pandad.DataFrame
            Dataframe with the status of the experiments.  See the help
            in Experiment.status() for a description of the columns of the
            dataframe.  One experiment is reported per row
        '''
        dfs = []
        for experiment in self.experiments:
            dfs.append(experiment.status())
        out = pd.concat(dfs)
        out = out.set_index('Experiment')
        return out
    
    def cnn_classification_histogram(self):
        '''
        Produces a histogram of the probability of each
        Trial in the Collection
        '''        
        
        query = 'select probability from cnn_classification'
        probs = []
        for experiment in self.experiments:
            cursor = experiment.__sto__.__cursor__.execute(query)
            probs = probs + cursor.fetchall()
            
        probs = pd.DataFrame(probs)
        fig, ax = plt.subplots()
        probs.hist(ax=ax)
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.grid(False)
        plt.title('Probability of gait (%s)'%self.title)
        
        return fig
        
    def clean_gait_classify(self, verbose = False):
        """
        Classify the records as clean gait records or not

        Returns
        -------
        classification : List of booleans with the classification

        """
        return Experiment.clean_gait_classify(self,verbose)
    
    def __prepare_df__(self,gait_only = False, is_gait_probability=0.8):
        '''
        This function helps prepare the data to plot activity in heatmaps
        

        Parameters
        ----------
        gait_only : Bool, optional
            True if data should be gait only data. The default is False.
        
        is_gait_probability: float
        The probabilty threshold to accept a record as containing gait information.
        It only works when gait_only is set to True.

        Returns
        -------
        df : pandas.df
            Data frame with the information on the dates.

        '''

        #%% Get the get indexes if needed
        dates = []
        if gait_only:

            query = 'select id from cnn_classification where probability >= %2.2f'%is_gait_probability
            for experiment in self.experiments:
                cursor = experiment.__sto__.__cursor__.execute(query)
                ids = cursor.fetchall()
                for trial_id in ids:
                    date = datetime.strptime(experiment.trials[trial_id[0]].get_specific_parameter("Date").decode(),"%Y_%m_%d-%H-%M-%S")
                    dates.append(date)

        else:
            #%% Adding to data frame all the data
            for trial in self.trials:
                date = datetime.strptime(trial.get_specific_parameter("Date").decode(),"%Y_%m_%d-%H-%M-%S")
                dates.append(date)

        #%% Breaking the record
        df = pd.DataFrame(dates)
        df['Hour'] = df[0].dt.hour
        df['Date'] = df[0].dt.month.astype(str) + '/' + df[0].dt.day.astype(str)
        df['Day'] = df[0] - min(df[0])
        df['Day'] = df['Day'].dt.days
        df['Day of the week'] = df[0].dt.dayofweek

        return df
    
    def week_heatmap(self, gait_only = False, is_gait_probability=0.8):
        '''
        Heatmap showing the number of records as a function of the day of the 
        week (x-axis) and the hour of the day (y-axis).  Darker colors 
        indicate a higher number of records.

        Parameters
        ----------
        gait_only : Bool, optional
            If true only parameters classified as gait are used for the plot.
            The default is False.

        Returns
        -------
        None.

        '''
        
        # Get the dataframe of dates
        df = self.__prepare_df__(gait_only, is_gait_probability=is_gait_probability)

        #%% Pivot table
        sns.heatmap(pd.pivot_table(df, values = 0, index = 'Hour', columns= 'Day of the week', aggfunc='count'), cmap = 'rocket_r')
        plt.xlabel ('Day of the week')
        plt.ylabel ('Hour of the day')
        if gait_only:
            plt.title ('Number of records (gait only)')
        else:
            plt.title ('Number of records')
        
    def activity_heatmap(self, gait_only = False, is_gait_probability =0.8):
        '''
        Plots the activity plot for the collection.  This heatmap has the
        day of installation (starting at zero) in the x-axis and the
        hour of the day in the y-axis.  Darker colors indicate more activity

         Parameters
         ----------
         gait_only : Bool, optional
             If true only parameters classified as gait are used for the plot.
             The default is False.
        
        is_gait_probability: float
        The probabilty threshold to accept a record as containing gait information.
        It only works when gait_only is set to True.
             
        Returns
        -------
        None.

        '''
        
        # Get the dataframe of dates
        df = self.__prepare_df__(gait_only, is_gait_probability=is_gait_probability)

        #%% Pivot table
        sns.heatmap(pd.pivot_table(df, values = 0, index = 'Hour', columns= 'Day', aggfunc='count'), cmap = 'rocket_r')
        plt.xlabel ('Day of installation')
        plt.ylabel ('Hour of the day')
        if gait_only:
            plt.title ('Number of records (gait only)')
        else:
            plt.title ('Number of records')
            
    def heatmaps(self, is_gait_probability=0.8):
        """
        Plots a 2x2 figure.  The top row contains unfiltered data and the
        bottom row is the filtered data (gait only).  The first column
        is the heatmap of the day of installation and hours.  The second
        column is a heatmap of the day of the week and hour
        Parameter
        ---------
        is_gait_probability: float
        The probabilty threshold to accept a record as containing gait information.
        It only works when gait_only is set to True.
        Returns
        -------
        None.

        """
        plt.subplot(2,2,1)
        self.activity_heatmap(gait_only = False)
        plt.subplot(2,2,2)
        self.week_heatmap(gait_only = False)

        plt.subplot(2,2,3)
        self.activity_heatmap(gait_only = True, is_gait_probability=is_gait_probability)
        plt.subplot(2,2,4)
        self.week_heatmap(gait_only = True, is_gait_probability=is_gait_probability)
        
        plt.tight_layout()