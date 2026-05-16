# -*- coding: utf-8 -*-
"""
Classes that define the recorder
"""

import tables

class TrialTable(tables.IsDescription):
    """
    Defining the experiments table
    
    This table has the following columns:
        
        - parameter (string): The name of the parameter
        - value (duoble): Value of the parameter
        - units (string): Units of the parameter
        
    For example, for the frequency of acquisition would be:
        - parameter: sampling_frequency
        - value: 1652
        - units: Hz
    
    """
    parameter = tables.StringCol(120)
    value = tables.Float64Col()
    units = tables.StringCol(30)

class SensorsTable(tables.IsDescription):
    """
    Defining the sensors table
    
    This table has the following columns:
        
        - model (string): model of the sensor
        - serial (string): serial number of the sensor
        - sensitivity (float): sensitivity of the sensor
        - sensitivity_units (string): units of the sensitivity
        - units (string): units of the measurement after converted
            to engineering units
        - location_x (float): location of the sensor (x-axis)
        - location_y (float): location of the sensor (y-axis)
        - location_z (float): location of the sensor (z-axis)
        - location_units (string): Units of the location of the sensor
        - direction_x (float): sensor direction (x-axis)
        - direction_y (float): sensor direction (y-axis)
        - direction_z (float): sensor direction (z-axis)
        - channel (string): Channel description from the DAQ
    """
    model = tables.StringCol(15)
    serial = tables.StringCol(15)
    sensitivity = tables.Float64Col()
    sensitivity_units = tables.StringCol(10)
    units = tables.StringCol(10)
    location_x = tables.Float64Col()
    location_y = tables.Float64Col()
    location_z = tables.Float64Col()
    location_units = tables.StringCol(10)
    direction_x = tables.Float64Col()
    direction_y = tables.Float64Col()
    direction_z = tables.Float64Col()
    channel = tables.StringCol(30)
    max_val = tables.Float64Col()
    min_val = tables.Float64Col()
    trigger = tables.BoolCol()           # True if this channel is used for triggering
    trigger_value = tables.Float64Col()
    sensor_type = tables.StringCol(30)   # Sensor type
    units = tables.StringCol(10)         # Units


class RecordParameters(tables.IsDescription):
    """
    Defining the table of the parameters per record
    
    The columns for this table are:
        - id (int): row of the record
        - parameter (string): parameter name
        - value (float): value of the parameter
        - units (float): units of the parameter
    """
    id = tables.IntCol()
    parameter = tables.StringCol(15)
    value = tables.StringCol(150)
    units = tables.StringCol(30)
    
class IO_moments(tables.IsDescription):
    """
    Defines the table that contains the moments between output signal and
    the input collected signal.The moments include the mean, the standard deviation
    and the difference in percentage of the two moments.
    """
    id = tables.IntCol()
    mean_val = tables.Float64Col()
    std_val = tables.Float64Col()
