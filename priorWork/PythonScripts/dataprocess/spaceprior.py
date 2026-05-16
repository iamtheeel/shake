# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import numpy as np
from nidaqpy import Experiment as Experiment

def tom(vertex,units):
    '''
    This function converts the units from any
    unit to meters.

    Args:
        vertex (list): List of couples or list of lists
            of the form [(x1,y1),(x2,y2) ...].
        units (char): Units.  Units can be 'm', 'ft'
            'in', 'cm'

    Raises:
        Exception: If the unit is not recognized.

    Returns:
        vertex (list): List of the vertex in meters.

    '''
    if units == 'in' or units == 'inches':
        vertex = np.array(vertex) / 39.37
        vertex = vertex.tolist()
    elif units == 'cm' or units == 'centimeters':
        vertex = np.array(vertex)/ 100
        vertex = vertex.tolist()
    elif units == 'ft' or units == 'feet':
        vertex = np.array(vertex)/ 3.281
        vertex = vertex.tolist()
    elif units == 'm' or units == 'meters':
        pass
    else:
        raise Exception('Units %s not recognized'%units)
    return vertex

class SpacePrior():
    
    def __init__(self,parent,vertex,units):
        
        self.__parent__ = parent
        
        # Properties for the spatial prior
        # It is a list of dictionaries of the 
        # following form: {'object': object, 'logp':1}
        # where 'object' is a predetermine object
        # to define priors in space.
        self.__pshapes__ = []
        vertex = tom(vertex, units)
        self.domain = path.Path(vertex, closed = True)
        
    def addpolygon(self,vertex,logp,units,label = ''):
        '''
        Adds a polygon to the installation.  This can be used
        to represent tables, hallways, etc.

        Args:
            vertex (list): List of couples.  For example
                [(x1,y1), (x2,y2), ...].
            logp (float): Log of the probability for the
                polygon.
            units (char): Units of the polygon.
                Possible units are 'm', 'in','cm', 'ft'.
                Units are automatically converted to meters
                and stored in meters for consitency.
            label(char): Label of the polygon

        Returns:
            None.

        '''
        vertex = tom(vertex,units)
        shape = {'object':path.Path(vertex, closed = True),
                 'logp':logp,
                 'units':units,
                 'label':label}
        self.__pshapes__.append(shape)

    def logp(self,point):
        '''
        Prior of the installation.  This is calculated
        as the multiplication of all the magnitudes
        for all polygons

        Args:
            point (tuple): Coordinate (x,y).

        Returns:
            p (float): DESCRIPTION.

        '''
        if self.domain.contains_point(point):
            # If the point is inside the domain, add
            # the logp of all the shapes intersecting
            # at this point
            logp = 0
            for shape in self.__pshapes__:
                if shape['object'].contains_point(point):
                    logp = logp+shape['logp']
        else:
            # If the requested point is ouside the domain
            # the probability should be zero (logp = -inf)
            logp = -np.inf
        return logp

    def plot(self):
        # Create the new figure
        fig,ax = plt.subplots()
        
        # Add a green patch as the domain
        patch = patches.PathPatch(self.domain, facecolor='green', alpha = 0.1)
        ax.add_patch(patch)
        
        # Add red patches on the added polygons
        for shape in self.__pshapes__:
            if isinstance(shape['object'],path.Path):
                patch = patches.PathPatch(shape['object'], facecolor='red', lw=2, alpha = 0.05)
                ax.add_patch(patch)
                coord = shape['object'].vertices[0:-1]
                x = np.mean(coord[:,0])
                y = np.mean(coord[:,1])
                plt.text(x,y,shape['label'],horizontalalignment='center',verticalalignment='bottom')
        
        # If the parent is an experiment, plot the sensors
        if isinstance(self.__parent__,Experiment):
            for sensor in self.__parent__.sensors:
                if sensor['sensor_type'] == 'Accelerometer':
                    coord = tom([sensor['location'][0],sensor['location'][1]],sensor['location_units'])
                    plt.plot(coord[0],coord[1],'ko')
                    plt.text(coord[0],coord[1],sensor['serial'],horizontalalignment='center',verticalalignment='bottom')
        
        # Final details for the plot
        ax.axis('equal')
        plt.xlabel('m')
        plt.ylabel ('m')
        plt.title(self.__parent__.title)