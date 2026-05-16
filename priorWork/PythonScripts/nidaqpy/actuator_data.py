# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:15:57 2022

@author: PEARSAS
"""

import json

#Data for all actuators

act = {"APS113":   #Model Number
       {"2473":     #Shaker serial number
        {"type": "shaker", 
         "frequency_range": "0 to 200",
         "frequency_units": "Hz", 
         "masses_attached": True, 
         "total_mass": 47, 
         "mass_unit": "kg"
         }
        }
       }


with open('actuators.json', 'w') as outfile:
    json.dump(act, outfile)
    
