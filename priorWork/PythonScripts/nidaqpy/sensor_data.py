"""
This script creates a json file that contains the parameters of the sensors
used for the DAQ.  The output file is sensors.json
"""
import json

# data for all sensors
sen = {"393B05":
        {"47085":  # ASSET sensor
            {"sensitivity": 9980,
             "sensitivity_units": 'mV/g',
             "type": 'Accelerometer',
             "units": 'g',
             "bias_level" : 11400,
             "bias_level_units":"mVDC"},
        "46978":  # ASSET sensor
            {"sensitivity": 10030,
             "sensitivity_units": 'mV/g',
             "type": 'Accelerometer',
             "units": 'g',
             "bias_level" : 11400,
             "bias_level_units":"mVDC"},
        "47083":
            {"sensitivity": 10020,
             "sensitivity_units": 'mV/g',
             "type": 'Accelerometer',
             "units": 'g',
             "bias_level" : 11400,
             "bias_level_units":"mVDC"},
         "46339":
             {"sensitivity": 10300,
              "sensitivity_units": 'mV/g',
              "type":"Accelerometer",
              "units": 'g',
              "bias_level": 11500,
              "bias_level_units": "mVDC"}
                },
       "3701D1FA20G":       #model number
          {"7707":              #serial number
            {"sensitivity": 100.4,      
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "7708":              #serial number
            {"sensitivity": 100.1,       
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "7709":              #serial number              
            {"sensitivity": 99.2,        
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "7710":              #serial number
            {"sensitivity": 101.0,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "7711":              #serial number
            {"sensitivity": 99.2,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            }
            },
       "333B50":            #model number
          {"40789":             #serial number
            {"sensitivity": 1054,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "39833":             #serial number
            {"sensitivity": 995,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "39831":             #serial number
            {"sensitivity": 1019,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "40787":             #serial number
            {"sensitivity": 1035,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "40790":             #serial number
            {"sensitivity": 1062,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "39832":             #serial number
            {"sensitivity": 1001,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "40788":             #serial number
            {"sensitivity": 1084,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "40786":             #serial number
            {"sensitivity": 1048,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "LW51249":           #serial number
            {"sensitivity": 982,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "LW51250":           #serial number
            {"sensitivity": 1019,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "LW51384":           #serial number
            {"sensitivity": 995,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            },
           "LW51385":        #serial number
            {"sensitivity": 1019,
             "sensitivity_units": "mV/g",
             "type":"Accelerometer",
             "units":"g"
            }
            },
      "C615":            #model number
          {"0001":         #serial number
               {"resolution_units":[["name","h_resolution_pixels","v_resolution_pixels","fps"]],
                "resolution":[["1080p",1920,1080,30],["720p",1024,720,60]],
                "intrinsic_parameters":["c_x","c_y","f_x","f_y","k_1","k_2","k_3","p_1","p_2"],
                "intrinsic_values":[0,0,0,0,0,0,0,0,0],
                "type":"Camera",
                "brand":"logitech",
                "owner":"USC",   
                },
           "0002":         #serial number
               {"resolution_units":[["name","h_resolution_pixels","v_resolution_pixels","fps"]],
                "resolution":[["1080p",1920,1080,30],["720p",1024,720,60]],
                "intrinsic_parameters":["c_x","c_y","f_x","f_y","k_1","k_2","k_3","p_1","p_2"],
                "intrinsic_values":[0,0,0,0,0,0,0,0,0],
                "type":"Camera",
                "brand":"logitech",
                "owner":"USC",               
                },
              },
     
              
       "393B31":            #model number
          {"51814":         #serial number
               {"sensitivity":9950,
                "sensitivity_units":"mV/g",
                "type":"Accelerometer",
                "units":"g",
                "bias_level":11500,
                "bias_level_units":"mVDC",
                "model":"M393B31",
                "owner":"SFSU",
                },
           "51815":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9890,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11500,
                "owner":"SFSU",
                },
           "51819":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9710,
                "bias_level_units":"mVDC",
                "bias_level":11800,
                "units":"g",
                "owner":"SFSU",
                },
           "51820":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9940,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11800,
                "owner":"SFSU",
                },
           "51835":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9770,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11400,
                "owner":"SFSU",
                },
           "51836":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9730,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11300,
                "owner":"SFSU",
                },
           "58288":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9770,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11000,
                "owner":"UofSC",
                },
           "58290":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9750,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11000,
                "owner":"UofSC",
                },
           "58291":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9770,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11000,
                "owner":"UofSC",
                },
           "58292":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9960,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11100,
                "owner":"UofSC",
                },
           "58293":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9950,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11200,
                "owner":"UofSC",
                },
           "58593":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9770,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11400,
                "owner":"UofSC",
                },
           "61292":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9910,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":12100,
                "owner":"SFSU",
                },
           "61321":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9700,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":12500,
                "owner":"SFSU",
                },
           "61462":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9770,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11300,
                "owner":"SFSU",
                },
           "61592":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9640,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11400,
                "owner":"SFSU",
                },
           "61593":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9620,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11400,
                "owner":"SFSU",
                },
           "61594":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9710,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11200,
                "owner":"SFSU", 
                },
           "61595":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9660,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11200,
                "owner":"SFSU",    
                },
           "61596":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9620,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11300,
                "owner":"SFSU",
                },
           "61603":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9820,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11100,
                "owner":"SFSU",
                },
           "61605":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9580,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11900,
                "owner":"SFSU",  
                },
           "63998":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":9910,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11200,
                "owner":"UofSC",
                 },
           "63999":         #serial number
               {"type":"Accelerometer",
                "sensitivity_units":"mV/g",
                "sensitivity":10110,
                "units":"g",
                "bias_level_units":"mVDC",
                "bias_level":11200,
                "owner":"UofSC",
                 },  
           "69611":   #serial number
                 {"sensitivity": 9940,
                  "sensitivity_units": 'mV/g',
                  "type": 'Accelerometer',
                  "units": 'g',
                  "bias_level" : 11700,
                  "bias_level_units":"mVDC",
                  "owner":"UofSC",
                 },
           "70565":    #serial number
                {"sensitivity": 10050,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11200,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70752":    #serial number
                {"sensitivity": 10250,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11100,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70003":    #serial number
                {"sensitivity": 9600,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 10800,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70566":    #serial number
                {"sensitivity": 9790,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11200,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70753":    #serial number
                {"sensitivity": 9590,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11200,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70754":    #serial number
                {"sensitivity": 9800,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11100,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70564":    #serial number
                {"sensitivity": 10310,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11200,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },
           "70751":    #serial number
                {"sensitivity": 9560,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11600,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 }, 
           "72538":    #serial number
                {"sensitivity": 9940,
                 "sensitivity_units": 'mV/g',
                 "type": 'Accelerometer',
                 "units": 'g',
                 "bias_level" : 11200,
                 "bias_level_units":"mVDC",
                 "owner":"UofSC",
                 },                     
               },
              
               
       "3711B1110G":        #model number
          {"LW8813":        #serial number
            {"sensitivity":198,
             "sensitivity_units":"mV/g",
             "type":"Accelerometer",
             "units":"g",
             "bias_level":-1.9,
             "bias_level_units":"mVDC",
             "owner":"SFSU",
             }
            },
       "086D05":            #model number
           {"36630":        #serial number
            {"type":"Hammer",
             "units":"N",
             "owner":"SFSU",
                "no_extender":
                {"sensitivity": 0.23,
                "sensitivity_units":"mV/N",
                },
                "steel_extender":
                {"sensitivity": 0.23,
                 "sensitivity_units": "mV/N"
                } 
             
            }
           },
            
       "086D20":            #model number
           {"40304":        #serial number
               {"type":"Hammer",
                "units":"N",
                "owner":"SFSU",
                "no_extender":
                {"sensitivity": 0.2508,
                "sensitivity_units":"mV/N",
                },
                "steel_extender":
                {"sensitivity": 0.2508,
                 "sensitivity_units": "mV/N"
                } 
              }
           },
               
       "130F20":            #model number
          {"51061":            #serial number
            {"sensitivity": 40.2,
             "sensitivity_units": "mV/Pa",
             "type":"Sound",
             "units":"Pa"
            }
            },
       "208C03":            #model number
          {"LW48153":            #serial number
            {"sensitivity": 2.414,
             "sensitivity_units": "mV/N",
             "type":"Force",
             "units":"N"
            },
           "LW48154":            #serial number
            {"sensitivity": 2.355,
             "sensitivity_units": "mV/N",
             "type":"Force",
             "units":"N"
            },
           "LW48155":            #serial number 
            {"sensitivity": 4.405,
             "sensitivity_units": "mV/N",
             "type":"Force",
             "units":"N"
            },
           "LW48161":            #serial number
            {"sensitivity": 2.385,
             "sensitivity_units": "mV/N",
             "type":"Force",
             "units":"N"
            }
            },
       "086C03":            #model number
          {"23410":              #serial number
            {"type":"Hammer",
             "units":"N",
             "no_extender": 
              {"sensitivity": 2.22,
               "sensitivity_units": "mV/N"
              },
             "steel_extender":
              {"sensitivity": 2.33,
               "sensitivity_units": "mV/N"
              }
            }
          },
       "086D50":            #model number
          {"31296":              #serial number
            {"type":"Hammer",
             "units":"N",
             "no_extender":
              {"sensitivity": 2.22,
               "sensitivity_units": "mV/N"
              },
             "steel_extender":
              {"sensitivity": 2.33,
               "sensitivity_units": "mV/N"
              }
            }   
         }
      }


with open('sensors.json', 'w') as outfile:
    json.dump(sen, outfile)
    