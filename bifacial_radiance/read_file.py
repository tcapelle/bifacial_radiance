#Version 0.2.5b

simulationParamsDict = {'testfolder': r'C:\Users\sayala\Documents\RadianceScenes\DemoD',
                        'weatherFile': r'C:\Users\sayala\Documents\RadianceScenes\Demo\EPWs\USA_VA_Richmond.Intl.AP.724010_TMY.epw',
                        'getEPW':False,
                        'simulationname': 'Demo1',
                        'custommodule': True,
                        'moduletype': 'Longi',
                        'rewriteModule': True, 
                        'cellLevelModule': False,
                        'axisofrotationTorqueTube': True, 
                        'torqueTube': True,
                        'hpc': True,
                        'tracking': True, 
                         'cumulativeSky': False,
                         'timestampRangeSimulation': False, 
                         'daydateSimulation': True, 
                         'latitude': 37.5,
                         'longitude': -77.6}

timeControlParamsDict = {'HourStart': 11,
                         'HourEnd': 11,
                         'DayStart': 17,
                         'DayEnd': 17,
                         'MonthStart': 2,
                         'MonthEnd': 2,
                         'timeindexstart': 4020,
                         'timeindexend': 4024} 
                             
moduleParamsDict = {'numpanels': 2, 
                    'x': 0.98, 
                    'y': 1.980, 
                    'bifi': 0.90, 
                    'xgap': 0.020, 
                    'ygap': 0.150, 
                    'zgap': 0.100}

sceneParamsDict = {'gcrorpitch': 'pitch',
                   'gcr': 0.350,
                   'pitch': 10.0, 
                   'albedo': 0.30, 
                   'nMods': 20, 
                   'nRows': 7, 
                   'azimuth_ang': 180, 
                   'tilt': 30, 
                   'clearance_height': 0.7, 
                   'hub_height': 2.35, 
                   'axis_azimuth': 180}

trackingParamsDict = {'backtrack': True, 
                      'limit_angle': 60, 
                      'angle_delta': 30}    

torquetubeParamsDict = {'diameter': 0.10, 
                        'tubetype': 'Hex', 
                        'torqueTubeMaterial': 'Metal_Grey'}

analysisParamsDict = {'sensorsy': 9, 
                      'modWanted': 10, 
                      'rowWanted': 3}

cellLevelModuleParamsDict = {'numcellsx': 12, 
                             'numcellsy': 6, 
                             'xcell': 0.150, 
                             'ycell': 0.150, 
                             'xcellgap': 0.100, 
                             'ycellgap': 0.100}