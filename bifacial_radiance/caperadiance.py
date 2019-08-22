from multiprocessing import Pool
from functools import partial

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .main import RadianceObj,AnalysisObj

project_name = 'IncaFixed'
# lspv_path = Path("C:/Users/tc256760/Documents/Modelisation Framework/lspv_analyseSoft")
lspv_path = Path.home()/Path("Documents/lspv_analyseSoft")
# ines_meteo_file = lspv_path/'Inca/tmy_INCA_bifi_1H.hdf'

project_path = lspv_path/'RayTracing_simulations/dev_nbs'/project_name
if not project_path.exists(): project_path.mkdir()


metadata = {'Name': 'Chamberytown',
            'latitude': 45.637, 
            'longitude': 5.881, 
            'altitude': 235.0, 
            'State':'NZ', 
            'USAF':1, 
            'TZ':-1.0}


def get_time_interval(inca, date):
    sl = inca.input_meteo.index.get_loc(date)
    if type(sl) is slice:
        return sl.start, sl.stop
    else: return (sl, sl+1)


def define_meteo(inca, ines_meteo_file):
    meteo = pd.read_hdf(ines_meteo_file, key='df')
    input_meteo = meteo.meteo[['ghi', 'dni', 'dhi']].rename(columns=str.upper)
    inca.input_meteo = input_meteo
    return inca.readInesMeteoFile(input_meteo, metadata)

def define_scene(inca, monitor=5):
    inca.setGround(0.4)

    mod1 = 'test'
    inca.makeModule(name=mod1,x=0.99,y=1.65,numpanels = 1,xgap=0.04,ygap=0.05)
    mod2 = 'prismSolar'
    inca.makeModule(name=mod2, x=0.99,y=1.65,numpanels = 2,xgap=0.04,ygap=0.05)
    sceneObjs = []
    sceneDict1 = {'tilt':30,'pitch': 9.5,'clearance_height':1.63,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': -3.09, 'originy': 0.736} 
    sceneObjs += [inca.makeScene(moduletype=mod1,sceneDict=sceneDict1, hpc=True)]

    sceneDict2 = {'tilt':30,'pitch': 9.5,'clearance_height':0.77,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 0, 'originy': 0} 
    sceneObjs += [inca.makeScene(moduletype=mod2,sceneDict=sceneDict2, hpc=True)]

    sceneDict3 = {'tilt':30,'pitch': 9.5,'clearance_height':1.63,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': 3.09, 'originy': 0.736} 
    sceneObjs += [inca.makeScene(moduletype=mod1,sceneDict=sceneDict3, hpc=True)]

    sceneDict4 = {'tilt':30,'pitch': 9.5,'clearance_height':0.77,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 6.17, 'originy': 0} 
    sceneObjs += [inca.makeScene(moduletype=mod2,sceneDict=sceneDict4, hpc=True)]

    sceneDict5 = {'tilt':30,'pitch': 9.5,'clearance_height':0.77,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': -6.17, 'originy': 0} 
    sceneObjs += [inca.makeScene(moduletype=mod2,sceneDict=sceneDict5, hpc=True)]

    sceneDict6 = {'tilt':30,'pitch': 9.5,'clearance_height':0.77,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': -12.36, 'originy': 0} 
    sceneObjs += [inca.makeScene(moduletype=mod2,sceneDict=sceneDict6, hpc=True)]

    sceneDict7 = {'tilt':30,'pitch': 9.5,'clearance_height':0.77,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 12.36, 'originy': 0} 
    sceneObjs += [inca.makeScene(moduletype=mod2,sceneDict=sceneDict7, hpc=True)]
    
    inca.monitored_obj = sceneObjs[monitor]
    return inca.monitored_obj

def add_ref_cell(inca):
    moduletype_refCell = 'celda_ref'
    inca.makeModule(name=moduletype_refCell,x=0.156,y=0.156,numpanels = 1,xgap=0.04,ygap=0.05)
    sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.05,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': -12.90, 'originy': 0} 
    sceneObj_rCell = inca.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)
    return sceneObj_rCell

def add_diag_posts(inca):
    name='DPost1'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t -15.965 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost2'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t -12.8750 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost3'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t -9.785 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost4'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t -6.685 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost5'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t -3.595 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost6'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t 5.655 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='DPost7'
    text='! genbox black DiagPost 0.12 3.0 0.24 | xform -rx 30 -t 0 0 0.5622 -a 1 -t 0.196 0 0 -a 2 -t 0 9.5 0 -i 1 -t -0.0 -0.0 0 -rz 0 -t 8.745 -1.33 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)
    
    return

def add_vert_posts(inca):
    name = 'post1'
    text= '! genbox black VertPost 0.12 0.24 0.77 | xform -t -15.965 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post2'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t -12.8750 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post3'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t -9.785 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post4'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t -6.685 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post5'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t -3.595 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post6'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t 5.655 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)

    name='Post7'
    text='! genbox black VertPost 0.12 0.24 0.77 | xform -t 8.745 -1.45 0'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)
    
    return
    
def add_box(inca):
    name='Boite_electrique'
    text='! genbox black originMarker 0.12 0.20 0.24 | xform -t -12.90 0.3725 1.13'
    customObject = inca.makeCustomObject(name,text)
    inca.appendtoScene(inca.scene.radfiles, customObject, '!xform -rz 0', hpc=True)
    return

def compute_radiance(timeindex, inca, sim_name='sim'):
    if (inca.input_meteo.iloc[timeindex,0] >= 10) or  (inca.input_meteo.iloc[timeindex,1]>10):
        skyname = inca.gendaylit(inca.metdata,timeindex) 
        filelist = inca.getfilelist()
        filelist[1] = skyname
        octfile = inca.makeOct(filelist, octname = 'inca_period_test'+str(timeindex) )
        analysis = AnalysisObj(octfile, inca.basename) 
        frontscan, backscan = analysis.moduleAnalysis(inca.monitored_obj)#(scene, modWanted = modWanted, rowWanted = rowWanted, sensorsy=sensorsy)
        front, back = analysis.analysis(octfile, sim_name+str(timeindex), frontscan, backscan) 
        return front['Wm2'] + back['Wm2']
    else:
        return [0]*(2*9)

def delete_oct_files(project_path):
    files = [file for file in project_path.iterdir() if file.name.endswith('.oct')]
    for f in files:
        f.unlink()
    print(f'Deleted {len(files)} .oct files')
    return

if __name__ == '__main__':
    sim_name = 'inca_period_test'
    ines_meteo_file = Path.home()/'DATA/INCA/chic_bi3p/tmy_INCA_bifi_1H.hdf'
    sensorsy=9
    inca = RadianceObj(sim_name, str(project_path.absolute())) # Radiance object named inca_first_test
    define_scene(inca)
    define_meteo(inca, ines_meteo_file)
    
    date = '18 July 2017'
    ti, tf = get_time_interval(inca, date)
    print(f'Timeindexes : {ti}, {tf}')
    
    f = partial(compute_radiance, inca=inca, sim_name=sim_name)
    p = Pool(6)
    res_list = p.map(f, range(ti,tf))
    p.close()
    p.join()

    res_df = pd.DataFrame(data=res_list, index = inca.input_meteo[date].index, 
                          columns = [f'g_{i}' for i in range(sensorsy)]+[f'gb_{i}' for i in range(sensorsy)])

    res_df.to_hdf(project_path/'sim_18jul.hdf', key='df')
    res_df.to_csv(project_path/'sim_18jul.csv')

    print("THE END")