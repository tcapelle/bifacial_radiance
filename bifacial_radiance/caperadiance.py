from multiprocessing import Pool
from functools import partial

from pathlib import Path
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import tqdm
from fire import Fire
from bifacial_radiance.main import RadianceObj,AnalysisObj, _popen

Path.ls = lambda x: sorted(list(x.iterdir()))



metadata = {'Name': 'Chamberytown',
            'latitude': 45.637, 
            'longitude': 5.881, 
            'altitude': 235.0, 
            'State':'NZ', 
            'USAF':1, 
            'TZ':0}


def get_time_interval(model, date):
    sl = model.input_meteo.index.get_loc(date)
    if type(sl) is slice:
        return sl.start, sl.stop
    else: return (sl, sl+1)


def _convert_meteo(df):
    return df[['ghi', 'dni', 'dhi']].rename(columns=str.upper)

def define_meteo(model, df):
    model.input_meteo  = _convert_meteo(df)
    return model.readInesMeteoFile(model.input_meteo, metadata)

def define_scene(model, monitor=5,rack ='rackC3',withGroupF=False):
    
    height    = 0.77
    originy   = -0.1
    sceneObjs = []
     
    if rack  == 'rackC3':
         d = 0
         m1_npanels = 2
         height_m1 = height
         full = True   

    if rack  == 'rackC2': 
         d = 0.736
         m1_npanels = 1
         height_m1 = 1.63
         full = True

    if rack  == 'rackC1': 
         d = 0.736
         m1_npanels = 1
         height_m1 = 1.63
         full = False

    mod1 = 'mod1'
    model.makeModule(name=mod1,x=0.99,y=1.65,numpanels = m1_npanels ,xgap=0.04,ygap=0.05)
    mod2 = 'mod2'
    model.makeModule(name=mod2, x=0.99,y=1.65,numpanels = 2,xgap=0.04,ygap=0.05)

        
    scDict_smGC = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': -3.09, 'originy': originy + d} 
    sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGC, hpc=True)] # sm = single module G: group 

    scDict_GC  = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 0, 'originy': originy } 
    sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GC, hpc=True)] #makeScene creates a .rad file

    scDict_smGD = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': 3.09, 'originy': originy + d} 
    sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGD, hpc=True)]

    scDict_GD = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 6.17, 'originy': originy } 
    sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GD, hpc=True)]

    scDict_GB = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': -6.17, 'originy': originy } 
    sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GB, hpc=True)]

    scDict_GA = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': -12.36, 'originy': originy } 
    sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GA, hpc=True)]

    scDict_GE = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 12.36, 'originy': originy } 
    sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GE, hpc=True)]

    if withGroupF:
        scDict_GF = {'tilt':30,'pitch': 9.5,'clearance_height':height,'azimuth':180, 'nMods': 5, 'nRows': 2, 'appendRadfile':True,'originx': 18.54, 'originy': originy } 
        sceneObjs += [model.makeScene(moduletype=mod2,sceneDict=scDict_GF, hpc=True)]
   
    
    if full == True: 
        
         scDict_smGE = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': 9.26, 'originy': originy + d } 
         sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGE, hpc=True)]

         scDict_smGB = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': -9.26, 'originy': originy + d } 
         sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGB, hpc=True)]

         scDict_smGA = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': -15.45, 'originy': originy + d } 
         sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGA, hpc=True)]
         
         if withGroupF:
            scDict_smGF = {'tilt':30,'pitch': 9.5,'clearance_height':height_m1,'azimuth':180, 'nMods': 1, 'nRows': 2, 'appendRadfile':True,'originx': 15.45, 'originy': originy + d } 
            sceneObjs += [model.makeScene(moduletype=mod1,sceneDict=scDict_smGF, hpc=True)]

    model.module6 = sceneObjs[monitor]
    return model.module6

STRUCT_HEIGHT = 0.60

def genbox(model, 
           name, 
           scene_name='cScene.rad', 
           material='Metal_Aluminum_Anodized',  
           dim=(1.0,1.0,1.0), 
           r=(0,0,0), 
           t=(0.0,0.0,0.0), 
           hpc=True):
    genbox_cmd = f'!genbox {material} {name} {dim[0]} {dim[1]} {dim[2]} '
    xform_cmd = f'| xform -rx {r[0]} -ry {r[1]} -rz {r[2]} -t {t[0]} {t[1]} {t[2]}'
    cmd = genbox_cmd + xform_cmd
    box = model.makeCustomObject(name, cmd)
    model.appendtoScene(scene_name, box,  hpc=hpc)
    return

def add_vert_posts(model,    
                    scene_name='cScene.rad', 
                    material='Metal_Aluminum_Anodized',
                    rowoffset=0,
                    hpc=True):
    height = STRUCT_HEIGHT
    genbox(model,'vert_post1', scene_name, material, dim=(0.12, 0.24, height), t=(-15.965, -1.45 + rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post2', scene_name, material, dim=(0.12, 0.24, height), t=(-12.8750, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post3', scene_name, material, dim=(0.12, 0.24, height), t=(-9.785, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post4', scene_name, material, dim=(0.12, 0.24, height), t=(-6.685, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post5', scene_name, material, dim=(0.12, 0.24, height), t=(-3.595, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post6', scene_name, material, dim=(0.12, 0.24, height), t=(-0.505, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post7', scene_name, material, dim=(0.12, 0.24, height), t=(2.585, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post8', scene_name, material, dim=(0.12, 0.24, height), t=(5.655, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post9', scene_name, material, dim=(0.12, 0.24, height), t=(8.745, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post10', scene_name, material, dim=(0.12, 0.24, height), t=(11.835, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post11', scene_name, material, dim=(0.12, 0.24, height), t=(14.925, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post12', scene_name, material, dim=(0.12, 0.24, height), t=(18.015, -1.45+ rowoffset, 0), hpc=hpc)
    genbox(model,'vert_post13', scene_name, material, dim=(0.12, 0.24, height), t=(21.105, -1.45+ rowoffset, 0), hpc=hpc)
    #soil rack
    genbox(model,'rack_cables', scene_name, material, dim=(24.7, 0.24, 0.1), t=(-15.965, -1.45+0.24+ rowoffset, 0), hpc=hpc)
    return

def pivoting_structure(model, material='Metal_Aluminum_Anodized', angle=30,rowoffset=0, hpc=True):
    def _t(alpha, h, l,rowoffset):
        'disgusting geometry'
        n = np.sqrt(h**2 + l**2)
        alpha = np.deg2rad(alpha)
        beta = np.arctan(h/l)
        gamma = beta-alpha
        y = -l + n*np.cos(gamma)
        z = h - n*np.sin(gamma)
        print(f'alpha, beta, gamma: {alpha, beta, gamma}')
        print(f'n: {n}')
        return (0, y + rowoffset, z)
    add_diag_posts(model, 'pivoting_struct.rad', material)
    add_hor_posts(model, 'pivoting_struct.rad', material)
    add_diag_posts_intra(model, 'pivoting_struct.rad', material)
    t = _t(angle, STRUCT_HEIGHT, 1.45,rowoffset)
    print(f'moving struct to {t}')
    cmd = f'!xform -rx {angle} -t {t[0]} {t[1]} {t[2]} '
    print(cmd)
    model.radfiles.pop() #remove non pivoted scene
    model.appendtoScene(f'pivoting_struct_{angle}.rad', 'objects/pivoting_struct.rad', cmd, hpc=hpc) 
    return

def add_diag_posts(model,    
                    scene_name='cScene.rad', 
                    material='Metal_Aluminum_Anodized',
                    hpc=True):
    length = 3.5
    zheight = 0.24
    height = STRUCT_HEIGHT - zheight
    genbox(model,'diag_post1', scene_name, material, dim=(0.12, length, 0.24), t=(-15.965, -1.45, height), hpc=hpc)
    genbox(model,'diag_post2', scene_name, material, dim=(0.12, length, 0.24), t=(-12.8750, -1.45, height), hpc=hpc)
    genbox(model,'diag_post3', scene_name, material, dim=(0.12, length, 0.24), t=(-9.785, -1.45, height), hpc=hpc)
    genbox(model,'diag_post4', scene_name, material, dim=(0.12, length, 0.24), t=(-6.685, -1.45, height), hpc=hpc)
    genbox(model,'diag_post5', scene_name, material, dim=(0.12, length, 0.24), t=(-3.595, -1.45, height), hpc=hpc)
    genbox(model,'diag_post6', scene_name, material, dim=(0.12, length, 0.24), t=(-0.505, -1.45, height), hpc=hpc)
    genbox(model,'diag_post7', scene_name, material, dim=(0.12, length, 0.24), t=(2.585, -1.45, height), hpc=hpc)
    genbox(model,'diag_post8', scene_name, material, dim=(0.12, length, 0.24), t=(5.655, -1.45, height), hpc=hpc)
    genbox(model,'diag_post9', scene_name, material, dim=(0.12, length, 0.24), t=(8.745, -1.45, height), hpc=hpc) 
    genbox(model,'diag_post10', scene_name, material, dim=(0.12, length, 0.24), t=(11.835, -1.45, height), hpc=hpc) 
    genbox(model,'diag_post11', scene_name, material, dim=(0.12, length, 0.24), t=(14.925, -1.45, height), hpc=hpc) 
    genbox(model,'diag_post12', scene_name, material, dim=(0.12, length, 0.24), t=(18.015, -1.45, height), hpc=hpc)
    genbox(model,'diag_post13', scene_name, material, dim=(0.12, length, 0.24), t=(21.105, -1.45, height), hpc=hpc)
    return

def add_hor_posts(model,    
                    scene_name='cScene.rad', 
                    material='Metal_Aluminum_Anodized',
                    hpc=True):
    size = 0.09
    height = STRUCT_HEIGHT
    length = 3.5 - size
    bottom_left = array([-15.965, -1.45, height])
    top_left = array([-15.965, -1.45+length, height])
    # midde_left = (top_left + bottom_left)/2
    genbox(model,'hor_post_bottom', scene_name, material, dim=(37.08, size, size), t=bottom_left, hpc=hpc)
    genbox(model,'hor_post_top', scene_name, material, dim=(37.08, size, size), t=top_left, hpc=hpc)
    # genbox(model,'hor_post_middle', scene_name, material, dim=(24.7, size, size), t=midde_left, hpc=hpc)
    return

def add_diag_posts_intra(model,    
                    scene_name='cScene.rad', 
                    material='Metal_Aluminum_Anodized',
                    hpc=True):
    zsize = 0.09
    xsize = 0.045
    height = STRUCT_HEIGHT
    length = 3.5
    z_struct=0.09
    modulex = 0.99 + xsize/2
    t = array([-15.965, -1.45, height+z_struct])
    for i in range(24):
        genbox(model,f'diag_post_intra1.{i}', scene_name, material, 
               dim=(xsize, length, zsize), t=t + i*array([modulex,0,0]), hpc=hpc)
    return

def add_box(model,    
            scene_name='cScene.rad', 
            material='beigeroof',
            hpc=True):
    genbox(model,'Boite_electrique', scene_name, material, dim=(0.12, 0.20, 0.24), t=(-12.875, 0.75, 1.36), hpc=hpc)
    # genbox(model,'cables', scene_name, material, dim=(0.04, 0.1, 0.07), t=(-12.83, 0.75, 1.24), hpc=hpc)
    return

def add_really_big_box(model,    
            scene_name='cScene.rad', 
            material='beigeroof',
            hpc=True):
    genbox(model,'building', scene_name, material, dim=(67, 10, 12), r=(0, 0, 15), t=(-25.875, 58, 0), hpc=hpc)
    return

def add_ref_cell(model,group_ref_cell='A2',rowoffset=0):
    if group_ref_cell == 'A2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -12.815, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)

    if group_ref_cell == 'B2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -6.625, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True) 

    if group_ref_cell == 'C2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -0.505, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)

    if group_ref_cell == 'D2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': 5.68, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)

    if group_ref_cell == 'E2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx':11.86, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)

    if group_ref_cell == 'F2':  
        moduletype_refCell = 'ref_cell'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': 18.04, 'originy': 0.55+ rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)       

    if group_ref_cell == '2_B':    
        moduletype_refCell = 'ref_cell_2_b'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':2.0,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -0.44, 'originy': 1.63 + rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)
    if group_ref_cell == '2_A':    
        moduletype_refCell = 'ref_cell_2_a'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':1.3,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -0.44, 'originy': 0.55 + rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)
    if group_ref_cell == '2_C':    
        moduletype_refCell = 'ref_cell_2_c'
        model.makeModule(name=moduletype_refCell,x=0.12,y=0.12,numpanels = 1)
        sceneRef_rCell = {'tilt':30,'pitch': 9.5,'clearance_height':0.46,'azimuth':180, 
                          'nMods': 1, 'nRows': 1, 'appendRadfile':True,'originx': -0.44, 'originy': -0.93 + rowoffset} 
        sceneObj_rCell = model.makeScene(moduletype=moduletype_refCell, sceneDict=sceneRef_rCell, hpc=True)        

    return sceneObj_rCell

                           
 

def delete_oct_files(project_path):
    for f in project_path.glob('*.oct'):
        f.unlink()
    print(f'Deleted .oct files')
    return

def delete_rad_files(project_path):
    for f in (project_path).glob('*/*.rad'):
        f.unlink()
    print(f'Deleted .rad files')
    return

def view(file_list, view='front', program='rvu'):
    'Renders a view of the file'
    views = {'diag': '-vp -17 3 1 -vd 2 -1 -0.3 -vu 0 0 1 -av 0.2 0.2 0.2',
             'side': '-vp -14.3 0.2 1.5 -vd 1 0 0 -vu 0 0 1 -av 0.2 0.2 0.2',
             'side2': '-vp -17 -6 3.5 -vd 0.2 1 -0.3 -vu 0 0 1 -av 10 10 10 -ab 2',
             'back': '-vp -13.215 2 1 -vd 0.3 -1 0 -vu 0 0 1 -av 0.2 0.2 0.2',
             'front': '-vp 2.5 -40 25 -vd 0 1 -0.5 -vu 0 0 1 -av 0.2 0.2 0.2',
             'top': '-vp -17.5 1.6 2.7 -vd 1 0 -0.1 -vu 0 0 1',
             'bottom': '-vp -17.5 -1.55 1.0 -vd 1 0.05 -0.1 -vu 0 0 1',
             'front_low': '-vp -17 -5 2 -vd 0.5 1 -0.05 -vu 0 0 1 -av 0.2 0.2 0.2'}

    program = 'objview' if file_list[0].endswith('rad') else program
    if isinstance(file_list,list):
        files = ' '.join([file_list[0]]+[s[s.find('objects'):] for s in file_list if ('objects' in s)])
    if isinstance(file_list, str):
        files = file_list
    vp = views[view] if view in views else view
    cmd = _cmd(program, vp, files)
    return _popen(cmd, None)

def _cmd(program, vp, filename):
    vp = ' '+vp+' ' if program =='rvu' else f' -v "{vp}" '
    cmd = program + vp + filename
    print(f' cmd: {cmd}')
    return  cmd


def run_simulation(date='18 July 2017', 
                   outfile='new_results', 
                   cores=10, 
                   albedo=0.4,
                   rack = 'rackc3',
                   rowoffset = 9.5,  
                   add_struct=True, 
                   ref_cell=True,
                   group_ref_cell='A2',
                   withGroupF=False,
                   project_name = 'INCAFixed',
                   lspv_path = Path.home()/Path("Documents/lspv_analyseSoft/'RayTracing_simulations/dev_nbs_fc'"),
                   ines_meteo_file = Path.home()/'DATA/INCA/chic_bi3p/tmy_INCA_bifi_5T.hdf'):

    project_path = lspv_path/project_name
    if not project_path.exists(): project_path.mkdir()

    delete_oct_files(project_path)
    delete_rad_files(project_path)
    
    sim_name = 'inca'
    sensorsy = 9
    inca = RadianceObj(sim_name, str(project_path.absolute())) # Radiance object named model_first_test
    inca.setGround(albedo)
    #define the scene with all the modules and scene dicts
    module6 = define_scene(inca, 5,rack,withGroupF)  #unused if ref_cell present

    #add strcutures
    if add_struct:
        pivoting_structure(inca,rowoffset)
        add_box(inca)
        if add_really_big_box:
            add_really_big_box(inca)
    inca.monitored_obj = add_ref_cell(inca,group_ref_cell) if ref_cell else module6 

    #append the ines meteo file
    meteo_df = pd.read_hdf(ines_meteo_file).loc[date,:]
    define_meteo(inca, meteo_df)

    #chose the date of your sim, must be in the input meteo file
    results_file = date.replace(' ', '') + 'w_'+rack+'_sf_'+group_ref_cell+'.hdf' 
    if withGroupF: 
        results_file = date.replace(' ', '') + 'w_'+rack+'_'+group_ref_cell+'.hdf' 
    
    #ti, tf = get_time_interval(inca, date)
    #print(f'Timeindexes : {ti}, {tf}')
    ti = 528
    tf = 529

    import tqdm
    pool = Pool(1)
    res_list = []
    f = partial(compute_radiance, model=inca, sim_name=sim_name, sensorsy=sensorsy)
    for x in tqdm.tqdm(pool.imap(f, range(ti,tf)), total=tf-ti):
        res_list.append(x)
        pass
    pool.close()
    pool.join()

    #pool = Pool(cores)
    #res_list = []
    #print(f'Launching simulation with {cores} processes')
    #f = partial(compute_radiance, model=inca, sim_name=sim_name, sensorsy=sensorsy)
    #if cores <2:
    #    res_list =  list(map(f, range(ti, tf)))
    #   res_list = [f(x) for x in range(ti, tf)]
    #else:
    #for x in tqdm.tqdm(pool.imap(f, range(ti,tf)), total=tf-ti):
    #    res_list.append(x)
    #    pass
    #pool.close()
    #pool.join()
    #print(f'res_list: {res_list}')
    results = pd.DataFrame(data=res_list, 
                           index = inca.input_meteo[date].index, 
                           columns = [f'g_{i}' for i in range(sensorsy)]+[f'gb_{i}' for i in range(sensorsy)])
    
    print(f'Results file: {results_file}')
    results.to_hdf(project_path/('results/'+results_file), key='df')
    #results.to_csv(project_path/('results/'+results_file+'.csv'))
    print(project_path/('results/'+results_file+'.hdf'))
    return inca

if __name__ == '__main__':
    Fire(run_simulation)