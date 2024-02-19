#!/usr/bin/env python
import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
import matplotlib.pyplot as plt
import torch

import warnings
import time
import os, sys
warnings.filterwarnings("ignore")

from BAT import BATmain
from MDAnalysis.analysis.bat import BAT


def load_data(dat_path, prmtop_path, out_path, t_id, sel_string):
    """
    
    """
    u = mda.Universe(prmtop_path, dat_path)
    sel = u.select_atoms(sel_string)
    file = out_path + 'tors' + str(t_id) +'.npy'
    
    if os.path.exists(file):
        d = np.load(file, allow_pickle=True)
    else:
        R = BAT(sel)
        d = {}
        d['n_atoms'] = R.atoms.n_atoms
        d['root'] = R._root_XYZ_inds
        d['tor_ids'] = R._torsion_XYZ_inds
        np.save(file, d)
    return d, u, sel



def GetXYZ(u, selected, fr=10000, device='cpu'):
    xyz = []
    for i in u._trajectory:
        xyz.append(selected.positions)
    xyz = np.array(xyz)
    xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)
    return xyz[0:fr]







def t_coords2BAT(xyz ,n_atoms, tor_indxs, root_XYZ_inds , device='cpu'):
    """ 
    
    """
    start = time.time()
    #######  
    B = BATmain(n_atoms, tor_indxs, root_XYZ_inds , device=device)    
    n1, n2, va, vb = B.Coords2MainVecs(xyz)
    bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)
    #######
    end = time.time()
    t = end - start

    return B, bonds, angles, torsions, t, n_atoms
    

def t_BAT2coords(B, bonds, angles, torsions,root_XYZ_inds):
    """
    
    """
    start = time.time()
    #######  
    root_3_xyz = xyz[:,root_XYZ_inds,:]
    xyz_new = B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
    #######
    end = time.time()
    t = end - start

    return t 




if __name__=='__main__':



    dat_list = ['/data4/talant/AB13/TRAJ/AB13.dcd',
                '/data4/talant/Abeta/TRAJ/idps/2nao_mono_1.dcd',            
                '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/MT11_r1_500ns_noZN.dcd',
                '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/MT11_r1_500ns_noZN.dcd',
                '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/MT11_r1_500ns_noZN.dcd']

    prmtop_list = ['/data4/talant/AB13/TRAJ/AB13.prmtop',
                   '/data4/talant/Abeta/TRAJ/idps/top.prmtop',
                   '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/protein_noZN.prmtop',
                   '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/protein_noZN.prmtop',
                   '/data3/talant/DeepLearning/data/Ubi_MT/PC_motions/protein_noZN.prmtop']

    sel_st_list = ["protein", "protein", "resid 320:397", "resid 1:150", "resid 1:319"]
    out_path = './out/'
    

    n_traj = len(dat_list)
    d_list = []
    sel_list = []
    u_list = []

    for i in range(n_traj):
        d, u, sel= load_data(dat_list[i], prmtop_list[i], out_path,
                             t_id=i, sel_string=sel_st_list[i])
        d_list.append(d.item())
        u_list.append(u)
        sel_list.append(sel)


    n_list = [i["n_atoms"] for i in d_list]
    tor_list = [i["tor_ids"] for i in d_list]
    root_list = [i["root"] for i in d_list]device = 'cpu'
    time_forw_cpu = []
    time_back_cpu = []
    
    for i in range(n_traj):
        xyz = GetXYZ(u_list[i], sel_list[i], device=device)
        B, bonds, angles, torsions, tf, n_atoms = t_coords2BAT(xyz ,n_list[i], tor_list[i], \
                                                               root_list[i] , device=device)
        tb = t_BAT2coords(B, bonds, angles, torsions, root_list[i])
        
        time_forw_cpu.append(tf)
        time_back_cpu.append(tb)    


    device = 'cuda'
    time_forw_gpu = []
    time_back_gpu = []
    for i in range(5):
        xyz = GetXYZ(u_list[i], sel_list[i], device=device)
        B, bonds, angles, torsions, tf, n_atoms = t_coords2BAT(xyz ,n_list[i], tor_list[i], \
                                                               root_list[i] , device=device)
        tb = t_BAT2coords(B, bonds, angles, torsions, root_list[i])
        
        time_forw_gpu.append(tf)
        time_back_gpu.append(tb)


    n_list = np.array(n_list)
    n_list1 = ["0.2K","0.6K", "1.3K", "2.4K", "5.0K"]
    time_forw_cpu = np.array(time_forw_cpu) # msec
    time_forw_gpu = np.array(time_forw_gpu) # msec
    time_back_cpu = np.array(time_back_cpu) # msec
    time_back_gpu = np.array(time_back_gpu) # msec
    
    
    bg_cl = '#1b212c'#'#1A1A1A'#'black'
    ts = 12
    text_size = ts
    tcl = 'cyan'
    
    system = ['AB13', 'AB42', 'Ubiquitin', 'PLPro150', 'PLPro320']
    fig = plt.figure(figsize=[10,4], facecolor=bg_cl)
    plt.rcParams['axes.facecolor'] = bg_cl
    
    ax1 = plt.subplot(1, 2, 1)
    plt.bar(x = n_list1, height = time_forw_cpu, width=-0.27, align='edge')
    plt.bar(x = n_list1, height = time_back_cpu, width=0.27, align='edge')
    plt.tick_params(colors=tcl)
    
    plt.xlabel('System size ', size=text_size, color=tcl)
    plt.ylabel('Compute time (sec)', size=text_size, color=tcl)
    plt.legend(['XYZ --> BAT', 'BAT --> XYZ'], fontsize=text_size,labelcolor=tcl, loc='upper left')
    plt.title('CPU', size = text_size+2, color=tcl)
    plt.ylim(0,8)

    for t in range(len(system)):
        plt.text(t*0.9, time_back_cpu[t]+0.5, s=system[t], size=text_size-1, color=tcl)


    ax2 = plt.subplot(1, 2, 2)
    plt.bar(x = n_list1, height = time_forw_gpu, width=-0.27, align='edge')
    plt.bar(x = n_list1, height = time_back_gpu, width=0.27, align='edge')
    plt.tick_params(colors=tcl)
    plt.xlabel('System size ', size=text_size, color=tcl)
    plt.title('GPU', size = text_size+2, color=tcl)
    plt.ylim(0,2)

    for t in range(len(system)):
        plt.text(t*0.9, time_back_gpu[t]+0.2, s=system[t], size=text_size-1, color=tcl)
        
    fig.tight_layout()
    plt.savefig('Benchmark.png', dpi=500)
    plt.show()



