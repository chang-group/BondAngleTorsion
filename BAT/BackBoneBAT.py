import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.bat import BAT
import matplotlib.pyplot as plt
import torch
import time

   
def bb_tor_inds(n_atoms):
    """
    This doesn't follow original atom indecies
 
    """
    
    tor_ids = []    
    for i in range(3, n_atoms):
        tor_ids.append([i, i-1, i-2, i-3])
   
    return tor_ids


if __name__=='__main__':
     
    from BATmain import BATmain
    
    #data_path = '/data0/talant/AB13/TRAJ/AB13_1000.dcd'
    #prmtop_path = '/data0/talant/AB13/TRAJ/AB13.prmtop'
    data_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13_1000.dcd'
    prmtop_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13.prmtop'

    
    device = 'cpu'
    u = mda.Universe(prmtop_path,data_path)
    selected = u.select_atoms("name C or name CA or name N")
 
    n_atoms = selected.n_atoms
    n_tors = n_atoms - 3   
    print('N atoms -- ', n_atoms)
    
    xyz = []
    for i in u._trajectory:
        xyz.append(selected.positions)
    xyz = np.array(xyz)
    bb_xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)

    tor_ids =  bb_tor_inds(n_atoms)
    root_XYZ_inds = [0,1,2]
    
    B = BATmain(n_atoms, tor_ids, root_XYZ_inds , device=device)    
    n1, n2, va, vb  = B.Coords2MainVecs(bb_xyz)
    bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)

       
    root_3_xyz = bb_xyz[:, root_XYZ_inds,:]
    
    bb_xyz_new = B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
    rmsd = (bb_xyz - bb_xyz_new).pow(2).mean().sqrt()
    print('BAT to Coords reconstruction Err --' , rmsd.item())   
  
    
    bb_xyz_new = bb_xyz_new.detach().cpu().numpy()
    with mda.Writer('../data/out_bb.pdb', n_atoms=selected.n_atoms) as w:
        i = 0
        for ts in u.trajectory:
            selected.positions = bb_xyz_new[i,:,:]
            i = i + 1
            w.write(selected)
