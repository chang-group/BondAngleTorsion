import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.bat import BAT
import matplotlib.pyplot as plt
import torch
import time


def GetFragmentTorInds(R, sel_atom = 'N'):
    """
    Attach each selected atom to root based atoms via pseudo bond
    by modifying torsion list
    """
    tor_indxs = R._torsion_XYZ_inds
        
    m = 0
    for i in R._ag1:
        if i.name == sel_atom :
            tor_indxs[m][1] = R._root_XYZ_inds[2]
            tor_indxs[m][2] = R._root_XYZ_inds[1]
            tor_indxs[m][3] = R._root_XYZ_inds[0]
        m = m + 1
    
    return tor_indxs



if __name__=='__main__':
         
    
    from BATmain import BATmain
    
    data_path = '/data0/talant/AB13/TRAJ/AB13_1000.dcd'
    prmtop_path = '/data0/talant/AB13/TRAJ/AB13.prmtop'
    device = 'cpu'

    u = mda.Universe(prmtop_path,data_path)
    selected = u.select_atoms("protein")
    R = BAT(selected)
    n_atoms = R.atoms.n_atoms
    
    xyz = []
    for i in u._trajectory:
        xyz.append(i.positions)
    xyz = np.array(xyz)
    xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)

    tor_indxs = GetFragmentTorInds(R)
    root_XYZ_inds = R._root_XYZ_inds
    B = BATmain(n_atoms, tor_indxs, root_XYZ_inds , device=device)    
    root_3_xyz = xyz[:,root_XYZ_inds,:]
    
    n1, n2, va, vb = B.Coords2MainVecs(xyz)
    bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)    
    xyz_new = B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
   
    rmsd = (xyz - xyz_new).pow(2).mean().sqrt()
    print('BAT to Coords reconstruction Err --' , rmsd.item())
 
    xyz_new = xyz_new.detach().cpu().numpy()
    with mda.Writer('../data/out.dcd', n_atoms=u.atoms.n_atoms) as w:
        i = 0
        for ts in u.trajectory:
            ts.positions = xyz_new[i,:,:]
            i = i + 1
            w.write(u.atoms)

    
  
