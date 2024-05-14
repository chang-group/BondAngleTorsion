import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.bat import BAT
import matplotlib.pyplot as plt
import torch
import time


class BATmain:
   
    def __init__(self, n_atoms, tor_indxs, root_XYZ_inds, device='cpu'):
        """
        

        Parameters
        ----------
        n_atoms : int
            total number of atoms
        tor_indxs : list of ints (n_torsions, 4)
            Torsion list of ints - defines all four body connections  
        root_XYZ_inds : list of ints
            three selected atoms - root based atoms. 
        device : string, optional # 'cuda' or 'cpu'
            Device type to performe calculation.The default is 'cpu'.

        Returns
        -------
        None.

        """
        
        self.tor_indxs = torch.tensor(tor_indxs,
                                      dtype=torch.long,
                                      device=device)
        self.device = device        
        self.n_tors = n_atoms - 3
        self.root_XYZ_inds = root_XYZ_inds        

   
 
    def Coords2MainVecs(self, xyz):
        """
        is differentiable -- Yes
        
        """
        
        
        #get all positions for torsion calc
        p1 = xyz[:, self.tor_indxs[:,0]]
        p2 = xyz[:, self.tor_indxs[:,1]]
        p3 = xyz[:, self.tor_indxs[:,2]]
        p4 = xyz[:, self.tor_indxs[:,3]]
        
        # get all bond vecs for all bat
        va = p2 - p1
        vb = p3 - p2 # middle bond vector
        vc = p4 - p3

        n1 = torch.cross(-va, vb)  # n1 is normal vector to -va, vb
        n2 = torch.cross(-vb, vc)  # n2 is normal vector to -vb, vc

        return n1, n2, va, vb
    
    def BondAngleTorsion(self, n1, n2, va, vb):
        """
        is differentiable -- Yes
        
        """
        
        #get bond length
        bonds = va.pow(2).sum(dim=-1).sqrt()
        
        #get bond angles
        x1 = (-va*vb).sum(dim=-1)
        y1 = torch.cross(va, vb).pow(2).sum(axis=-1).sqrt()
        angles = torch.atan2(y1,x1)
        
        #get torsions
        xp = torch.cross(n1, n2)
        x2 = (n1 * n2).sum(dim=-1)
        y2  = (xp * vb).sum(dim=-1) / (vb*vb).sum(dim=-1).sqrt()
        torsions = torch.atan2(y2,x2)
    
        return bonds, angles, torsions


    def BatV2Coords(self, bonds, angles, torsions, root_3_xyz):
        """
    
        """          
        
        n_frames = torsions.shape[0]
        XYZ = torch.zeros(n_frames, self.n_tors + 3, 3, device=self.device)
        XYZ[:,self.root_XYZ_inds,:] = root_3_xyz
        
        sn_ang = torch.sin(angles)  
        cs_ang = torch.cos(angles)
        sn_tor = torch.sin(torsions)
        cs_tor = torch.cos(torsions)
                  
        i = 0
        for (a0,a1,a2,a3) in self.tor_indxs:
            
            pos1 = XYZ[:,a1,:]
            pos2 = XYZ[:,a2,:]
            pos3 = XYZ[:,a3,:]
        
            v21 = (pos1 - pos2) 
            v21 /= v21.pow(2).sum(dim=-1, keepdims=True).sqrt()
            v32 = (pos2 - pos3)
            v32 /= v32.pow(2).sum(dim=-1, keepdims=True).sqrt()
        
            vp = torch.cross(v32, v21)
            cs = (v21 * v32).sum(dim=-1, keepdims=True)
            
            sn = (1.0 - cs * cs ).sqrt()
            vp = vp / sn
            vu = torch.cross(vp, v21)

            XYZ[:,a0,:] = pos1 + bonds[:,i:i+1]*(vu*sn_ang[:,i:i+1]*cs_tor[:,i:i+1] + \
                          vp*sn_ang[:,i:i+1]*sn_tor[:,i:i+1] - v21*cs_ang[:,i:i+1])
          
            i = i + 1
                  
        return XYZ


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



def GetFragmentTorIndsSS(R, resid_list, sel_atom = 'N' ):
    """
    Attach each selected atom to root based atoms via pseudo bond
    by modifying torsion list
    """
    tor_indxs = R._torsion_XYZ_inds
        
    m = 0
    for i in R._ag1:
        if (i.name == sel_atom) & ( i.resid in resid_list):
            tor_indxs[m][1] = R._root_XYZ_inds[2]
            tor_indxs[m][2] = R._root_XYZ_inds[1]
            tor_indxs[m][3] = R._root_XYZ_inds[0]
        m = m + 1
        
    return tor_indxs



def bb_tor_inds(n_atoms):
    """
    This doesn't follow original atom indecies
 
    """
    
    tor_ids = []    
    for i in range(3, n_atoms):
        tor_ids.append([i, i-1, i-2, i-3])
   
    return tor_ids



if __name__=='__main__':
         
    
    #data_path = '/data0/talant/AB13/TRAJ/AB13_1000.dcd'
    data_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13_1000.dcd'
    #prmtop_path = '/data0/talant/AB13/TRAJ/AB13.prmtop'
    prmtop_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13.prmtop'

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

    tor_indxs = R._torsion_XYZ_inds
    root_XYZ_inds = R._root_XYZ_inds

    B = BATmain(n_atoms, tor_indxs, root_XYZ_inds , device=device)    
    root_3_xyz = xyz[:,root_XYZ_inds,:]
       
    n1, n2, va, vb = B.Coords2MainVecs(xyz)
    bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)
       
    xyz_new = B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
    rmsd = (xyz - xyz_new).pow(2).mean().sqrt()
    print('BAT to Coords reconstruction Err --' , rmsd.item())
   

    xyz_new = xyz_new.detach().cpu().numpy()
    with mda.Writer('./out.dcd', n_atoms=u.atoms.n_atoms) as w:
        i = 0
        for ts in u.trajectory:
            ts.positions = xyz_new[i,:,:]
            i = i + 1
            w.write(u.atoms)

    
  
