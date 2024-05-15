import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
import torch
import unittest
from BAT import BATmain
from MDAnalysis.analysis.bat import BAT


class TesReconstruction(unittest.TestCase):

    def setUp(self):

        data_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13_1000.dcd'
        prmtop_path = '/home/talant/Projects/UCRiver/DeepLearning/AB13/TRAJ/AB13.prmtop'
        self.device = 'cpu'
        self.u = mda.Universe(prmtop_path, data_path)
        selected = self.u.select_atoms("protein")
        R = BAT(selected)
        n_atoms = R.atoms.n_atoms

        tor_indxs = R._torsion_XYZ_inds
        self.root_XYZ_inds = R._root_XYZ_inds
        self.B = BATmain(n_atoms, tor_indxs, self.root_XYZ_inds , device=self.device)    
  
        
        
    def test(self):
         
        xyz = []
        for i in self.u._trajectory:
            xyz.append(i.positions)
        xyz = np.array(xyz)
        xyz = torch.as_tensor(xyz, device=self.device, dtype=torch.float32)
        root_3_xyz = xyz[:,self.root_XYZ_inds,:]
        
        n1, n2, va, vb = self.B.Coords2MainVecs(xyz)
        bonds, angles, torsions = self.B.BondAngleTorsion(n1, n2, va, vb)
        
        xyz_new = self.B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
        rmsd = (xyz - xyz_new).pow(2).mean().sqrt()
        
        self.assertTrue( rmsd.item() < 0.001 )

if __name__=='__main__':

    unittest.main()
