## Multilayer BAT example using Plpro-Ubiqutin complex

1. In order to construct continuous torsion list we first join $\rm{chain}_1$ and $\rm{chain}_2$ with pseudobond.
2. Pseudobond connects c-terminal carbon atom of $\rm{chain}_1$ with n-terminal nitrogen atom of $\rm{chain}_2$.
3. Finally we connect selected atom of each amino acid (or fragment head) to last root based atom via pseudobonds. Length of these pseudobonds would be a column of all atom distance matrix (N-3)x(N-3)


```python
import MDAnalysis as mda
import numpy as np
import torch
from MDAnalysis import Universe
from MDAnalysis.analysis.bat import BAT
import matplotlib.pyplot as plt
import pickle
import sys
import os

import nglview as nv
from nglview import NGLWidget
import warnings
warnings.filterwarnings("ignore")

from BAT import BATmain, GetFragmentTorInds
```

### Load Data


```python
data_path = './dat/plpro_ubi.dcd'
prmtop_path = './dat/plpro_ubi.prmtop'
u = mda.Universe(prmtop_path,data_path)
selected = u.select_atoms("protein")
```


```python
u.bonds
```




    <TopologyGroup containing 6291 bonds>




```python
selected.fragments
```




    (<AtomGroup with 4974 atoms>, <AtomGroup with 1258 atoms>)



### Look at atoms of the complex to indentify atoms to bond


```python
for i in selected:
    if (i.resid > 318) & (i.resid < 321):
        print(i.index, i.name, i.resid, i.resname)
```

    4961 N 319 ASP
    4962 H 319 ASP
    4963 CA 319 ASP
    4964 HA 319 ASP
    4965 CB 319 ASP
    4966 HB2 319 ASP
    4967 HB3 319 ASP
    4968 CG 319 ASP
    4969 OD1 319 ASP
    4970 OD2 319 ASP
    4971 C 319 ASP
    4972 O 319 ASP
    4973 OXT 319 ASP
    4974 N 320 MET
    4975 H1 320 MET
    4976 H2 320 MET
    4977 H3 320 MET
    4978 CA 320 MET
    4979 HA 320 MET
    4980 CB 320 MET
    4981 HB2 320 MET
    4982 HB3 320 MET
    4983 CG 320 MET
    4984 HG2 320 MET
    4985 HG3 320 MET
    4986 SD 320 MET
    4987 CE 320 MET
    4988 HE1 320 MET
    4989 HE2 320 MET
    4990 HE3 320 MET
    4991 C 320 MET
    4992 O 320 MET


### Subset plpro and ubi


```python
plpro = u.select_atoms('resid 1:319')
ubi = u.select_atoms('resid 320:397')
```

### Add new bond indecies to a overall list & add Top attribute

We simply concatenate plpro, new (C-N), and ubiqutin bond indecies into long numpy array and add it to topology.


```python
b_new_id = np.array([4971, 4974]).reshape(1,2) #this is where pseudo bond is placed
new_bond_inds = np.concatenate([plpro.bonds.indices, b_new_id, ubi.bonds.indices], axis=0) 
```


```python
#u.add_TopologyAttr('bonds', new_bond_inds) # old
u.add_bonds(new_bond_inds)
```


```python
print('added bond index -- ', selected.bonds.values().argmax())
print('added bond length -- ', selected.bonds.values().max())
```

    added bond index --  5026
    added bond length --  37.646184430226455



```python
u.bonds
```




    <TopologyGroup containing 6292 bonds>




```python
selected.fragments
```




    (<AtomGroup with 6232 atoms>,)



After adding a new pseudobond, we have 6292 bonds and only one fragment!


```python

```

### Generate new torsion list using BAT, or load from file if saved previously. 


```python
out_path = './output/'
if os.path.exists(out_path + 'R.pickle'):
    with open(out_path + 'R.pickle', 'rb') as file:
        u, R = pickle.load(file)    
else:
    R = BAT(selected)
    with open(out_path + 'R.pickle', 'wb') as file:
        pickle.dump((u,R), file) 
```


```python

```

### Check new torsion identities 


```python
m = 0
id1 = 4971; id2 = 4974
for (i, j, k, l) in zip(R._ag1, R._ag2, R._ag3, R._ag4):
    if (i.resid > 318) & (i.resid < 321):
        print((i.resid, i.resname, i.id-1, i.name), \
              (j.resid, j.resname, j.id-1, j.name), \
              (k.resid, k.resname, k.id-1, k.name), \
              (l.resid, l.resname, l.id-1, l.name))
    m = m + 1
```

    (319, 'ASP', 4961, 'N') (318, 'SER', 4959, 'C') (318, 'SER', 4952, 'CA') (318, 'SER', 4953, 'HA')
    (319, 'ASP', 4962, 'H') (319, 'ASP', 4961, 'N') (318, 'SER', 4959, 'C') (318, 'SER', 4952, 'CA')
    (319, 'ASP', 4963, 'CA') (319, 'ASP', 4961, 'N') (318, 'SER', 4959, 'C') (318, 'SER', 4952, 'CA')
    (319, 'ASP', 4964, 'HA') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4961, 'N') (319, 'ASP', 4962, 'H')
    (319, 'ASP', 4965, 'CB') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4961, 'N') (319, 'ASP', 4962, 'H')
    (319, 'ASP', 4966, 'HB2') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (319, 'ASP', 4967, 'HB3') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (319, 'ASP', 4968, 'CG') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (319, 'ASP', 4969, 'OD1') (319, 'ASP', 4968, 'CG') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4966, 'HB2')
    (319, 'ASP', 4970, 'OD2') (319, 'ASP', 4968, 'CG') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4966, 'HB2')
    (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4965, 'CB') (319, 'ASP', 4966, 'HB2')
    (320, 'MET', 4974, 'N') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (319, 'ASP', 4972, 'O') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (319, 'ASP', 4973, 'OXT') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA') (319, 'ASP', 4964, 'HA')
    (320, 'MET', 4975, 'H1') (320, 'MET', 4974, 'N') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA')
    (320, 'MET', 4976, 'H2') (320, 'MET', 4974, 'N') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA')
    (320, 'MET', 4977, 'H3') (320, 'MET', 4974, 'N') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA')
    (320, 'MET', 4978, 'CA') (320, 'MET', 4974, 'N') (319, 'ASP', 4971, 'C') (319, 'ASP', 4963, 'CA')
    (320, 'MET', 4979, 'HA') (320, 'MET', 4978, 'CA') (320, 'MET', 4974, 'N') (320, 'MET', 4975, 'H1')
    (320, 'MET', 4980, 'CB') (320, 'MET', 4978, 'CA') (320, 'MET', 4974, 'N') (320, 'MET', 4975, 'H1')
    (320, 'MET', 4981, 'HB2') (320, 'MET', 4980, 'CB') (320, 'MET', 4978, 'CA') (320, 'MET', 4979, 'HA')
    (320, 'MET', 4982, 'HB3') (320, 'MET', 4980, 'CB') (320, 'MET', 4978, 'CA') (320, 'MET', 4979, 'HA')
    (320, 'MET', 4983, 'CG') (320, 'MET', 4980, 'CB') (320, 'MET', 4978, 'CA') (320, 'MET', 4979, 'HA')
    (320, 'MET', 4984, 'HG2') (320, 'MET', 4983, 'CG') (320, 'MET', 4980, 'CB') (320, 'MET', 4981, 'HB2')
    (320, 'MET', 4985, 'HG3') (320, 'MET', 4983, 'CG') (320, 'MET', 4980, 'CB') (320, 'MET', 4981, 'HB2')
    (320, 'MET', 4986, 'SD') (320, 'MET', 4983, 'CG') (320, 'MET', 4980, 'CB') (320, 'MET', 4981, 'HB2')
    (320, 'MET', 4987, 'CE') (320, 'MET', 4986, 'SD') (320, 'MET', 4983, 'CG') (320, 'MET', 4984, 'HG2')
    (320, 'MET', 4988, 'HE1') (320, 'MET', 4987, 'CE') (320, 'MET', 4986, 'SD') (320, 'MET', 4983, 'CG')
    (320, 'MET', 4989, 'HE2') (320, 'MET', 4987, 'CE') (320, 'MET', 4986, 'SD') (320, 'MET', 4983, 'CG')
    (320, 'MET', 4990, 'HE3') (320, 'MET', 4987, 'CE') (320, 'MET', 4986, 'SD') (320, 'MET', 4983, 'CG')
    (320, 'MET', 4991, 'C') (320, 'MET', 4978, 'CA') (320, 'MET', 4980, 'CB') (320, 'MET', 4981, 'HB2')
    (320, 'MET', 4992, 'O') (320, 'MET', 4991, 'C') (320, 'MET', 4978, 'CA') (320, 'MET', 4979, 'HA')


### Load trajectory into torch tensor


```python
device = 'cpu'
n_atoms = R.atoms.n_atoms 
tor_indxs = GetFragmentTorInds(R, sel_atom='N')
root_XYZ_inds = R._root_XYZ_inds
```


```python
xyz = []
for i in u._trajectory:
    xyz.append(i.positions)
xyz = np.array(xyz)
xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)
xyz.shape
```




    torch.Size([21, 6232, 3])



### Calculate BAT from XYZ


```python
B = BATmain(n_atoms, tor_indxs, root_XYZ_inds , device=device)    
n1, n2, va, vb = B.Coords2MainVecs(xyz)
bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)
```


```python

```

### Transform BAT back to XYZ


```python
root_3_xyz = xyz[:,root_XYZ_inds,:]
xyz_new = B.BatV2Coords(bonds, angles, torsions, root_3_xyz)
```


```python
rmsd = (xyz - xyz_new).pow(2).mean().sqrt()
print('BAT to Coords reconstruction Err --' , rmsd.item())
```

    BAT to Coords reconstruction Err -- 0.8757097125053406


There is small error due to pseudobond! 

### Write to .dcd if needed


```python
xyz_new = xyz_new.detach().cpu().numpy()
out_path = './output/out.dcd'
with mda.Writer(out_path, n_atoms=u.atoms.n_atoms) as w:
    i = 0
    for ts in u.trajectory:
        ts.positions = xyz_new[i,:,:]
        i = i + 1
        w.write(u.atoms)
```

## Load reconstructed traj and visualize togather


```python
u_new = mda.Universe(prmtop_path, out_path)
```


```python
view = NGLWidget()
view.add_trajectory(u)
view.add_trajectory(u_new)
view[0].clear(); view[1].clear() 
view[0].add_cartoon('all', color='cyan') # original data
view[1].add_cartoon('all', color='orange')   # reconstruction
```


```python
view
```


    NGLWidget(max_frame=20)



```python

```


```python

```


```python

```
