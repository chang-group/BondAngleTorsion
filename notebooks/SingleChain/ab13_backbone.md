## Backbone Bond Angle Torsion

#### Import Libraries


```python
import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
import matplotlib.pyplot as plt
import torch
import nglview as nv
from nglview import NGLWidget
import warnings
warnings.filterwarnings("ignore")

from BAT import BATmain, bb_tor_inds
from MDAnalysis.analysis.bat import BAT 
```

### Load data and construct torsion list with MDAnalysis


```python
data_path = '/data0/talant/AB13/TRAJ/AB13_1000.dcd'
prmtop_path = '/data0/talant/AB13/TRAJ/AB13.prmtop'
device = 'cpu'

u = mda.Universe(prmtop_path,data_path)
selected = u.select_atoms("name C or name CA or name N")
n_atoms = selected.n_atoms
n_tors = n_atoms - 3   
print('N atoms -- ', n_atoms)
tor_indxs =  bb_tor_inds(n_atoms)
root_XYZ_inds = [0,1,2]
```

    N atoms --  39


### Look at torsion list


```python
for i, v in enumerate(tor_ids):
    if i  < 10:
        print(v)
```

    [3, 2, 1, 0]
    [4, 3, 2, 1]
    [5, 4, 3, 2]
    [6, 5, 4, 3]
    [7, 6, 5, 4]
    [8, 7, 6, 5]
    [9, 8, 7, 6]
    [10, 9, 8, 7]
    [11, 10, 9, 8]
    [12, 11, 10, 9]



```python

```

### Load trajectory into torch tensor


```python
xyz = []
for i in u._trajectory:
    xyz.append(selected.positions)

xyz = np.array(xyz)
xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)
xyz.shape
```




    torch.Size([1001, 39, 3])




```python

```

### Calculate BAT from XYZ


```python
B = BATmain(n_atoms, tor_indxs, root_XYZ_inds , device=device)    
n1, n2, va, vb = B.Coords2MainVecs(xyz)
bonds, angles, torsions = B.BondAngleTorsion(n1, n2, va, vb)
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

    BAT to Coords reconstruction Err -- 2.804330279104761e-06


#### Reconstruction is almost 100% accurate!


```python

```

### Write to .dcd if needed


```python
xyz_new = xyz_new.detach().cpu().numpy()
out_path = './dat/out_bb.dcd'
with mda.Writer(out_path, n_atoms=selected.n_atoms) as w:
        i = 0
        for ts in u.trajectory:
            selected.positions = xyz_new[i,:,:]
            i = i + 1
            w.write(selected)

```

## Use VMD to visualize


```python

```


```python

```


```python

```
