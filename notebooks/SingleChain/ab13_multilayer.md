## Fully Atomistic Bond Angle Torsion Multilayer

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

from BAT import BATmain, GetFragmentTorInds
from MDAnalysis.analysis.bat import BAT
```

### Load data and construct torsion list with MDAnalysis


```python
data_path = '/data0/talant/AB13/TRAJ/AB13_1000.dcd'
prmtop_path = '/data0/talant/AB13/TRAJ/AB13.prmtop'
device = 'cpu'

u = mda.Universe(prmtop_path,data_path)
selected = u.select_atoms("protein")
R = BAT(selected)

n_atoms = R.atoms.n_atoms 
tor_indxs = GetFragmentTorInds(R, sel_atom='N')
root_XYZ_inds = R._root_XYZ_inds
```

### Look at original torsion list

* All selected atoms ('N') are connected to root based atoms via pseudo bond


```python
t = 0
for (i, j, k, l) in zip(R._ag1, R._ag2, R._ag3, R._ag4):
    if i.name == 'N':
        print(tor_indxs_orig[t], \
              (i.resname, i.name), (j.resname, j.name), \
              (k.resname, k.name), (l.resname, l.name), t)
    t += 1    
```

    [176, 178, 198, 200] ('ARG', 'N') ('ARG', 'CA') ('ARG', 'CB') ('ARG', 'HB2') 20
    [159, 178, 198, 200] ('MET', 'N') ('MET', 'CA') ('MET', 'CB') ('MET', 'HB2') 37
    [144, 178, 198, 200] ('GLU', 'N') ('GLU', 'CA') ('GLU', 'CB') ('GLU', 'HB2') 52
    [133, 178, 198, 200] ('SER', 'N') ('SER', 'CA') ('SER', 'CB') ('SER', 'HB2') 63
    [114, 178, 198, 200] ('LEU', 'N') ('LEU', 'CA') ('LEU', 'CB') ('LEU', 'HB2') 82
    [107, 178, 198, 200] ('GLY', 'N') ('GLY', 'CA') ('GLY', 'C') ('LEU', 'N') 88
    [93, 178, 198, 200] ('THR', 'N') ('THR', 'CA') ('THR', 'CB') ('THR', 'HB') 103
    [81, 178, 198, 200] ('ASP', 'N') ('ASP', 'CA') ('ASP', 'CB') ('ASP', 'HB2') 115
    [61, 178, 198, 200] ('PHE', 'N') ('PHE', 'CA') ('PHE', 'CB') ('PHE', 'HB2') 135
    [37, 178, 198, 200] ('TRP', 'N') ('TRP', 'CA') ('TRP', 'CB') ('TRP', 'HB2') 159
    [26, 178, 198, 200] ('SER', 'N') ('SER', 'CA') ('SER', 'CB') ('SER', 'HB2') 170
    [12, 178, 198, 200] ('PRO', 'N') ('PRO', 'CD') ('PRO', 'CG') ('PRO', 'HG2') 184
    [0, 178, 198, 200] ('ALA', 'N') ('ALA', 'CA') ('ALA', 'CB') ('ALA', 'HB1') 194



```python
t = 0
for (i, j, k, l) in zip(R._ag1, R._ag2, R._ag3, R._ag4):
    if i.name == 'N':
        print(tor_indxs[t], \
              (i.resname, i.name), (j.resname, j.name), \
              (k.resname, k.name), (l.resname, l.name), t)
    t += 1    
```

    [176, 178, 198, 200] ('ARG', 'N') ('ARG', 'CA') ('ARG', 'CB') ('ARG', 'HB2') 20
    [159, 178, 198, 200] ('MET', 'N') ('MET', 'CA') ('MET', 'CB') ('MET', 'HB2') 37
    [144, 178, 198, 200] ('GLU', 'N') ('GLU', 'CA') ('GLU', 'CB') ('GLU', 'HB2') 52
    [133, 178, 198, 200] ('SER', 'N') ('SER', 'CA') ('SER', 'CB') ('SER', 'HB2') 63
    [114, 178, 198, 200] ('LEU', 'N') ('LEU', 'CA') ('LEU', 'CB') ('LEU', 'HB2') 82
    [107, 178, 198, 200] ('GLY', 'N') ('GLY', 'CA') ('GLY', 'C') ('LEU', 'N') 88
    [93, 178, 198, 200] ('THR', 'N') ('THR', 'CA') ('THR', 'CB') ('THR', 'HB') 103
    [81, 178, 198, 200] ('ASP', 'N') ('ASP', 'CA') ('ASP', 'CB') ('ASP', 'HB2') 115
    [61, 178, 198, 200] ('PHE', 'N') ('PHE', 'CA') ('PHE', 'CB') ('PHE', 'HB2') 135
    [37, 178, 198, 200] ('TRP', 'N') ('TRP', 'CA') ('TRP', 'CB') ('TRP', 'HB2') 159
    [26, 178, 198, 200] ('SER', 'N') ('SER', 'CA') ('SER', 'CB') ('SER', 'HB2') 170
    [12, 178, 198, 200] ('PRO', 'N') ('PRO', 'CD') ('PRO', 'CG') ('PRO', 'HG2') 184
    [0, 178, 198, 200] ('ALA', 'N') ('ALA', 'CA') ('ALA', 'CB') ('ALA', 'HB1') 194


### Load trajectory into torch tensor


```python
xyz = []
for i in u._trajectory:
    xyz.append(i.positions)
xyz = np.array(xyz)
xyz = torch.as_tensor(xyz, device=device, dtype=torch.float32)
xyz.shape
```




    torch.Size([1001, 201, 3])



### Calculate BAT from XYZ


```python

```


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

    BAT to Coords reconstruction Err -- 1.0541142728470732e-05


#### Reconstruction is almost 100% accurate!


```python

```

### Write to .dcd if needed


```python
xyz_new = xyz_new.detach().cpu().numpy()
out_path = 'out.dcd'
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
view[1].add_licorice('all', color='orange')   # reconstruction
```


```python
view
```


    NGLWidget(max_frame=1000)



```python

```


```python

```


```python

```
