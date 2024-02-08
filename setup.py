from setuptools import setup
import os
import sys
import sysconfig

if __name__=='__main__':
	Packages = ['BAT']
	
	setup(	name='BAT',
		version="0.1",
		packages = Packages,
		author="Talant Ruzmetov" ,
		author_email="talantruzmetov@gmail.com",
		description="This is GPU optimized fylly otomistic `Bond Angle Torsion` for molecular systems (proteins, ligands, protein-ligand, protein-protein .. etc.)",
		license="MIT",
		keywords="protein, ligand, internal coordinates, torch, differentiable,free energy",
		url="https://github.com/truzmeto/BondAngleTorsion",
		project_urls={
			"Bug Tracker": "https://github.com/truzmeto/BondAngleTorsion/issues",
			#"Documentation": "https://github.com/truzmeto/BKit/tree/Release/Doc",
			#"Source Code": "https://github.com/truzmeto/BKit",
		})
