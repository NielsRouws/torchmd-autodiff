"""
Water box from the torchmd test suite.
"""

import torch
from moleculekit.molecule import Molecule

import torchmd
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann, BOLTZMAN
import os
import numpy as np
from natsort import natsorted
from glob import glob
import parmed


class WaterBox:
    """
    Water box for use in torchmd.
    """
    TESTDIR = os.path.normpath(
        os.path.join(
            os.path.dirname(torchmd.__file__),
            os.pardir, 
            "test-data/waterbox"
        )
    )
    psfFile = os.path.join(TESTDIR, "structure.psf")
    xtcFile = os.path.join(TESTDIR, "output.xtc")
    prmFiles = [os.path.join(TESTDIR, "parameters.prm")]
    
    def __init__(self, nreplicas=1, T=300.0, dtype=torch.double, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.mol = self._init_mol()
        self.forces, self.ff = self._init_forces(self.mol)
        self.system = self._init_system(self.mol, self.forces, nreplicas=nreplicas, T=300.0)

    def _init_mol(self):
        mol = Molecule(self.psfFile)
        mol.read([self.xtcFile])
        mol.dropFrames(keep=0)
        return mol
        
    def _init_system(self, mol, forces, nreplicas, T):
        system = System(mol.numAtoms, nreplicas, self.dtype, self.device)
        system.set_positions(mol.coords)
        system.set_box(mol.box)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=T, replicas=nreplicas))
        return system
    
    def _init_forces(self, mol):
        coords = mol.coords
        coords = coords[:, :, 0].squeeze()
        cutoff = 9.0 #np.min(mol.box) / 2 - 0.01
        switch_dist = 7.5
        rfa = True

        struct = parmed.charmm.CharmmPsfFile(self.psfFile)
        prm = parmed.charmm.CharmmParameterSet(*self.prmFiles)
        prm_org = parmed.charmm.CharmmParameterSet(*self.prmFiles)

        ff = ForceField.create(mol, prm)
        parameters = Parameters(ff, mol, precision=self.dtype, device=self.device)
        forces = Forces(
            parameters, cutoff=cutoff, switch_dist=switch_dist, rfa=rfa,
        )
        return forces, ff
