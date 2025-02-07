from dataclasses import dataclass
import numpy as np 
import itertools

class Oh: 
    def __init__(self):
        pass

    




@dataclass
class IrrepNames:
    '''Oh and LG'''
    name: str
    wp: str
    np: str
    ferm: bool  # Double cover
    lg: bool
    dim: int
    G: int

irrep_names_no_par = {
    "A1": IrrepNames(name="A1", wp="A1", np="A1", ferm=False, lg=False, dim=1, G=0),
    "A2": IrrepNames(name="A2", wp="A2", np="A2", ferm=False, lg=False, dim=1, G=0),
    "T1": IrrepNames(name="T1", wp="T1", np="T1", ferm=False, lg=False, dim=3, G=0),
    "T2": IrrepNames(name="T2", wp="T2", np="T2", ferm=False, lg=False, dim=3, G=0),
    "E": IrrepNames(name="E", wp="E", np="E", ferm=False, lg=False, dim=2, G=0),
    "G1": IrrepNames(name="G1", wp="G1", np="G1", ferm=True, lg=False, dim=2, G=0),
    "G2": IrrepNames(name="G2", wp="G2", np="G2", ferm=True, lg=False, dim=2, G=0),
    "H": IrrepNames(name="H", wp="H", np="H", ferm=True, lg=False, dim=4, G=0),
    "D4A1": IrrepNames(name="D4A1", wp="D4A1", np="D4A1", ferm=False, lg=True, dim=1, G=0),
    "D4A2": IrrepNames(name="D4A2", wp="D4A2", np="D4A2", ferm=False, lg=True, dim=1, G=0),
    "D4E1": IrrepNames(name="D4E1", wp="D4E1", np="D4E1", ferm=True, lg=True, dim=2, G=0),
    "D4E2": IrrepNames(name="D4E2", wp="D4E2", np="D4E2", ferm=False, lg=True, dim=2, G=0),
    "D4E3": IrrepNames(name="D4E3", wp="D4E3", np="D4E3", ferm=True, lg=True, dim=2, G=0),
    "D4B1": IrrepNames(name="D4B1", wp="D4B1", np="D4B1", ferm=False, lg=True, dim=1, G=0),
    "D4B2": IrrepNames(name="D4B2", wp="D4B2", np="D4B2", ferm=False, lg=True, dim=1, G=0),
    "D3A1": IrrepNames(name="D3A1", wp="D3A1", np="D3A1", ferm=False, lg=True, dim=1, G=0),
    "D3A2": IrrepNames(name="D3A2", wp="D3A2", np="D3A2", ferm=False, lg=True, dim=1, G=0),
    "D3E1": IrrepNames(name="D3E1", wp="D3E1", np="D3E1", ferm=True, lg=True, dim=2, G=0),
    "D3E2": IrrepNames(name="D3E2", wp="D3E2", np="D3E2", ferm=False, lg=True, dim=2, G=0),
    "D3B1": IrrepNames(name="D3B1", wp="D3B1", np="D3B1", ferm=False, lg=True, dim=1, G=0),
    "D3B2": IrrepNames(name="D3B2", wp="D3B2", np="D3B2", ferm=False, lg=True, dim=1, G=0),
    "D2A1": IrrepNames(name="D2A1", wp="D2A1", np="D2A1", ferm=False, lg=True, dim=1, G=0),
    "D2A2": IrrepNames(name="D2A2", wp="D2A2", np="D2A2", ferm=False, lg=True, dim=1, G=0),
    "D2E": IrrepNames(name="D2E", wp="D2E", np="D2E", ferm=True, lg=True, dim=2, G=0),
    "D2B1": IrrepNames(name="D2B1", wp="D2B1", np="D2B1", ferm=False, lg=True, dim=1, G=0),
    "D2B2": IrrepNames(name="D2B2", wp="D2B2", np="D2B2", ferm=False, lg=True, dim=1, G=0),
    "C4nm0A1": IrrepNames(name="C4nm0A1", wp="C4nm0A1", np="C4nm0A1", ferm=False, lg=True, dim=1, G=0),
    "C4nm0A2": IrrepNames(name="C4nm0A2", wp="C4nm0A2", np="C4nm0A2", ferm=False, lg=True, dim=1, G=0),
    "C4nm0E": IrrepNames(name="C4nm0E", wp="C4nm0E", np="C4nm0E", ferm=True, lg=True, dim=2, G=0),
    "C4nnmA1": IrrepNames(name="C4nnmA1", wp="C4nnmA1", np="C4nnmA1", ferm=False, lg=True, dim=1, G=0),
    "C4nnmA2": IrrepNames(name="C4nnmA2", wp="C4nnmA2", np="C4nnmA2", ferm=False, lg=True, dim=1, G=0),
    "C4nnmE": IrrepNames(name="C4nnmE", wp="C4nnmE", np="C4nnmE", ferm=True, lg=True, dim=2, G=0),
}

