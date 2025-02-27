from dataclasses import dataclass

'''
to calculate a 2pt correlator we first 
- Read in source operators from an input list 
- Read in sink operators from an input list 
- parse the operator string and force them to be ``QuantumNum`` objects 
- Select the irreps commensurate with the given momentum, P^2

'''

@dataclass 
class Exotraction:
    '''parameters for exotraction
    this interfaces with an input file 
    ''' 



