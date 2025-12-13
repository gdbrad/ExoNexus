import numpy as np 
#import scipy as scp
import pprint

sigma: dict[int, np.ndarray] = {}
gamma: dict[int, np.ndarray] = {}
I = np.array([[1, 0],
              [0, 1]])
IDEN = np.identity(4)
sigma[1] = np.array([[0, 1],
                    [1, 0]])
sigma[2] = np.array([[0, -1j],
                    [1j, 0]])
sigma[3] = np.array([[1,  0],
                    [0, -1]])

gamma[1] = np.zeros((4, 4), dtype=np.cdouble)
gamma[1][:2, 2:] = 1j * sigma[1]
gamma[1][2:, :2] = -1j * sigma[1]

gamma[2] = np.zeros((4, 4), dtype=np.cdouble)
gamma[2][:2, 2:] = -1j * sigma[2]
gamma[2][2:, :2] = 1j * sigma[2]

gamma[3] = np.zeros((4, 4), dtype=np.cdouble)
gamma[3][:2, 2:] = 1j * sigma[3]
gamma[3][2:, :2] = -1j * sigma[3]

gamma[4] = np.zeros((4, 4), dtype=np.cdouble)
gamma[4][:2, 2:] = I
gamma[4][2:, :2] = I

gamma[5] = gamma[1] @ gamma[2] @ gamma[3] @ gamma[4]
gamma_i = [gamma[1],gamma[2],gamma[3],gamma[4]]

# TODO when are these necessary to use? for the deriv operators? 
#I_sparse = scp.sparse.csr_matrix(I)
# g5_block = scp.sparse.kron(I,gamma.gamma[5],format='csr')
# g5_sparse = scp.sparse.csr_matrix(gamma.gamma[5])
# g4_block = scp.sparse.kron(I,gamma.gamma[4],format='csr')
# g4_sparse = scp.sparse.csr_matrix(gamma.gamma[4])
# g3_block = scp.sparse.kron(I,gamma.gamma[3],format='csr')
# g3_sparse = scp.sparse.csr_matrix(gamma.gamma[3])
# g2_block = scp.sparse.kron(I,gamma.gamma[2],format='csr')
# g2_sparse = scp.sparse.csr_matrix(gamma.gamma[2])
# g1_block = scp.sparse.kron(I,gamma.gamma[1],format='csr')
# g1_sparse = scp.sparse.csr_matrix(gamma.gamma[1])

#to be looped over based on value of i index in derivative 
# gamma_i_block = [g1_block,g2_block,g3_block]
# gamma_i_sparse = [g1_sparse,g2_sparse,g3_sparse]


