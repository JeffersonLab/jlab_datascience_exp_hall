import numpy as np

## Applies the four-vector dot product to all samples.
## Inputs should be (N,4) arrays. 
def four_vector_dot(x: np.ndarray, y: np.ndarray):
    return x[:,0]*y[:,0] - x[:,1]*y[:,1] - x[:,2]*y[:,2] - x[:,3]*y[:,3]

def norm3(pi):
    return np.sqrt(pi[:,1]**2 + pi[:,2]**2 + pi[:,3]**2)[:,np.newaxis]

def dot3(a,b):
    return (a[:,1]*b[:,1] + a[:,2]*b[:,2] + a[:,3]*b[:,3])[:,np.newaxis]

# Calculate alpha from 4-vectors pi_plus and pi_minus
def get_alpha(pi_plus, pi_minus):
    N=len(pi_plus[:,0])
    ez = np.zeros((N,4))
    ez[:,3] = 1
    nnp = pi_plus / norm3(pi_plus)
    nnm = pi_minus / norm3(pi_minus)
    ezdotnnp = dot3(ez, nnp)
    aa = 1/np.sqrt(1 - ezdotnnp**2)
    ba = ezdotnnp*aa
    nnmdotnnp = dot3(nnm,nnp)
    ab = 1/np.sqrt(1 - nnmdotnnp**2)
    bb = -nnmdotnnp*ab
    gamma = -aa*ez + ba*nnp
    beta = ab*nnm + bb*nnp

    delta = gamma[:,[0,2,3,1]] * beta[:,[0,3,1,2]] - gamma[:,[0,3,1,2]] * beta[:,[0,2,3,1]]
    prob = dot3(delta,nnp)
    gammadotbeta = np.arccos(dot3(gamma,beta))
    alpha=np.where(prob>=0,gammadotbeta,2.*np.pi-gammadotbeta)
    return alpha.squeeze()