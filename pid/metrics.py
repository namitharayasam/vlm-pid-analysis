# vlm-pid-analysis/pid/metrics.py

import numpy as np
from scipy.special import rel_entr

def MI(P):
    """Calculates Mutual Information I(X1; X2) for a joint P(X1, X2)."""
    # Ensures P is 2D
    if len(P.shape) > 2:
        P = P.reshape((-1, P.shape[-1])) # Flattens all but the last dimension if needed
        
    margin_1 = P.sum(axis=1)[:, np.newaxis]
    margin_2 = P.sum(axis=0)[np.newaxis, :]
    outer = margin_1 * margin_2
    
    # Avoid log(0) issues
    P = np.maximum(P, 1e-20) 
    outer = np.maximum(outer, 1e-20) 
    
    return np.sum(rel_entr(P, outer))

def CoI(P):
    """Calculates Co-Information CoI(X1, X2; Y) for a joint P(X1, X2, Y)."""
    # I(X1, X2; Y) = I(X1; Y) + I(X2; Y) - I(X1, X2)
    Px1y = P.sum(axis=1) # P(X1, Y)
    Px2y = P.sum(axis=0) # P(X2, Y)
    Px1x2 = P.sum(axis=2) # P(X1, X2)
    
    return MI(Px1y) + MI(Px2y) - MI(Px1x2)

def CI(P, Q):
    """Calculates Common Information CI = I(X1, X2; Y) - I_Q(X1, X2; Y)."""
    # This definition corresponds to Redundancy_Total_Information
    # Total Information T = I(X1, X2; Y)
    # T = R + U1 + U2 + S (Redundancy, Unique 1, Unique 2, Synergy)
    # Your definition of CI is equivalent to Synergy (S) in your code
    # CI = T(P) - T(Q) = I_P(X1, X2; Y) - I_Q(X1, X2; Y)
    
    Total_Info_P = MI(P.transpose([2, 0, 1]).reshape((P.shape[2], -1)))
    Total_Info_Q = MI(Q.transpose([2, 0, 1]).reshape((Q.shape[2], -1)))
    
    return Total_Info_P - Total_Info_Q

def UI(P, cond_id=0):
    """
    Calculates Unique Information U(Xi | Xj, Y) where i is the conditioned variable.
    cond_id=0: U(X1 | X2, Y) (Your unique2, Image)
    cond_id=1: U(X2 | X1, Y) (Your unique1, Text)
    """
    if cond_id == 0:
        # Calculate conditional MI I(X1; Y | X2)
        J = P.sum(axis=(1, 2)) # P(X1)
        s = 0
        for i in range(P.shape[0]):
            p = P[i, :, :] / np.maximum(P[i, :, :].sum(), 1e-20) # P(X2, Y | X1=i)
            s += MI(p) * J[i]
        return s
        
    elif cond_id == 1:
        # Calculate conditional MI I(X2; Y | X1)
        J = P.sum(axis=(0, 2)) # P(X2)
        s = 0
        for i in range(P.shape[1]):
            p = P[:, i, :] / np.maximum(P[:, i, :].sum(), 1e-20) # P(X1, Y | X2=i)
            s += MI(p) * J[i]
        return s
        
    else:
        raise ValueError("cond_id must be 0 or 1")

def get_measure(P, name="ipfp", max_iters=500):
    """
    Calculates the four PID measures using the IPFP algorithm.
    P: joint distribution P(X1, X2, Y).
    """
    from .ipfp import alternating_minimization_ipfp # Relative import
    
    if name == 'ipfp':
        # Q is the closest distribution to P with I(X1, X2 | Y)=0
        Q = alternating_minimization_ipfp(P, max_outer=max_iters)

    # Note: Redundancy is I_Q(X1, X2; Y) = CoI(Q)
    redundancy = CoI(Q)
    
    # Unique: U(X2 | X1, Y) = I(X2; Y | X1)_Q
    unique1 = UI(Q, cond_id=1) 
    
    # Unique: U(X1 | X2, Y) = I(X1; Y | X2)_Q
    unique2 = UI(Q, cond_id=0) 
    
    # Synergy: CI(P, Q) = T(P) - T(Q)
    synergy = CI(P, Q)

    return {'redundancy': redundancy, 'unique1': unique1, 'unique2': unique2, 'synergy': synergy}