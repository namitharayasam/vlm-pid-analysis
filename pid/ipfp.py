# vlm-pid-analysis/pid/ipfp.py

import numpy as np
from scipy.special import logsumexp

def alternating_minimization_ipfp(P, rng_seed=42, max_outer=50, max_sink=100,
                                  tol_outer=1e-8, tol_sink=1e-8, eps=1e-20, verbose=False):
    """
    Alternating minimization algorithm for finding the minimal I(X1, X2 | Y) 
    distribution Q that matches the marginals P(X1, Y) and P(X2, Y).
    """
    np.random.seed(rng_seed)
    Px1y = P.sum(axis=1)
    Px2y = P.sum(axis=0)
    py = Px1y.sum(axis=0)

    if not np.allclose(py, Px2y.sum(axis=0), atol=1e-8):
        raise ValueError("Marginals Px1y and Px2y are not consistent with py.")

    m, k = Px1y.shape
    n, _ = Px2y.shape

    # Initialize Q as the product of marginals P(X1|Y) * P(X2|Y) * P(Y)
    Q = np.zeros((m, n, k))
    for y in range(k):
        if py[y] > eps:
            Q[:, :, y] = np.outer(Px1y[:, y], Px2y[:, y]) / py[y]
    
    Q = np.maximum(Q, eps)
    Q /= Q.sum()
    prev_obj = np.inf

    for t in range(max_outer):
        # Step 1: Update A (Marginal P(X1, X2))
        Q_marg = Q.sum(axis=2)
        A = Q_marg / k
        A_log = np.log(np.maximum(A, eps))
        Q_new = np.zeros_like(Q)

        # Step 2: Sinkhorn-Knopp for each slice P(X1, X2 | Y=y)
        for y in range(k):
            if py[y] < eps:
                continue
            
            log_r = np.log(np.maximum(Px1y[:, y], eps))
            log_c = np.log(np.maximum(Px2y[:, y], eps))
            log_v = np.zeros(n)
            
            for s in range(max_sink):
                log_u = log_r - logsumexp(A_log + log_v[np.newaxis, :], axis=1)
                log_v_new = log_c - logsumexp(A_log + log_u[:, np.newaxis], axis=0)

                if s > 0 and np.max(np.abs(log_v_new - log_v)) < tol_sink:
                    break
                log_v = log_v_new
                
            Q_new[:, :, y] = np.exp(A_log + log_u[:, np.newaxis] + log_v[np.newaxis, :])
        
        Q_new = np.maximum(Q_new, eps)
        Q_new /= Q_new.sum()

        # Step 3: Check convergence via objective function
        Q_marg_new = Q_new.sum(axis=2)
        Q_tilde = Q_marg_new[:, :, None] / k
        obj = np.sum(Q_new * (np.log(Q_new + eps) - np.log(Q_tilde + eps)))

        if verbose:
            print(f"Iteration {t+1}, objective = {obj:.6e}")

        if t > 0 and np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < tol_outer:
            break

        Q = Q_new
        prev_obj = obj

    return Q

def extract_categorical_from_data(x):
    """Converts a list of raw data points into discrete indices."""
    supp = set(x)
    raw_to_discrete = dict()
    for i in supp:
        raw_to_discrete[i] = len(raw_to_discrete)
    discrete_data = [raw_to_discrete[x_] for x_ in x]
    return discrete_data, raw_to_discrete

def convert_data_to_distribution(x1, x2, y):
    """Converts discrete data arrays (X1, X2, Y) into a joint distribution P(X1, X2, Y)."""
    # Flattens input arrays if they have an extra dimension (e.g., from reshape(-1, 1))
    x1 = x1.squeeze()
    x2 = x2.squeeze()
    y = y.squeeze()

    assert x1.size == x2.size
    assert x1.size == y.size
    numel = x1.size

    x1_discrete, x1_raw_to_discrete = extract_categorical_from_data(x1)
    x2_discrete, x2_raw_to_discrete = extract_categorical_from_data(x2)
    y_discrete, y_raw_to_discrete = extract_categorical_from_data(y)

    joint_distribution = np.zeros((len(x1_raw_to_discrete), len(x2_raw_to_discrete), len(y_raw_to_discrete)))
    for i in range(numel):
        joint_distribution[x1_discrete[i], x2_discrete[i], y_discrete[i]] += 1
        
    joint_distribution /= np.sum(joint_distribution)

    # Returning maps is useful for debugging but not used in the analysis loop
    return joint_distribution, (x1_raw_to_discrete, x2_raw_to_discrete, y_raw_to_discrete)