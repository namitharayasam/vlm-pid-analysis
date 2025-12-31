# vlm-pid-analysis/pid/__init__.py

"""
The 'pid' package contains the core IPFP algorithm, information metrics, 
and clustering utilities for the Partial Information Decomposition analysis.
"""
from .metrics import get_measure
from .ipfp import alternating_minimization_ipfp
from .utils import clustering, cluster_embeddings