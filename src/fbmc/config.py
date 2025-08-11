"""FBMC configuration parameters."""

from dataclasses import dataclass
from typing import List

@dataclass
class FBMCConfig:
    """Configuration parameters for FBMC calculations."""
    reliability_margin_factor: float = 0.1
    min_ram: float = 0.0
    line_usage_threshold: float = 0.2

    # GSK Method options:
    # "ADJUSTABLE_CAP" - Share of Adjustable Capacity
    # "CURRENT_GENERATION" - Current Generation
    # "ITERATIVE_UNCERTAINTY" - Iterative Uncertainty
    # "ITERATIVE_FBMC" - Iterative FBMC
    gsk_method: str = "ITERATIVE_UNCERTAINTY"  
    
    # Uncertainty-based GSK parameters
    uncertain_carriers: List[str] = ("offshore-wind", "onshore-wind")
    num_scenarios: int = 100
    gen_variation_std_dev: float = 0.1
    load_variation_std_dev: float = 0.1

    # Iterative GSK parameters
    max_gsk_iterations: int = 5
    initial_gsk_method: str = "CURRENT_GENERATION"

    # Iterative FBMC parameters
    fbmc_iter_tolerance: float = 0.01

    # Random seed for reproducibility
    base_seed: int = None