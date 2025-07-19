"""FBMC configuration parameters."""

from dataclasses import dataclass
import enum

class GSKMethod(enum.Enum):
    """Tracks implemented GSK methods."""
    ADJUSTABLE_CAP: str = "ADJUSTABLE_CAP"
    CURRENT_GENERATION: str = "CURRENT_GENERATION"
    ITERATIVE_UNCERTAINTY: str = "ITERATIVE_UNCERTAINTY"
    ITERATIVE_FBMC: str = "ITERATIVE_FBMC"
    MERIT_ORDER: str = "MERIT_ORDER"

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
    
    # use the GSKMethod class 
    gsk_method: str = GSKMethod.ADJUSTABLE_CAP
    # gsk_method: str = "ADJUSTABLE_CAP"
    # gsk_method = GSKMethod(gsk_method)
    
    # Uncertainty-based GSK parameters
    uncertain_carriers: tuple[str] = ("offshore-wind", "onshore-wind")
    num_scenarios: int = 100
    gen_variation_std_dev: float = 0.1
    load_variation_std_dev: float = 0.1

    # Iterative GSK parameters
    max_gsk_iterations: int = 5
    initial_gsk_method: str = GSKMethod.CURRENT_GENERATION
