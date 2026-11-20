"""Compatibility facade for the pricing analysis services.

Data preparation lives in :mod:`pricing_data`, model evaluation in
:mod:`demand_forecasting`, and business metrics/scenarios in
:mod:`pricing_insights`. Existing app and report imports remain supported here.
"""

from demand_forecasting import (
    BlendedDemandModel,
    ConstantDemandModel,
    DefaultDemandEstimatorProvider,
    XGB_AVAILABLE,
    model_demand,
)
from pricing_data import (
    demand_features,
    prepare_demand_data,
    prepare_pricing_data,
    split_by_date,
    split_for_backtest,
)
from pricing_insights import (
    assess_pricing_constraints,
    business_segments,
    build_price_response_curve,
    calculate_price_elasticity,
    calculate_promotion_lift,
    explain_model,
    generate_business_insights,
    run_sensitivity_analysis,
)

__all__ = [
    "BlendedDemandModel",
    "ConstantDemandModel",
    "DefaultDemandEstimatorProvider",
    "XGB_AVAILABLE",
    "assess_pricing_constraints",
    "business_segments",
    "build_price_response_curve",
    "calculate_price_elasticity",
    "calculate_promotion_lift",
    "demand_features",
    "explain_model",
    "generate_business_insights",
    "model_demand",
    "prepare_demand_data",
    "prepare_pricing_data",
    "run_sensitivity_analysis",
    "split_by_date",
    "split_for_backtest",
]