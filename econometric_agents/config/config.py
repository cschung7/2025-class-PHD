"""
Configuration settings for Econometric Agents
"""
import os
from typing import Dict, List, Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4o"
OPENAI_MODEL_MINI = "gpt-4o-mini"

# Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Using latest available Gemini model

# Alternative Models for Specialized Agents
MACRO_MODEL = "gpt-4o-mini"  # Can be switched to Gemini when available

# Agent Configuration
MAX_TOKENS = 4000
TEMPERATURE = 0.1
MAX_RETRIES = 3

# Econometric Analysis Settings
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_CONFIDENCE_INTERVAL = 0.95

# Data Processing Settings
MAX_DATA_POINTS = 10000
DEFAULT_MISSING_VALUE_STRATEGY = "drop"  # "drop", "interpolate", "forward_fill"

# Output Settings
RESULTS_DIR = "/mnt/nas/Class/2025/PHD/econometric_agents/results"
CHARTS_DIR = "/mnt/nas/Class/2025/PHD/econometric_agents/charts"
LOGS_DIR = "/mnt/nas/Class/2025/PHD/econometric_agents/logs"

# Supported Econometric Models
SUPPORTED_MODELS = {
    "regression": ["ols", "wls", "gls", "logit", "probit", "poisson"],
    "time_series": ["arima", "var", "vecm", "garch", "ardl"],
    "panel_data": ["fixed_effects", "random_effects", "pooled_ols", "first_difference"],
    "causality": ["granger", "johansen", "engle_granger", "instrumental_variables"]
}

# Agent Specializations
AGENT_TYPES = {
    "regression_analyst": "Specialized in cross-sectional regression analysis",
    "time_series_analyst": "Specialized in time series econometrics",
    "panel_data_analyst": "Specialized in panel data analysis",
    "causality_analyst": "Specialized in causal inference and identification",
    "diagnostic_analyst": "Specialized in model diagnostics and validation",
    "forecasting_analyst": "Specialized in economic forecasting",
    "macro_analyst": "Specialized in macroeconomic analysis, inflation, and monetary policy"
}

# Diagnostic Tests
DIAGNOSTIC_TESTS = {
    "regression": ["heteroscedasticity", "autocorrelation", "normality", "linearity", "multicollinearity"],
    "time_series": ["stationarity", "autocorrelation", "arch_effects", "cointegration"],
    "panel_data": ["hausman", "breusch_pagan", "fixed_effects_test"]
}

# Macroeconomic Models and Indicators
MACRO_MODELS = {
    "phillips_curve": "Relationship between inflation and unemployment",
    "taylor_rule": "Monetary policy rule relating interest rates to inflation and output",
    "is_lm": "Investment-Savings and Liquidity preference-Money supply model",
    "dsge": "Dynamic Stochastic General Equilibrium models",
    "var_macro": "Vector Autoregression for macroeconomic variables",
    "vecm_macro": "Vector Error Correction for cointegrated macro variables"
}

MACRO_INDICATORS = {
    "inflation": ["cpi", "pce", "core_inflation", "inflation_expectations"],
    "monetary_policy": ["federal_funds_rate", "money_supply", "yield_curve", "qe_measures"],
    "output": ["gdp", "industrial_production", "capacity_utilization", "output_gap"],
    "employment": ["unemployment_rate", "nfp", "participation_rate", "job_openings"],
    "financial": ["stock_prices", "bond_yields", "credit_spreads", "exchange_rates"],
    "expectations": ["inflation_expectations", "consumer_confidence", "business_sentiment"]
}

MACRO_RELATIONSHIPS = {
    "phillips_curve": {
        "dependent": "inflation",
        "independent": ["unemployment", "inflation_expectations"],
        "description": "Trade-off between inflation and unemployment"
    },
    "taylor_rule": {
        "dependent": "policy_rate",
        "independent": ["inflation_gap", "output_gap"],
        "description": "Central bank policy rule"
    },
    "okuns_law": {
        "dependent": "unemployment_change",
        "independent": ["gdp_growth"],
        "description": "Relationship between economic growth and unemployment"
    }
}
