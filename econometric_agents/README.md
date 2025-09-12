# Econometric Agents System

A comprehensive AI-powered econometric analysis system with specialized agents for different types of economic analysis. The system integrates OpenAI GPT models and Google Gemini for intelligent economic interpretation.

## 🚀 Features

### Specialized Agents
- **Regression Agent**: Cross-sectional regression analysis, model selection, diagnostics
- **Time Series Agent**: ARIMA, VAR, VECM, stationarity tests, forecasting
- **Panel Data Agent**: Fixed effects, random effects, Hausman tests
- **Macroeconomic Agent**: Phillips Curve, Taylor Rule, inflation analysis, monetary policy (uses GPT-4o-mini or Gemini-2.0-Flash)

### AI-Powered Analysis
- Intelligent economic interpretation using advanced language models
- Automated variable identification and model selection
- Comprehensive policy analysis and recommendations
- Interactive chat interface with domain experts

### Comprehensive Functionality
- Automated analysis pipelines
- Model diagnostics and validation
- Economic theory integration
- Policy scenario analysis
- Professional reporting and visualization

## 📦 Installation

1. Clone or download the econometric agents system
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up API keys as environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"  # Optional, for macro agent
```

## 🎯 Quick Start

### Basic Usage

```python
import asyncio
from orchestrator import EconometricOrchestrator

async def main():
    # Initialize the system
    orchestrator = EconometricOrchestrator()
    
    # Load your data
    orchestrator.load_data_to_all(your_dataframe)
    
    # Run automated analysis
    results = await orchestrator.auto_analysis_pipeline(
        research_question="What drives inflation in this economy?",
        data=your_dataframe
    )
    
    # Get specific agent
    macro_agent = orchestrator.get_agent("macro")
    
    # Run Phillips Curve analysis
    phillips_results = await macro_agent.phillips_curve_analysis(
        inflation_var="inflation_rate",
        unemployment_var="unemployment_rate"
    )

asyncio.run(main())
```

### Interactive Chat

```python
# Chat with specific agent
await orchestrator.interactive_chat("macro")

# Or general orchestrator chat
await orchestrator.interactive_chat()
```

## 🏛️ Macroeconomic Agent Features

The macroeconomic agent is specialized for economic interpretation and includes:

### Economic Models
- **Phillips Curve**: Inflation-unemployment relationship analysis
- **Taylor Rule**: Monetary policy rule estimation and evaluation
- **Inflation Dynamics**: Persistence, volatility, and trend analysis
- **Monetary Policy**: Policy stance and transmission analysis

### AI Models
- **Primary**: GPT-4o-mini for fast, accurate economic analysis
- **Alternative**: Gemini-2.0-Flash (when available) for enhanced reasoning
- **Fallback**: Automatic fallback between models for reliability

### Economic Theory Integration
- Phillips Curve theory (short-run vs long-run)
- Taylor Principle evaluation
- Quantity theory of money
- Okun's Law relationships

## 📊 Examples

### Run Basic Examples
```bash
cd examples
python basic_usage.py
```

### Macroeconomic Analysis
```bash
cd examples
python macro_analysis_example.py
```

## 🔧 Configuration

Edit `config/config.py` to customize:
- Model selection (GPT-4o, GPT-4o-mini, Gemini)
- Analysis parameters
- Output directories
- Supported models and indicators

## 📁 Project Structure

```
econometric_agents/
├── config/
│   └── config.py              # Configuration settings
├── core/
│   └── base_agent.py          # Base agent class
├── agents/
│   ├── regression_agent.py    # Regression analysis
│   ├── time_series_agent.py   # Time series analysis
│   ├── panel_data_agent.py    # Panel data analysis
│   └── macro_agent.py         # Macroeconomic analysis
├── examples/
│   ├── basic_usage.py         # Basic usage examples
│   └── macro_analysis_example.py  # Macro-specific examples
├── results/                   # Analysis results
├── charts/                    # Generated charts
├── logs/                      # System logs
├── orchestrator.py           # Main orchestrator
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🎓 Academic Applications

### Research Areas
- Monetary Economics
- Macroeconomic Modeling
- Policy Analysis
- Inflation Dynamics
- Central Banking
- Economic Forecasting

### Suitable For
- PhD research projects
- Policy analysis
- Academic papers
- Teaching econometrics
- Professional economic analysis

## 🔬 Advanced Features

### Automated Pipeline
```python
# Comprehensive automated analysis
results = await orchestrator.auto_analysis_pipeline(
    research_question="How does monetary policy affect economic growth?",
    data=macro_data,
    data_description="Quarterly US macro data 1990-2023"
)
```

### Economic Interpretation
```python
# Get AI-powered economic interpretation
interpretation = await macro_agent.economic_interpretation(
    analysis_results=your_results,
    policy_context="Federal Reserve policy analysis"
)
```

### Policy Scenarios
```python
# Analyze different policy scenarios
scenario_results = await macro_agent.taylor_rule_analysis(
    target_inflation=3.0  # Alternative inflation target
)
```

## 📈 Model Capabilities

### Regression Agent
- OLS, WLS, GLS, Logit, Probit, Poisson
- Heteroscedasticity tests (Breusch-Pagan, White)
- Autocorrelation tests (Durbin-Watson)
- Multicollinearity (VIF)
- Model selection (forward, backward, stepwise)

### Time Series Agent
- ARIMA modeling with auto-selection
- VAR and VECM for multivariate analysis
- Stationarity tests (ADF, KPSS)
- Granger causality testing
- Forecasting with confidence intervals

### Panel Data Agent
- Fixed effects and random effects models
- Hausman test for model selection
- Clustered standard errors
- Balanced and unbalanced panels

### Macro Agent
- Phillips Curve estimation (basic and expectations-augmented)
- Taylor Rule analysis with Taylor Principle evaluation
- Inflation persistence and volatility analysis
- Monetary policy stance assessment
- Economic theory integration

## 🤖 AI Integration

### Language Models
- **OpenAI GPT-4o**: Primary model for most agents
- **OpenAI GPT-4o-mini**: Fast model for macro agent
- **Google Gemini-2.0-Flash**: Alternative model for macro analysis

### AI Capabilities
- Economic theory contextualization
- Policy implications analysis
- Statistical interpretation
- Model selection guidance
- Research recommendations

## ⚠️ Requirements

### Software
- Python 3.8+
- Required packages (see requirements.txt)

### API Keys
- OpenAI API key (required)
- Google Gemini API key (optional, for enhanced macro analysis)

### Hardware
- Minimum 8GB RAM recommended
- Internet connection for AI model access

## 📝 Citation

If you use this system in academic research, please cite:

```
Econometric Agents System (2025)
AI-Powered Economic Analysis with Specialized Agents
```

## 🤝 Contributing

This system is designed for academic and research use. Contributions welcome for:
- Additional econometric models
- Enhanced AI interpretations
- New economic theories integration
- Performance improvements

## 📄 License

Academic and research use. Please respect API usage terms for OpenAI and Google.

## 🆘 Support

For issues or questions:
1. Check the examples directory
2. Review configuration settings
3. Ensure API keys are properly set
4. Check system logs in the logs directory

---

**Happy Econometric Analysis! 📊🤖**

