"""
Macroeconomic Analysis Agent
Specialized in macroeconomic analysis, inflation, monetary policy, and economic interpretation
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from termcolor import colored
import asyncio
import json

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import Google Gemini (if available)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from core.base_agent import BaseEconometricAgent
from config.config import (
    MACRO_MODEL, GEMINI_API_KEY, GEMINI_MODEL, OPENAI_MODEL_MINI,
    MACRO_MODELS, MACRO_INDICATORS, MACRO_RELATIONSHIPS
)


class MacroEconomicAgent(BaseEconometricAgent):
    """
    Specialized agent for macroeconomic analysis and economic interpretation
    """
    
    def __init__(self):
        # Initialize with macro-specific model
        super().__init__(
            agent_name="MacroEconomicAnalyst",
            specialization="Macroeconomic Analysis, Inflation, and Monetary Policy",
            model_name=MACRO_MODEL
        )
        
        # Initialize Gemini if available and API key is set
        self.use_gemini = False
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                self.use_gemini = True
                print(colored(f"✓ Gemini {GEMINI_MODEL} initialized for macro analysis", "green"))
            except Exception as e:
                print(colored(f"⚠️  Gemini initialization failed, using OpenAI: {str(e)}", "yellow"))
        
        self.supported_models = MACRO_MODELS
        self.macro_indicators = MACRO_INDICATORS
        self.macro_relationships = MACRO_RELATIONSHIPS
        
        # Economic theory knowledge base
        self.economic_theories = {
            "phillips_curve": {
                "description": "Inverse relationship between unemployment and inflation",
                "key_variables": ["inflation", "unemployment", "nairu"],
                "policy_implications": "Trade-off between inflation and unemployment in short run"
            },
            "taylor_rule": {
                "description": "Monetary policy rule for setting interest rates",
                "key_variables": ["federal_funds_rate", "inflation", "output_gap"],
                "policy_implications": "Guide for central bank interest rate decisions"
            },
            "quantity_theory": {
                "description": "MV = PY relationship between money supply and price level",
                "key_variables": ["money_supply", "velocity", "price_level", "output"],
                "policy_implications": "Long-run relationship between money growth and inflation"
            },
            "okuns_law": {
                "description": "Relationship between economic growth and unemployment",
                "key_variables": ["gdp_growth", "unemployment_change"],
                "policy_implications": "Employment effects of economic growth"
            }
        }
    
    def get_available_methods(self) -> List[str]:
        """Return available macroeconomic methods"""
        return (list(self.supported_models.keys()) + 
                ["phillips_curve_analysis", "taylor_rule_analysis", "inflation_analysis",
                 "monetary_policy_analysis", "economic_interpretation", "policy_simulation",
                 "identify_macro_variables", "structural_break_test"])
    
    async def _call_ai_api(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> str:
        """
        Call AI API (Gemini or OpenAI) based on availability
        """
        if self.use_gemini:
            try:
                # Convert messages to Gemini format
                prompt = self._convert_messages_to_gemini_prompt(messages)
                
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=self.temperature
                    )
                )
                
                return response.text
                
            except Exception as e:
                print(colored(f"⚠️  Gemini API failed, falling back to OpenAI: {str(e)}", "yellow"))
                # Fall back to OpenAI
                return await self._call_openai_api(messages, max_tokens)
        else:
            # Use OpenAI
            return await self._call_openai_api(messages, max_tokens)
    
    def _convert_messages_to_gemini_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to Gemini prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    async def perform_analysis(self, 
                             analysis_type: str,
                             variables: Union[str, List[str]] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Perform macroeconomic analysis
        
        Args:
            analysis_type: Type of macro analysis
            variables: Variables for analysis
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing analysis results
        """
        if self.current_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        print(colored(f"🏛️  Running {analysis_type.upper()} macroeconomic analysis...", "cyan"))
        
        try:
            if analysis_type == "phillips_curve":
                return await self.phillips_curve_analysis(**kwargs)
            elif analysis_type == "taylor_rule":
                return await self.taylor_rule_analysis(**kwargs)
            elif analysis_type == "inflation_analysis":
                return await self.inflation_analysis(variables, **kwargs)
            elif analysis_type == "monetary_policy_analysis":
                return await self.monetary_policy_analysis(variables, **kwargs)
            elif analysis_type == "var_macro":
                return await self.macro_var_analysis(variables, **kwargs)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            error_msg = f"Error in {analysis_type} analysis: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def identify_macro_variables(self) -> Dict[str, Any]:
        """
        Automatically identify macroeconomic variables in the dataset
        
        Returns:
            Dictionary mapping macro categories to identified variables
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        print(colored("🔍 Identifying macroeconomic variables in dataset...", "cyan"))
        
        identified_vars = {}
        column_names = [col.lower() for col in self.current_data.columns]
        
        # Check for each macro indicator category
        for category, indicators in self.macro_indicators.items():
            identified_vars[category] = []
            
            for indicator in indicators:
                # Look for partial matches in column names
                for col, col_lower in zip(self.current_data.columns, column_names):
                    if any(keyword in col_lower for keyword in indicator.split('_')):
                        identified_vars[category].append(col)
        
        # Get AI interpretation of identified variables
        interpretation = await self._interpret_variable_identification(identified_vars)
        
        results = {
            "identified_variables": identified_vars,
            "total_macro_variables": sum(len(vars) for vars in identified_vars.values()),
            "ai_interpretation": interpretation
        }
        
        self.analysis_results["variable_identification"] = results
        
        print(colored("✅ Variable identification completed", "green"))
        return results
    
    async def phillips_curve_analysis(self, 
                                    inflation_var: str = None,
                                    unemployment_var: str = None,
                                    expectations_var: str = None) -> Dict[str, Any]:
        """
        Analyze Phillips Curve relationship
        
        Args:
            inflation_var: Inflation variable name
            unemployment_var: Unemployment variable name
            expectations_var: Inflation expectations variable (optional)
            
        Returns:
            Phillips Curve analysis results
        """
        print(colored("📈 Analyzing Phillips Curve relationship...", "cyan"))
        
        # Auto-identify variables if not provided
        if not inflation_var or not unemployment_var:
            var_id = await self.identify_macro_variables()
            
            if not inflation_var and var_id["identified_variables"]["inflation"]:
                inflation_var = var_id["identified_variables"]["inflation"][0]
            
            if not unemployment_var and var_id["identified_variables"]["employment"]:
                unemployment_var = var_id["identified_variables"]["employment"][0]
        
        if not inflation_var or not unemployment_var:
            raise ValueError("Could not identify inflation and unemployment variables")
        
        # Prepare data
        analysis_data = self.current_data[[inflation_var, unemployment_var]].dropna()
        
        if expectations_var and expectations_var in self.current_data.columns:
            analysis_data = pd.concat([analysis_data, self.current_data[expectations_var]], axis=1).dropna()
        
        # Basic Phillips Curve regression
        y = analysis_data[inflation_var]
        X = sm.add_constant(analysis_data[unemployment_var])
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Expectations-augmented Phillips Curve if expectations available
        if expectations_var and expectations_var in analysis_data.columns:
            X_aug = sm.add_constant(analysis_data[[unemployment_var, expectations_var]])
            model_aug = sm.OLS(y, X_aug)
            results_aug = model_aug.fit()
        else:
            results_aug = None
        
        # Extract results
        phillips_results = {
            "model_type": "phillips_curve",
            "inflation_variable": inflation_var,
            "unemployment_variable": unemployment_var,
            "expectations_variable": expectations_var,
            "observations": int(results.nobs),
            "r_squared": float(results.rsquared),
            "slope_coefficient": float(results.params[unemployment_var]),
            "slope_pvalue": float(results.pvalues[unemployment_var]),
            "intercept": float(results.params['const']),
            "intercept_pvalue": float(results.pvalues['const']),
            "phillips_slope_interpretation": "Negative" if results.params[unemployment_var] < 0 else "Positive",
            "statistical_significance": "Significant" if results.pvalues[unemployment_var] < 0.05 else "Not significant"
        }
        
        if results_aug:
            phillips_results["expectations_augmented"] = {
                "r_squared": float(results_aug.rsquared),
                "unemployment_coef": float(results_aug.params[unemployment_var]),
                "expectations_coef": float(results_aug.params[expectations_var]),
                "unemployment_pvalue": float(results_aug.pvalues[unemployment_var]),
                "expectations_pvalue": float(results_aug.pvalues[expectations_var])
            }
        
        # Economic interpretation
        interpretation = await self._interpret_phillips_curve(phillips_results)
        phillips_results["economic_interpretation"] = interpretation
        
        self.analysis_results["phillips_curve"] = phillips_results
        
        print(colored("✅ Phillips Curve analysis completed", "green"))
        return phillips_results
    
    async def taylor_rule_analysis(self,
                                 policy_rate_var: str = None,
                                 inflation_var: str = None,
                                 output_gap_var: str = None,
                                 target_inflation: float = 2.0) -> Dict[str, Any]:
        """
        Analyze Taylor Rule relationship
        
        Args:
            policy_rate_var: Policy interest rate variable
            inflation_var: Inflation variable
            output_gap_var: Output gap variable
            target_inflation: Inflation target (default 2%)
            
        Returns:
            Taylor Rule analysis results
        """
        print(colored("🏦 Analyzing Taylor Rule relationship...", "cyan"))
        
        # Auto-identify variables if not provided
        if not policy_rate_var or not inflation_var:
            var_id = await self.identify_macro_variables()
            
            if not policy_rate_var and var_id["identified_variables"]["monetary_policy"]:
                policy_rate_var = var_id["identified_variables"]["monetary_policy"][0]
            
            if not inflation_var and var_id["identified_variables"]["inflation"]:
                inflation_var = var_id["identified_variables"]["inflation"][0]
            
            if not output_gap_var and var_id["identified_variables"]["output"]:
                output_gap_var = var_id["identified_variables"]["output"][0]
        
        if not policy_rate_var or not inflation_var:
            raise ValueError("Could not identify policy rate and inflation variables")
        
        # Prepare data
        required_vars = [policy_rate_var, inflation_var]
        if output_gap_var:
            required_vars.append(output_gap_var)
        
        analysis_data = self.current_data[required_vars].dropna()
        
        # Calculate inflation gap
        analysis_data['inflation_gap'] = analysis_data[inflation_var] - target_inflation
        
        # Taylor Rule regression
        y = analysis_data[policy_rate_var]
        
        if output_gap_var:
            X = sm.add_constant(analysis_data[['inflation_gap', output_gap_var]])
        else:
            X = sm.add_constant(analysis_data['inflation_gap'])
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract results
        taylor_results = {
            "model_type": "taylor_rule",
            "policy_rate_variable": policy_rate_var,
            "inflation_variable": inflation_var,
            "output_gap_variable": output_gap_var,
            "target_inflation": target_inflation,
            "observations": int(results.nobs),
            "r_squared": float(results.rsquared),
            "intercept": float(results.params['const']),
            "inflation_response": float(results.params['inflation_gap']),
            "inflation_response_pvalue": float(results.pvalues['inflation_gap'])
        }
        
        if output_gap_var:
            taylor_results["output_response"] = float(results.params[output_gap_var])
            taylor_results["output_response_pvalue"] = float(results.pvalues[output_gap_var])
        
        # Taylor principle check (inflation response > 1)
        taylor_results["taylor_principle"] = {
            "satisfied": taylor_results["inflation_response"] > 1.0,
            "interpretation": "Satisfies Taylor Principle" if taylor_results["inflation_response"] > 1.0 else "Violates Taylor Principle"
        }
        
        # Economic interpretation
        interpretation = await self._interpret_taylor_rule(taylor_results)
        taylor_results["economic_interpretation"] = interpretation
        
        self.analysis_results["taylor_rule"] = taylor_results
        
        print(colored("✅ Taylor Rule analysis completed", "green"))
        return taylor_results
    
    async def inflation_analysis(self, 
                               inflation_vars: Union[str, List[str]],
                               **kwargs) -> Dict[str, Any]:
        """
        Comprehensive inflation analysis
        
        Args:
            inflation_vars: Inflation variable(s) to analyze
            
        Returns:
            Inflation analysis results
        """
        if isinstance(inflation_vars, str):
            inflation_vars = [inflation_vars]
        
        print(colored("💰 Performing comprehensive inflation analysis...", "cyan"))
        
        results = {
            "analysis_type": "inflation_analysis",
            "variables": inflation_vars,
            "descriptive_stats": {},
            "persistence": {},
            "volatility": {},
            "trends": {}
        }
        
        for var in inflation_vars:
            if var not in self.current_data.columns:
                continue
            
            series = self.current_data[var].dropna()
            
            # Descriptive statistics
            results["descriptive_stats"][var] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis())
            }
            
            # Persistence (AR(1) coefficient)
            if len(series) > 10:
                y = series[1:].values
                x = sm.add_constant(series[:-1].values)
                ar_model = sm.OLS(y, x).fit()
                results["persistence"][var] = {
                    "ar1_coefficient": float(ar_model.params[1]),
                    "ar1_pvalue": float(ar_model.pvalues[1]),
                    "interpretation": "Highly persistent" if ar_model.params[1] > 0.8 else "Moderately persistent" if ar_model.params[1] > 0.5 else "Low persistence"
                }
            
            # Volatility analysis (rolling standard deviation)
            if len(series) > 24:  # Need sufficient data for rolling window
                rolling_std = series.rolling(window=12).std().dropna()
                results["volatility"][var] = {
                    "average_volatility": float(rolling_std.mean()),
                    "volatility_trend": "Increasing" if rolling_std.corr(pd.Series(range(len(rolling_std)))) > 0.1 else "Decreasing" if rolling_std.corr(pd.Series(range(len(rolling_std)))) < -0.1 else "Stable"
                }
        
        # Economic interpretation
        interpretation = await self._interpret_inflation_analysis(results)
        results["economic_interpretation"] = interpretation
        
        self.analysis_results["inflation_analysis"] = results
        
        print(colored("✅ Inflation analysis completed", "green"))
        return results
    
    async def monetary_policy_analysis(self,
                                     policy_vars: Union[str, List[str]],
                                     **kwargs) -> Dict[str, Any]:
        """
        Analyze monetary policy variables and their relationships
        
        Args:
            policy_vars: Monetary policy variable(s)
            
        Returns:
            Monetary policy analysis results
        """
        if isinstance(policy_vars, str):
            policy_vars = [policy_vars]
        
        print(colored("🏦 Analyzing monetary policy variables...", "cyan"))
        
        results = {
            "analysis_type": "monetary_policy_analysis",
            "variables": policy_vars,
            "policy_stance": {},
            "transmission": {},
            "effectiveness": {}
        }
        
        # Analyze each policy variable
        for var in policy_vars:
            if var not in self.current_data.columns:
                continue
            
            series = self.current_data[var].dropna()
            
            # Policy stance analysis
            mean_level = series.mean()
            recent_level = series.tail(12).mean() if len(series) >= 12 else series.mean()
            
            results["policy_stance"][var] = {
                "historical_average": float(mean_level),
                "recent_average": float(recent_level),
                "stance": "Accommodative" if recent_level < mean_level else "Restrictive" if recent_level > mean_level else "Neutral",
                "change_direction": "Easing" if series.diff().tail(6).mean() < 0 else "Tightening" if series.diff().tail(6).mean() > 0 else "Stable"
            }
        
        # Economic interpretation
        interpretation = await self._interpret_monetary_policy(results)
        results["economic_interpretation"] = interpretation
        
        self.analysis_results["monetary_policy_analysis"] = results
        
        print(colored("✅ Monetary policy analysis completed", "green"))
        return results
    
    async def macro_var_analysis(self,
                               variables: List[str],
                               maxlags: int = 4) -> Dict[str, Any]:
        """
        Vector Autoregression analysis for macroeconomic variables
        
        Args:
            variables: List of macro variables
            maxlags: Maximum lags for VAR
            
        Returns:
            Macro VAR analysis results
        """
        print(colored("📊 Running Macroeconomic VAR analysis...", "cyan"))
        
        if len(variables) < 2:
            raise ValueError("VAR requires at least 2 variables")
        
        # Use the time series agent's VAR functionality but with macro interpretation
        from agents.time_series_agent import TimeSeriesAgent
        ts_agent = TimeSeriesAgent()
        ts_agent.current_data = self.current_data
        
        # Run VAR analysis
        var_results = await ts_agent._fit_var(variables, maxlags)
        
        # Add macroeconomic interpretation
        macro_interpretation = await self._interpret_macro_var(var_results, variables)
        var_results["macroeconomic_interpretation"] = macro_interpretation
        
        self.analysis_results["macro_var"] = var_results
        
        print(colored("✅ Macro VAR analysis completed", "green"))
        return var_results
    
    async def economic_interpretation(self, 
                                    analysis_results: Dict[str, Any],
                                    policy_context: str = "") -> str:
        """
        Provide comprehensive economic interpretation of results
        
        Args:
            analysis_results: Results from any econometric analysis
            policy_context: Additional policy context
            
        Returns:
            Comprehensive economic interpretation
        """
        print(colored("🎯 Generating comprehensive economic interpretation...", "cyan"))
        
        system_prompt = """You are a distinguished macroeconomist with expertise in monetary policy, inflation dynamics, and economic theory. 
        
        Provide a comprehensive economic interpretation that includes:
        1. Economic theory context and relevance
        2. Policy implications and recommendations
        3. Historical context and comparison to economic literature
        4. Transmission mechanisms and economic channels
        5. Potential risks and limitations
        6. Forward-looking implications for policy and markets
        
        Focus on practical insights for policymakers, economists, and market participants.
        Reference relevant economic theories, historical episodes, and policy frameworks."""
        
        user_prompt = f"""
        Analysis Results: {json.dumps(analysis_results, indent=2, default=str)}
        
        Policy Context: {policy_context}
        
        Please provide a comprehensive macroeconomic interpretation of these results.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        interpretation = await self._call_ai_api(messages, max_tokens=3000)
        
        print(colored("✅ Economic interpretation completed", "green"))
        return interpretation
    
    # AI Interpretation Methods
    async def _interpret_variable_identification(self, identified_vars: Dict[str, List[str]]) -> str:
        """AI interpretation of variable identification"""
        try:
            system_prompt = """You are a macroeconomic data expert. Analyze the identified macroeconomic variables and provide:
            1. Assessment of data coverage for macro analysis
            2. Key relationships that can be explored
            3. Potential macro models that can be estimated
            4. Data gaps and recommendations
            5. Suggested analysis priorities
            
            Focus on macroeconomic theory and empirical possibilities."""
            
            user_prompt = f"Identified Macroeconomic Variables: {json.dumps(identified_vars, indent=2)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages)
        except:
            return "Error generating variable identification interpretation"
    
    async def _interpret_phillips_curve(self, results: Dict[str, Any]) -> str:
        """AI interpretation of Phillips Curve results"""
        try:
            system_prompt = """You are a macroeconomic expert specializing in Phillips Curve analysis. Provide:
            1. Economic interpretation of the inflation-unemployment relationship
            2. Assessment of Phillips Curve slope and significance
            3. Implications for monetary policy and NAIRU
            4. Comparison with economic theory and empirical literature
            5. Policy recommendations and caveats
            
            Consider both short-run and long-run Phillips Curve theory."""
            
            user_prompt = f"Phillips Curve Results: {json.dumps(results, indent=2, default=str)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages)
        except:
            return "Error generating Phillips Curve interpretation"
    
    async def _interpret_taylor_rule(self, results: Dict[str, Any]) -> str:
        """AI interpretation of Taylor Rule results"""
        try:
            system_prompt = """You are a monetary policy expert. Analyze the Taylor Rule results and provide:
            1. Assessment of central bank policy rule adherence
            2. Taylor Principle evaluation and implications
            3. Policy responsiveness to inflation and output
            4. Comparison with theoretical Taylor Rule
            5. Implications for monetary policy effectiveness
            6. Historical context and policy regime analysis
            
            Focus on monetary policy theory and central banking practice."""
            
            user_prompt = f"Taylor Rule Results: {json.dumps(results, indent=2, default=str)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages)
        except:
            return "Error generating Taylor Rule interpretation"
    
    async def _interpret_inflation_analysis(self, results: Dict[str, Any]) -> str:
        """AI interpretation of inflation analysis"""
        try:
            system_prompt = """You are an inflation dynamics expert. Analyze the inflation results and provide:
            1. Assessment of inflation characteristics and behavior
            2. Persistence and volatility implications
            3. Inflation targeting and policy implications
            4. Risk assessment and forward-looking concerns
            5. Comparison with inflation theory and empirical patterns
            6. Recommendations for inflation management
            
            Focus on inflation theory, central banking, and policy implications."""
            
            user_prompt = f"Inflation Analysis Results: {json.dumps(results, indent=2, default=str)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages)
        except:
            return "Error generating inflation analysis interpretation"
    
    async def _interpret_monetary_policy(self, results: Dict[str, Any]) -> str:
        """AI interpretation of monetary policy analysis"""
        try:
            system_prompt = """You are a monetary policy expert. Analyze the monetary policy results and provide:
            1. Assessment of current policy stance
            2. Policy transmission mechanism evaluation
            3. Effectiveness of monetary policy tools
            4. Historical context and comparison
            5. Forward-looking policy implications
            6. Risks and limitations of current approach
            
            Focus on central banking theory and policy practice."""
            
            user_prompt = f"Monetary Policy Analysis Results: {json.dumps(results, indent=2, default=str)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages)
        except:
            return "Error generating monetary policy interpretation"
    
    async def _interpret_macro_var(self, var_results: Dict[str, Any], variables: List[str]) -> str:
        """AI interpretation of macro VAR results"""
        try:
            system_prompt = """You are a macroeconomic modeling expert. Analyze the VAR results for macroeconomic variables and provide:
            1. Dynamic relationships between macro variables
            2. Economic interpretation of coefficients and lags
            3. Policy transmission mechanisms
            4. Impulse response and forecast implications
            5. Structural interpretation and economic theory
            6. Policy and investment implications
            
            Focus on macroeconomic theory and policy analysis."""
            
            user_prompt = f"""
            VAR Results for variables: {variables}
            {json.dumps(var_results, indent=2, default=str)}
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_ai_api(messages, max_tokens=2500)
        except:
            return "Error generating macro VAR interpretation"
    
    def print_macro_summary(self):
        """Print summary of macroeconomic analyses"""
        print(colored("\n🏛️  Macroeconomic Analysis Summary", "cyan"))
        print(colored("=" * 50, "cyan"))
        
        if not self.analysis_results:
            print(colored("No analyses completed yet", "yellow"))
            return
        
        for analysis_type, results in self.analysis_results.items():
            print(colored(f"\n{analysis_type.upper().replace('_', ' ')}:", "yellow"))
            
            if analysis_type == "phillips_curve":
                print(f"  Inflation-Unemployment slope: {results['slope_coefficient']:.4f}")
                print(f"  Relationship: {results['phillips_slope_interpretation']}")
                print(f"  Statistical significance: {results['statistical_significance']}")
            
            elif analysis_type == "taylor_rule":
                print(f"  Inflation response: {results['inflation_response']:.4f}")
                print(f"  Taylor Principle: {results['taylor_principle']['interpretation']}")
                if 'output_response' in results:
                    print(f"  Output response: {results['output_response']:.4f}")
            
            elif analysis_type == "inflation_analysis":
                for var in results['variables']:
                    if var in results['descriptive_stats']:
                        stats = results['descriptive_stats'][var]
                        print(f"  {var}: Mean = {stats['mean']:.2f}%, Std = {stats['std']:.2f}%")
            
            elif analysis_type == "variable_identification":
                total_vars = results['total_macro_variables']
                print(f"  Total macro variables identified: {total_vars}")
        
        print(colored("\n" + "=" * 50, "cyan"))
