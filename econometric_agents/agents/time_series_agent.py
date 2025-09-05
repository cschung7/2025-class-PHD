"""
Time Series Analysis Agent
Specialized in time series econometrics using statsmodels
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from termcolor import colored

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from core.base_agent import BaseEconometricAgent


class TimeSeriesAgent(BaseEconometricAgent):
    """
    Specialized agent for time series econometric analysis
    """
    
    def __init__(self):
        super().__init__(
            agent_name="TimeSeriesAnalyst",
            specialization="Time Series Econometrics and Forecasting"
        )
        
        self.supported_models = {
            "arima": "ARIMA Models",
            "var": "Vector Autoregression",
            "vecm": "Vector Error Correction Model",
            "ardl": "Autoregressive Distributed Lag",
            "garch": "GARCH Models"
        }
        
        self.diagnostic_tests = [
            "stationarity",
            "autocorrelation",
            "cointegration",
            "granger_causality",
            "arch_effects"
        ]
    
    def get_available_methods(self) -> List[str]:
        """Return available time series methods"""
        return (list(self.supported_models.keys()) + 
                ["stationarity_test", "cointegration_test", "granger_causality", 
                 "seasonal_decomposition", "forecast", "impulse_response"])
    
    def set_time_index(self, date_column: str, freq: str = None):
        """
        Set time index for time series analysis
        
        Args:
            date_column: Name of the date column
            freq: Frequency of the time series (e.g., 'D', 'M', 'Q', 'A')
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        try:
            self.current_data[date_column] = pd.to_datetime(self.current_data[date_column])
            self.current_data = self.current_data.set_index(date_column)
            
            if freq:
                self.current_data = self.current_data.asfreq(freq)
            
            print(colored(f"✓ Time index set: {date_column} (freq: {freq or 'inferred'})", "green"))
            self.logger.info(f"Time index set: {date_column}")
            
        except Exception as e:
            error_msg = f"Error setting time index: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def stationarity_test(self, 
                               variables: Union[str, List[str]],
                               test_type: str = "both") -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests
        
        Args:
            variables: Variable name(s) to test
            test_type: "adf", "kpss", or "both"
            
        Returns:
            Dictionary containing stationarity test results
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        if isinstance(variables, str):
            variables = [variables]
        
        print(colored(f"🔍 Testing stationarity for: {', '.join(variables)}", "cyan"))
        
        results = {}
        
        for var in variables:
            if var not in self.current_data.columns:
                print(colored(f"⚠️  Variable {var} not found in data", "yellow"))
                continue
            
            series = self.current_data[var].dropna()
            var_results = {"variable": var}
            
            try:
                # Augmented Dickey-Fuller test
                if test_type in ["adf", "both"]:
                    adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(series)
                    
                    var_results["adf"] = {
                        "statistic": float(adf_stat),
                        "p_value": float(adf_pvalue),
                        "lags_used": int(adf_lags),
                        "observations": int(adf_nobs),
                        "critical_values": {k: float(v) for k, v in adf_critical.items()},
                        "interpretation": "Stationary" if adf_pvalue < 0.05 else "Non-stationary"
                    }
                
                # KPSS test
                if test_type in ["kpss", "both"]:
                    kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(series)
                    
                    var_results["kpss"] = {
                        "statistic": float(kpss_stat),
                        "p_value": float(kpss_pvalue),
                        "lags_used": int(kpss_lags),
                        "critical_values": {k: float(v) for k, v in kpss_critical.items()},
                        "interpretation": "Stationary" if kpss_pvalue > 0.05 else "Non-stationary"
                    }
                
                # Overall assessment
                if test_type == "both":
                    adf_stationary = var_results["adf"]["interpretation"] == "Stationary"
                    kpss_stationary = var_results["kpss"]["interpretation"] == "Stationary"
                    
                    if adf_stationary and kpss_stationary:
                        var_results["overall"] = "Stationary"
                    elif not adf_stationary and not kpss_stationary:
                        var_results["overall"] = "Non-stationary"
                    else:
                        var_results["overall"] = "Inconclusive"
                
                results[var] = var_results
                
                print(colored(f"  {var}: {var_results.get('overall', var_results.get('adf', var_results.get('kpss', {})).get('interpretation', 'Unknown'))}", "blue"))
                
            except Exception as e:
                print(colored(f"  Error testing {var}: {str(e)}", "red"))
                results[var] = {"error": str(e)}
        
        # Get AI interpretation
        interpretation = await self._interpret_stationarity_results(results)
        
        final_results = {
            "test_type": test_type,
            "variables": variables,
            "results": results,
            "ai_interpretation": interpretation
        }
        
        self.analysis_results["stationarity"] = final_results
        return final_results
    
    async def perform_analysis(self,
                             model_type: str,
                             variables: Union[str, List[str]],
                             **kwargs) -> Dict[str, Any]:
        """
        Perform time series analysis
        
        Args:
            model_type: Type of time series model
            variables: Variable name(s) for analysis
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary containing analysis results
        """
        if self.current_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        print(colored(f"🔍 Running {model_type.upper()} time series analysis...", "cyan"))
        
        try:
            if model_type == "arima":
                return await self._fit_arima(variables, **kwargs)
            elif model_type == "var":
                return await self._fit_var(variables, **kwargs)
            elif model_type == "vecm":
                return await self._fit_vecm(variables, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            error_msg = f"Error in {model_type} analysis: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def _fit_arima(self, variable: str, order: Tuple[int, int, int] = None, **kwargs) -> Dict[str, Any]:
        """
        Fit ARIMA model
        
        Args:
            variable: Variable name for ARIMA modeling
            order: ARIMA order (p, d, q)
            
        Returns:
            ARIMA results dictionary
        """
        if isinstance(variable, list):
            variable = variable[0]  # ARIMA is univariate
        
        series = self.current_data[variable].dropna()
        
        # Auto-determine order if not provided
        if order is None:
            print(colored("  Auto-determining ARIMA order...", "yellow"))
            # Simple grid search for demonstration
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            order = best_order
            print(colored(f"  Selected order: {order}", "blue"))
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Extract results
        results = {
            "model_type": "arima",
            "variable": variable,
            "order": order,
            "observations": int(fitted_model.nobs),
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "log_likelihood": float(fitted_model.llf),
            "coefficients": {},
            "residual_diagnostics": {}
        }
        
        # Extract coefficients
        for i, param in enumerate(fitted_model.params.index):
            results["coefficients"][param] = {
                "coefficient": float(fitted_model.params[param]),
                "std_error": float(fitted_model.bse[param]),
                "t_statistic": float(fitted_model.tvalues[param]),
                "p_value": float(fitted_model.pvalues[param])
            }
        
        # Residual diagnostics
        residuals = fitted_model.resid
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        results["residual_diagnostics"]["ljung_box"] = {
            "statistic": float(lb_test['lb_stat'].iloc[-1]),
            "p_value": float(lb_test['lb_pvalue'].iloc[-1]),
            "interpretation": "No autocorrelation" if lb_test['lb_pvalue'].iloc[-1] > 0.05 else "Autocorrelation detected"
        }
        
        # Store fitted model
        self.fitted_model = fitted_model
        
        # Get AI interpretation
        interpretation = await self._interpret_arima_results(results)
        results["ai_interpretation"] = interpretation
        
        self.analysis_results["arima"] = results
        
        print(colored("✅ ARIMA analysis completed", "green"))
        return results
    
    async def _fit_var(self, variables: List[str], maxlags: int = None, **kwargs) -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model
        
        Args:
            variables: List of variables for VAR
            maxlags: Maximum number of lags to consider
            
        Returns:
            VAR results dictionary
        """
        if isinstance(variables, str):
            variables = [variables]
        
        if len(variables) < 2:
            raise ValueError("VAR requires at least 2 variables")
        
        data = self.current_data[variables].dropna()
        
        # Determine optimal lag order if not specified
        if maxlags is None:
            maxlags = min(12, len(data) // 10)  # Rule of thumb
        
        model = VAR(data)
        
        # Lag order selection
        lag_order_results = model.select_order(maxlags=maxlags)
        optimal_lags = lag_order_results.selected_orders['aic']
        
        print(colored(f"  Selected lag order: {optimal_lags}", "blue"))
        
        # Fit VAR model
        fitted_model = model.fit(optimal_lags)
        
        # Extract results
        results = {
            "model_type": "var",
            "variables": variables,
            "lag_order": int(optimal_lags),
            "observations": int(fitted_model.nobs),
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "log_likelihood": float(fitted_model.llf),
            "equations": {}
        }
        
        # Extract equation results
        for i, var in enumerate(variables):
            eq_results = fitted_model.params.iloc[:, i]
            eq_pvalues = fitted_model.pvalues.iloc[:, i]
            eq_stderr = fitted_model.stderr.iloc[:, i]
            
            results["equations"][var] = {
                "coefficients": {},
                "r_squared": float(fitted_model.rsquared[i])
            }
            
            for param_name in eq_results.index:
                results["equations"][var]["coefficients"][param_name] = {
                    "coefficient": float(eq_results[param_name]),
                    "std_error": float(eq_stderr[param_name]),
                    "p_value": float(eq_pvalues[param_name])
                }
        
        # Store fitted model
        self.fitted_model = fitted_model
        
        # Get AI interpretation
        interpretation = await self._interpret_var_results(results)
        results["ai_interpretation"] = interpretation
        
        self.analysis_results["var"] = results
        
        print(colored("✅ VAR analysis completed", "green"))
        return results
    
    async def _fit_vecm(self, variables: List[str], **kwargs) -> Dict[str, Any]:
        """
        Fit Vector Error Correction Model (VECM)
        
        Args:
            variables: List of variables for VECM
            
        Returns:
            VECM results dictionary
        """
        if len(variables) < 2:
            raise ValueError("VECM requires at least 2 variables")
        
        data = self.current_data[variables].dropna()
        
        # Test for cointegration first
        johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        # Determine number of cointegrating relationships
        trace_stats = johansen_test.lr1
        critical_values = johansen_test.cvt[:, 1]  # 5% critical values
        
        coint_rank = 0
        for i, (stat, cv) in enumerate(zip(trace_stats, critical_values)):
            if stat > cv:
                coint_rank = len(variables) - i
                break
        
        print(colored(f"  Cointegrating relationships found: {coint_rank}", "blue"))
        
        if coint_rank == 0:
            print(colored("  Warning: No cointegration found. Consider VAR in levels or differences.", "yellow"))
        
        # Fit VECM
        model = VECM(data, k_ar_diff=1, coint_rank=coint_rank, deterministic='ci')
        fitted_model = model.fit()
        
        # Extract results
        results = {
            "model_type": "vecm",
            "variables": variables,
            "coint_rank": int(coint_rank),
            "observations": int(fitted_model.nobs),
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "log_likelihood": float(fitted_model.llf),
            "cointegrating_vectors": {},
            "error_correction_terms": {}
        }
        
        # Extract cointegrating vectors
        if coint_rank > 0:
            beta = fitted_model.beta
            for i in range(coint_rank):
                results["cointegrating_vectors"][f"coint_eq_{i+1}"] = {
                    var: float(beta[j, i]) for j, var in enumerate(variables)
                }
        
        # Store fitted model
        self.fitted_model = fitted_model
        
        # Get AI interpretation
        interpretation = await self._interpret_vecm_results(results)
        results["ai_interpretation"] = interpretation
        
        self.analysis_results["vecm"] = results
        
        print(colored("✅ VECM analysis completed", "green"))
        return results
    
    async def granger_causality_test(self, 
                                   variables: List[str],
                                   maxlag: int = 4) -> Dict[str, Any]:
        """
        Perform Granger causality tests
        
        Args:
            variables: List of variables to test
            maxlag: Maximum lag to test
            
        Returns:
            Granger causality test results
        """
        if len(variables) != 2:
            raise ValueError("Granger causality test requires exactly 2 variables")
        
        data = self.current_data[variables].dropna()
        
        print(colored(f"🔍 Testing Granger causality between {variables[0]} and {variables[1]}", "cyan"))
        
        results = {
            "variables": variables,
            "maxlag": maxlag,
            "tests": {}
        }
        
        # Test both directions
        for i, (cause, effect) in enumerate([(variables[0], variables[1]), (variables[1], variables[0])]):
            test_data = data[[effect, cause]]
            
            try:
                granger_results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                
                # Extract results for each lag
                lag_results = {}
                for lag in range(1, maxlag + 1):
                    if lag in granger_results:
                        ssr_ftest = granger_results[lag][0]['ssr_ftest']
                        lag_results[f"lag_{lag}"] = {
                            "f_statistic": float(ssr_ftest[0]),
                            "p_value": float(ssr_ftest[1]),
                            "causality": "Yes" if ssr_ftest[1] < 0.05 else "No"
                        }
                
                results["tests"][f"{cause}_causes_{effect}"] = lag_results
                
                # Overall assessment (using lag with minimum p-value)
                min_p_value = min([lag_results[f"lag_{lag}"]["p_value"] for lag in range(1, maxlag + 1)])
                overall_causality = "Yes" if min_p_value < 0.05 else "No"
                
                print(colored(f"  {cause} → {effect}: {overall_causality} (min p-value: {min_p_value:.4f})", "blue"))
                
            except Exception as e:
                print(colored(f"  Error testing {cause} → {effect}: {str(e)}", "red"))
                results["tests"][f"{cause}_causes_{effect}"] = {"error": str(e)}
        
        # Get AI interpretation
        interpretation = await self._interpret_granger_results(results)
        results["ai_interpretation"] = interpretation
        
        self.analysis_results["granger_causality"] = results
        return results
    
    async def seasonal_decomposition(self, 
                                   variable: str,
                                   model: str = "additive",
                                   period: int = None) -> Dict[str, Any]:
        """
        Perform seasonal decomposition
        
        Args:
            variable: Variable to decompose
            model: "additive" or "multiplicative"
            period: Seasonal period (auto-detected if None)
            
        Returns:
            Decomposition results
        """
        if variable not in self.current_data.columns:
            raise ValueError(f"Variable {variable} not found in data")
        
        series = self.current_data[variable].dropna()
        
        print(colored(f"🔍 Performing seasonal decomposition of {variable}", "cyan"))
        
        try:
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            results = {
                "variable": variable,
                "model": model,
                "period": int(decomposition.period) if hasattr(decomposition, 'period') else period,
                "components": {
                    "trend": decomposition.trend.dropna().to_dict(),
                    "seasonal": decomposition.seasonal.to_dict(),
                    "residual": decomposition.resid.dropna().to_dict()
                }
            }
            
            # Calculate component statistics
            for component in ["trend", "seasonal", "residual"]:
                comp_data = getattr(decomposition, component).dropna()
                results[f"{component}_stats"] = {
                    "mean": float(comp_data.mean()),
                    "std": float(comp_data.std()),
                    "min": float(comp_data.min()),
                    "max": float(comp_data.max())
                }
            
            # Get AI interpretation
            interpretation = await self._interpret_decomposition_results(results)
            results["ai_interpretation"] = interpretation
            
            self.analysis_results["seasonal_decomposition"] = results
            
            print(colored("✅ Seasonal decomposition completed", "green"))
            return results
            
        except Exception as e:
            error_msg = f"Error in seasonal decomposition: {str(e)}"
            print(colored(error_msg, "red"))
            raise
    
    async def forecast(self, 
                      steps: int = 10,
                      confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts using fitted model
        
        Args:
            steps: Number of periods to forecast
            confidence_interval: Confidence level for prediction intervals
            
        Returns:
            Forecast results
        """
        if not hasattr(self, 'fitted_model'):
            raise ValueError("No fitted model available. Run analysis first.")
        
        print(colored(f"🔮 Generating {steps}-step forecast...", "cyan"))
        
        try:
            forecast_result = self.fitted_model.forecast(steps=steps)
            
            if hasattr(self.fitted_model, 'get_prediction'):
                # Get prediction intervals for ARIMA
                prediction = self.fitted_model.get_prediction(
                    start=len(self.fitted_model.fittedvalues),
                    end=len(self.fitted_model.fittedvalues) + steps - 1,
                    dynamic=False
                )
                
                forecast_values = prediction.predicted_mean
                conf_int = prediction.conf_int(alpha=1-confidence_interval)
                
                results = {
                    "steps": steps,
                    "confidence_level": confidence_interval,
                    "forecasts": forecast_values.to_dict(),
                    "lower_bound": conf_int.iloc[:, 0].to_dict(),
                    "upper_bound": conf_int.iloc[:, 1].to_dict()
                }
            else:
                # For VAR/VECM models
                if isinstance(forecast_result, np.ndarray):
                    if len(forecast_result.shape) == 2:
                        # Multi-variable forecast
                        var_names = self.analysis_results.get("var", {}).get("variables", [f"var_{i}" for i in range(forecast_result.shape[1])])
                        results = {
                            "steps": steps,
                            "forecasts": {}
                        }
                        for i, var in enumerate(var_names):
                            results["forecasts"][var] = forecast_result[:, i].tolist()
                    else:
                        # Single variable forecast
                        results = {
                            "steps": steps,
                            "forecasts": forecast_result.tolist()
                        }
                else:
                    results = {
                        "steps": steps,
                        "forecasts": forecast_result
                    }
            
            # Get AI interpretation
            interpretation = await self._interpret_forecast_results(results)
            results["ai_interpretation"] = interpretation
            
            self.analysis_results["forecast"] = results
            
            print(colored("✅ Forecast completed", "green"))
            return results
            
        except Exception as e:
            error_msg = f"Error generating forecast: {str(e)}"
            print(colored(error_msg, "red"))
            raise
    
    # AI Interpretation Methods
    async def _interpret_stationarity_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of stationarity test results"""
        try:
            system_prompt = """You are an expert time series econometrician. Analyze the stationarity test results and provide:
            1. Assessment of stationarity for each variable
            2. Implications for time series modeling
            3. Recommendations for data transformation if needed
            4. Suggested modeling approaches based on stationarity
            
            Focus on practical econometric implications."""
            
            user_prompt = f"Stationarity Test Results:\n{self._format_stationarity_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating stationarity interpretation"
    
    async def _interpret_arima_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of ARIMA results"""
        try:
            system_prompt = """You are an expert time series econometrician. Analyze the ARIMA results and provide:
            1. Model adequacy assessment
            2. Coefficient interpretation
            3. Residual diagnostic evaluation
            4. Forecasting implications
            5. Model improvement suggestions
            
            Focus on practical time series modeling insights."""
            
            user_prompt = f"ARIMA Results:\n{self._format_arima_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating ARIMA interpretation"
    
    async def _interpret_var_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of VAR results"""
        try:
            system_prompt = """You are an expert time series econometrician. Analyze the VAR results and provide:
            1. Model specification assessment
            2. Dynamic relationships between variables
            3. Statistical significance evaluation
            4. Economic interpretation of relationships
            5. Suggestions for further analysis (impulse responses, etc.)
            
            Focus on multivariate time series insights."""
            
            user_prompt = f"VAR Results:\n{self._format_var_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating VAR interpretation"
    
    async def _interpret_vecm_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of VECM results"""
        try:
            system_prompt = """You are an expert time series econometrician. Analyze the VECM results and provide:
            1. Cointegration assessment
            2. Long-run equilibrium relationships
            3. Error correction mechanism evaluation
            4. Economic interpretation of cointegrating vectors
            5. Model adequacy and suggestions
            
            Focus on cointegration and error correction insights."""
            
            user_prompt = f"VECM Results:\n{self._format_vecm_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating VECM interpretation"
    
    async def _interpret_granger_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of Granger causality results"""
        try:
            system_prompt = """You are an expert econometrician. Analyze the Granger causality results and provide:
            1. Causality relationships assessment
            2. Economic interpretation of causal directions
            3. Statistical significance evaluation
            4. Implications for policy and forecasting
            5. Limitations and caveats of Granger causality
            
            Focus on causal inference in time series."""
            
            user_prompt = f"Granger Causality Results:\n{self._format_granger_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating Granger causality interpretation"
    
    async def _interpret_decomposition_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of seasonal decomposition results"""
        try:
            system_prompt = """You are an expert time series analyst. Analyze the seasonal decomposition results and provide:
            1. Trend component analysis
            2. Seasonal pattern assessment
            3. Residual component evaluation
            4. Overall time series characteristics
            5. Modeling implications
            
            Focus on understanding time series components."""
            
            user_prompt = f"Seasonal Decomposition Results:\n{self._format_decomposition_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating decomposition interpretation"
    
    async def _interpret_forecast_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of forecast results"""
        try:
            system_prompt = """You are an expert time series forecaster. Analyze the forecast results and provide:
            1. Forecast accuracy assessment
            2. Uncertainty quantification
            3. Economic interpretation of forecasts
            4. Risk assessment and confidence intervals
            5. Recommendations for forecast usage
            
            Focus on practical forecasting insights."""
            
            user_prompt = f"Forecast Results:\n{self._format_forecast_for_ai(results)}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self._call_openai_api(messages)
        except:
            return "Error generating forecast interpretation"
    
    # Formatting methods for AI interpretation
    def _format_stationarity_for_ai(self, results: Dict[str, Any]) -> str:
        """Format stationarity results for AI"""
        formatted = []
        for var, test_results in results.items():
            if "error" not in test_results:
                formatted.append(f"{var}: {test_results.get('overall', 'Unknown')}")
        return "\n".join(formatted)
    
    def _format_arima_for_ai(self, results: Dict[str, Any]) -> str:
        """Format ARIMA results for AI"""
        return f"Order: {results['order']}, AIC: {results['aic']:.2f}, Residual diagnostics: {results['residual_diagnostics']}"
    
    def _format_var_for_ai(self, results: Dict[str, Any]) -> str:
        """Format VAR results for AI"""
        return f"Variables: {results['variables']}, Lag order: {results['lag_order']}, AIC: {results['aic']:.2f}"
    
    def _format_vecm_for_ai(self, results: Dict[str, Any]) -> str:
        """Format VECM results for AI"""
        return f"Variables: {results['variables']}, Cointegrating rank: {results['coint_rank']}, AIC: {results['aic']:.2f}"
    
    def _format_granger_for_ai(self, results: Dict[str, Any]) -> str:
        """Format Granger causality results for AI"""
        formatted = []
        for test_name, test_results in results['tests'].items():
            if "error" not in test_results:
                min_p = min([lag_res['p_value'] for lag_res in test_results.values() if isinstance(lag_res, dict)])
                formatted.append(f"{test_name}: min p-value = {min_p:.4f}")
        return "\n".join(formatted)
    
    def _format_decomposition_for_ai(self, results: Dict[str, Any]) -> str:
        """Format decomposition results for AI"""
        return f"Variable: {results['variable']}, Model: {results['model']}, Period: {results.get('period', 'auto')}"
    
    def _format_forecast_for_ai(self, results: Dict[str, Any]) -> str:
        """Format forecast results for AI"""
        return f"Forecast steps: {results['steps']}, Confidence level: {results.get('confidence_level', 'N/A')}"
