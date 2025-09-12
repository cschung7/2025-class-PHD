"""
Regression Analysis Agent
Specialized in cross-sectional regression analysis using statsmodels
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from termcolor import colored

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy import stats

from core.base_agent import BaseEconometricAgent


class RegressionAgent(BaseEconometricAgent):
    """
    Specialized agent for regression analysis
    """
    
    def __init__(self):
        super().__init__(
            agent_name="RegressionAnalyst",
            specialization="Cross-sectional and Multiple Regression Analysis"
        )
        
        self.supported_models = {
            "ols": "Ordinary Least Squares",
            "wls": "Weighted Least Squares", 
            "gls": "Generalized Least Squares",
            "logit": "Logistic Regression",
            "probit": "Probit Regression",
            "poisson": "Poisson Regression"
        }
        
        self.diagnostic_tests = [
            "heteroscedasticity",
            "autocorrelation", 
            "normality",
            "multicollinearity",
            "outliers"
        ]
    
    def get_available_methods(self) -> List[str]:
        """Return available regression methods"""
        return list(self.supported_models.keys()) + ["diagnostics", "model_selection", "robust_se"]
    
    async def perform_analysis(self, 
                             dependent_var: str,
                             independent_vars: List[str],
                             model_type: str = "ols",
                             formula: Optional[str] = None,
                             robust_se: bool = False,
                             run_diagnostics: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Perform regression analysis
        
        Args:
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            model_type: Type of regression model
            formula: Optional R-style formula
            robust_se: Use robust standard errors
            run_diagnostics: Run diagnostic tests
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary containing regression results
        """
        if self.current_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        print(colored(f"🔍 Running {model_type.upper()} regression analysis...", "cyan"))
        
        try:
            # Prepare data
            if formula:
                # Use formula interface
                if model_type == "ols":
                    model = smf.ols(formula, data=self.current_data)
                elif model_type == "logit":
                    model = smf.logit(formula, data=self.current_data)
                elif model_type == "probit":
                    model = smf.probit(formula, data=self.current_data)
                elif model_type == "poisson":
                    model = smf.poisson(formula, data=self.current_data)
                else:
                    raise ValueError(f"Formula interface not supported for {model_type}")
            else:
                # Use array interface
                y = self.current_data[dependent_var]
                X = self.current_data[independent_vars]
                X = sm.add_constant(X)  # Add intercept
                
                if model_type == "ols":
                    model = sm.OLS(y, X)
                elif model_type == "wls":
                    weights = kwargs.get('weights', None)
                    if weights is None:
                        raise ValueError("WLS requires weights parameter")
                    model = sm.WLS(y, X, weights=self.current_data[weights])
                elif model_type == "gls":
                    model = sm.GLS(y, X)
                elif model_type == "logit":
                    model = sm.Logit(y, X)
                elif model_type == "probit":
                    model = sm.Probit(y, X)
                elif model_type == "poisson":
                    model = sm.Poisson(y, X)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            # Fit model
            if robust_se and model_type in ["ols", "wls", "gls"]:
                results = model.fit(cov_type='HC3')  # Robust standard errors
            else:
                results = model.fit()
            
            print(colored("✓ Model fitted successfully", "green"))
            
            # Extract results
            regression_results = {
                "model_type": model_type,
                "dependent_variable": dependent_var,
                "independent_variables": independent_vars if not formula else "from_formula",
                "formula": formula,
                "observations": int(results.nobs),
                "r_squared": float(results.rsquared) if hasattr(results, 'rsquared') else None,
                "adj_r_squared": float(results.rsquared_adj) if hasattr(results, 'rsquared_adj') else None,
                "f_statistic": float(results.fvalue) if hasattr(results, 'fvalue') else None,
                "f_pvalue": float(results.f_pvalue) if hasattr(results, 'f_pvalue') else None,
                "aic": float(results.aic),
                "bic": float(results.bic),
                "log_likelihood": float(results.llf),
                "coefficients": {},
                "robust_se": robust_se
            }
            
            # Extract coefficient information
            for i, param in enumerate(results.params.index):
                regression_results["coefficients"][param] = {
                    "coefficient": float(results.params[param]),
                    "std_error": float(results.bse[param]),
                    "t_statistic": float(results.tvalues[param]),
                    "p_value": float(results.pvalues[param]),
                    "conf_int_lower": float(results.conf_int().iloc[i, 0]),
                    "conf_int_upper": float(results.conf_int().iloc[i, 1])
                }
            
            # Store fitted model for diagnostics
            self.fitted_model = results
            
            # Run diagnostic tests if requested
            if run_diagnostics and model_type in ["ols", "wls", "gls"]:
                print(colored("🔬 Running diagnostic tests...", "yellow"))
                diagnostics = await self._run_diagnostics(results, y, X if not formula else None)
                regression_results["diagnostics"] = diagnostics
            
            # Get AI interpretation
            interpretation = await self._interpret_results(regression_results)
            regression_results["ai_interpretation"] = interpretation
            
            self.analysis_results["regression"] = regression_results
            
            print(colored("✅ Regression analysis completed", "green"))
            return regression_results
            
        except Exception as e:
            error_msg = f"Error in regression analysis: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def _run_diagnostics(self, results, y, X) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic tests
        
        Args:
            results: Fitted regression results
            y: Dependent variable
            X: Independent variables matrix
            
        Returns:
            Dictionary containing diagnostic test results
        """
        diagnostics = {}
        
        try:
            # Heteroscedasticity tests
            if X is not None:
                # Breusch-Pagan test
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(results.resid, X)
                diagnostics["breusch_pagan"] = {
                    "statistic": float(bp_stat),
                    "p_value": float(bp_pvalue),
                    "interpretation": "Homoscedasticity" if bp_pvalue > 0.05 else "Heteroscedasticity detected"
                }
                
                # White test
                white_stat, white_pvalue, _, _ = het_white(results.resid, X)
                diagnostics["white_test"] = {
                    "statistic": float(white_stat),
                    "p_value": float(white_pvalue),
                    "interpretation": "Homoscedasticity" if white_pvalue > 0.05 else "Heteroscedasticity detected"
                }
            
            # Durbin-Watson test for autocorrelation
            dw_stat = durbin_watson(results.resid)
            diagnostics["durbin_watson"] = {
                "statistic": float(dw_stat),
                "interpretation": self._interpret_durbin_watson(dw_stat)
            }
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = jarque_bera(results.resid)
            diagnostics["jarque_bera"] = {
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "interpretation": "Normal residuals" if jb_pvalue > 0.05 else "Non-normal residuals"
            }
            
            # Multicollinearity (VIF)
            if X is not None and X.shape[1] > 1:
                vif_data = []
                for i in range(X.shape[1]):
                    if X.columns[i] != 'const':  # Skip constant term
                        vif = variance_inflation_factor(X.values, i)
                        vif_data.append({
                            "variable": X.columns[i],
                            "vif": float(vif),
                            "interpretation": "Low" if vif < 5 else "Moderate" if vif < 10 else "High"
                        })
                
                diagnostics["multicollinearity"] = vif_data
            
            print(colored("✓ Diagnostic tests completed", "green"))
            
        except Exception as e:
            diagnostics["error"] = f"Error running diagnostics: {str(e)}"
            print(colored(f"⚠️  Warning: Some diagnostic tests failed: {str(e)}", "yellow"))
        
        return diagnostics
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic"""
        if dw_stat < 1.5:
            return "Positive autocorrelation likely"
        elif dw_stat > 2.5:
            return "Negative autocorrelation likely"
        else:
            return "No significant autocorrelation"
    
    async def _interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Use OpenAI to interpret regression results
        
        Args:
            results: Regression results dictionary
            
        Returns:
            AI interpretation of results
        """
        try:
            system_prompt = """You are an expert econometrician. Analyze the regression results and provide:
            1. Overall model assessment (fit, significance)
            2. Interpretation of key coefficients
            3. Statistical significance assessment
            4. Potential concerns or issues
            5. Practical implications
            
            Be specific about economic interpretation and statistical validity."""
            
            user_prompt = f"""
            Regression Results Summary:
            - Model: {results['model_type'].upper()}
            - Observations: {results['observations']}
            - R-squared: {results.get('r_squared', 'N/A')}
            - F-statistic p-value: {results.get('f_pvalue', 'N/A')}
            
            Coefficients:
            {self._format_coefficients_for_ai(results['coefficients'])}
            
            {f"Diagnostic Tests: {self._format_diagnostics_for_ai(results.get('diagnostics', {}))}" if 'diagnostics' in results else ""}
            
            Please provide a comprehensive interpretation of these results.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await self._call_openai_api(messages)
            return interpretation
            
        except Exception as e:
            return f"Error generating interpretation: {str(e)}"
    
    def _format_coefficients_for_ai(self, coefficients: Dict) -> str:
        """Format coefficients for AI interpretation"""
        formatted = []
        for var, stats in coefficients.items():
            significance = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
            formatted.append(f"{var}: {stats['coefficient']:.4f}{significance} (SE: {stats['std_error']:.4f}, p: {stats['p_value']:.4f})")
        return "\n".join(formatted)
    
    def _format_diagnostics_for_ai(self, diagnostics: Dict) -> str:
        """Format diagnostic results for AI interpretation"""
        formatted = []
        for test, result in diagnostics.items():
            if isinstance(result, dict) and 'interpretation' in result:
                formatted.append(f"{test}: {result['interpretation']}")
            elif isinstance(result, list):  # VIF results
                vif_summary = [f"{item['variable']}: {item['vif']:.2f} ({item['interpretation']})" for item in result]
                formatted.append(f"VIF: {', '.join(vif_summary)}")
        return "\n".join(formatted)
    
    async def model_selection(self, 
                            dependent_var: str,
                            candidate_vars: List[str],
                            selection_method: str = "forward") -> Dict[str, Any]:
        """
        Perform automated model selection
        
        Args:
            dependent_var: Dependent variable name
            candidate_vars: List of candidate independent variables
            selection_method: Selection method ("forward", "backward", "stepwise")
            
        Returns:
            Model selection results
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        print(colored(f"🎯 Performing {selection_method} model selection...", "cyan"))
        
        try:
            results = {}
            
            if selection_method == "forward":
                results = await self._forward_selection(dependent_var, candidate_vars)
            elif selection_method == "backward":
                results = await self._backward_selection(dependent_var, candidate_vars)
            elif selection_method == "stepwise":
                results = await self._stepwise_selection(dependent_var, candidate_vars)
            else:
                raise ValueError(f"Unsupported selection method: {selection_method}")
            
            # Get AI interpretation of model selection
            interpretation = await self._interpret_model_selection(results)
            results["ai_interpretation"] = interpretation
            
            return results
            
        except Exception as e:
            error_msg = f"Error in model selection: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def _forward_selection(self, dependent_var: str, candidate_vars: List[str]) -> Dict[str, Any]:
        """Implement forward selection algorithm"""
        selected_vars = []
        remaining_vars = candidate_vars.copy()
        selection_history = []
        
        y = self.current_data[dependent_var]
        
        while remaining_vars:
            best_var = None
            best_aic = float('inf')
            
            for var in remaining_vars:
                current_vars = selected_vars + [var]
                X = sm.add_constant(self.current_data[current_vars])
                
                try:
                    model = sm.OLS(y, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_var = var
                except:
                    continue
            
            if best_var is None:
                break
            
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
            
            selection_history.append({
                "step": len(selected_vars),
                "added_variable": best_var,
                "variables": selected_vars.copy(),
                "aic": best_aic
            })
            
            print(colored(f"  Step {len(selected_vars)}: Added {best_var} (AIC: {best_aic:.2f})", "blue"))
        
        return {
            "method": "forward",
            "final_variables": selected_vars,
            "selection_history": selection_history
        }
    
    async def _backward_selection(self, dependent_var: str, candidate_vars: List[str]) -> Dict[str, Any]:
        """Implement backward selection algorithm"""
        selected_vars = candidate_vars.copy()
        selection_history = []
        
        y = self.current_data[dependent_var]
        
        while len(selected_vars) > 1:
            worst_var = None
            best_aic = float('inf')
            
            for var in selected_vars:
                current_vars = [v for v in selected_vars if v != var]
                X = sm.add_constant(self.current_data[current_vars])
                
                try:
                    model = sm.OLS(y, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        worst_var = var
                except:
                    continue
            
            if worst_var is None:
                break
            
            selected_vars.remove(worst_var)
            
            selection_history.append({
                "step": len(candidate_vars) - len(selected_vars),
                "removed_variable": worst_var,
                "variables": selected_vars.copy(),
                "aic": best_aic
            })
            
            print(colored(f"  Step {len(selection_history)}: Removed {worst_var} (AIC: {best_aic:.2f})", "blue"))
        
        return {
            "method": "backward",
            "final_variables": selected_vars,
            "selection_history": selection_history
        }
    
    async def _stepwise_selection(self, dependent_var: str, candidate_vars: List[str]) -> Dict[str, Any]:
        """Implement stepwise selection algorithm"""
        # Simplified stepwise: forward selection with backward checking
        forward_result = await self._forward_selection(dependent_var, candidate_vars)
        
        # Could implement more sophisticated stepwise logic here
        return {
            "method": "stepwise",
            "final_variables": forward_result["final_variables"],
            "selection_history": forward_result["selection_history"]
        }
    
    async def _interpret_model_selection(self, results: Dict[str, Any]) -> str:
        """Get AI interpretation of model selection results"""
        try:
            system_prompt = """You are an expert econometrician. Analyze the model selection results and provide:
            1. Assessment of the selected model
            2. Interpretation of the selection process
            3. Potential concerns about the final model
            4. Recommendations for further analysis
            
            Focus on econometric best practices and model validity."""
            
            user_prompt = f"""
            Model Selection Results:
            - Method: {results['method']}
            - Final Variables: {results['final_variables']}
            - Selection Steps: {len(results['selection_history'])}
            
            Selection History:
            {self._format_selection_history(results['selection_history'])}
            
            Please interpret these model selection results.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await self._call_openai_api(messages)
            return interpretation
            
        except Exception as e:
            return f"Error generating model selection interpretation: {str(e)}"
    
    def _format_selection_history(self, history: List[Dict]) -> str:
        """Format selection history for AI interpretation"""
        formatted = []
        for step in history:
            if "added_variable" in step:
                formatted.append(f"Step {step['step']}: Added {step['added_variable']} (AIC: {step['aic']:.2f})")
            elif "removed_variable" in step:
                formatted.append(f"Step {step['step']}: Removed {step['removed_variable']} (AIC: {step['aic']:.2f})")
        return "\n".join(formatted)
    
    def print_summary(self):
        """Print a summary of the latest regression analysis"""
        if "regression" not in self.analysis_results:
            print(colored("No regression analysis results available", "yellow"))
            return
        
        results = self.analysis_results["regression"]
        
        print(colored(f"\n📊 {results['model_type'].upper()} Regression Summary", "cyan"))
        print(colored("=" * 50, "cyan"))
        
        print(f"Dependent Variable: {results['dependent_variable']}")
        print(f"Observations: {results['observations']}")
        
        if results.get('r_squared'):
            print(f"R-squared: {results['r_squared']:.4f}")
            print(f"Adj. R-squared: {results['adj_r_squared']:.4f}")
        
        if results.get('f_pvalue'):
            print(f"F-statistic p-value: {results['f_pvalue']:.4f}")
        
        print(f"AIC: {results['aic']:.2f}")
        print(f"BIC: {results['bic']:.2f}")
        
        print(colored("\nCoefficients:", "yellow"))
        for var, stats in results['coefficients'].items():
            significance = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
            print(f"  {var}: {stats['coefficient']:.4f}{significance} (SE: {stats['std_error']:.4f})")
        
        if "diagnostics" in results:
            print(colored("\nDiagnostic Tests:", "yellow"))
            diagnostics = results["diagnostics"]
            
            if "breusch_pagan" in diagnostics:
                print(f"  Breusch-Pagan: {diagnostics['breusch_pagan']['interpretation']}")
            
            if "durbin_watson" in diagnostics:
                print(f"  Durbin-Watson: {diagnostics['durbin_watson']['interpretation']}")
            
            if "jarque_bera" in diagnostics:
                print(f"  Jarque-Bera: {diagnostics['jarque_bera']['interpretation']}")
        
        print(colored("\n" + "=" * 50, "cyan"))

