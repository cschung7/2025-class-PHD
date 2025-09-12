"""
Panel Data Analysis Agent
Specialized in panel data econometrics using statsmodels
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from termcolor import colored

import statsmodels.api as sm
from statsmodels.stats.api import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from linearmodels import PanelOLS, RandomEffects, FirstDifferenceOLS, PooledOLS
from linearmodels.tests import hausman
import warnings
warnings.filterwarnings('ignore')

from core.base_agent import BaseEconometricAgent


class PanelDataAgent(BaseEconometricAgent):
    """
    Specialized agent for panel data econometric analysis
    """
    
    def __init__(self):
        super().__init__(
            agent_name="PanelDataAnalyst", 
            specialization="Panel Data Econometrics"
        )
        
        self.supported_models = {
            "pooled_ols": "Pooled OLS",
            "fixed_effects": "Fixed Effects (Within)",
            "random_effects": "Random Effects (GLS)",
            "first_difference": "First Difference",
            "between": "Between Estimator"
        }
        
        self.diagnostic_tests = [
            "hausman_test",
            "breusch_pagan",
            "fixed_effects_test",
            "time_effects_test"
        ]
        
        self.entity_col = None
        self.time_col = None
    
    def get_available_methods(self) -> List[str]:
        """Return available panel data methods"""
        return (list(self.supported_models.keys()) + 
                ["hausman_test", "set_panel_structure", "model_comparison"])
    
    def set_panel_structure(self, entity_col: str, time_col: str):
        """
        Set panel data structure
        
        Args:
            entity_col: Column name for entity (individual) identifier
            time_col: Column name for time identifier
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        if entity_col not in self.current_data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found in data")
        
        if time_col not in self.current_data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
        
        self.entity_col = entity_col
        self.time_col = time_col
        
        # Create MultiIndex for panel data
        self.current_data = self.current_data.set_index([entity_col, time_col])
        
        n_entities = len(self.current_data.index.get_level_values(0).unique())
        n_time_periods = len(self.current_data.index.get_level_values(1).unique())
        
        print(colored(f"✓ Panel structure set: {n_entities} entities, {n_time_periods} time periods", "green"))
        print(colored(f"  Total observations: {len(self.current_data)}", "blue"))
        
        self.logger.info(f"Panel structure set: {entity_col}, {time_col}")
    
    async def perform_analysis(self,
                             dependent_var: str,
                             independent_vars: List[str],
                             model_type: str = "fixed_effects",
                             entity_effects: bool = True,
                             time_effects: bool = False,
                             cluster_entity: bool = True,
                             cluster_time: bool = False,
                             **kwargs) -> Dict[str, Any]:
        """
        Perform panel data analysis
        
        Args:
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            model_type: Type of panel data model
            entity_effects: Include entity fixed effects
            time_effects: Include time fixed effects
            cluster_entity: Cluster standard errors by entity
            cluster_time: Cluster standard errors by time
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary containing panel data results
        """
        if self.current_data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        if self.entity_col is None or self.time_col is None:
            raise ValueError("Panel structure not set. Use set_panel_structure() first.")
        
        print(colored(f"🔍 Running {model_type.upper()} panel data analysis...", "cyan"))
        
        try:
            # Prepare data
            y = self.current_data[dependent_var]
            X = self.current_data[independent_vars]
            
            # Remove missing values
            data_clean = pd.concat([y, X], axis=1).dropna()
            y_clean = data_clean[dependent_var]
            X_clean = data_clean[independent_vars]
            
            # Fit model based on type
            if model_type == "pooled_ols":
                model = PooledOLS(y_clean, X_clean)
            elif model_type == "fixed_effects":
                model = PanelOLS(y_clean, X_clean, 
                               entity_effects=entity_effects,
                               time_effects=time_effects)
            elif model_type == "random_effects":
                model = RandomEffects(y_clean, X_clean)
            elif model_type == "first_difference":
                model = FirstDifferenceOLS(y_clean, X_clean)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Set clustering options
            cluster_args = {}
            if cluster_entity and cluster_time:
                cluster_args['cov_type'] = 'clustered'
                cluster_args['clusters'] = data_clean.index
            elif cluster_entity:
                cluster_args['cov_type'] = 'clustered'
                cluster_args['cluster_entity'] = True
            elif cluster_time:
                cluster_args['cov_type'] = 'clustered'
                cluster_args['cluster_time'] = True
            else:
                cluster_args['cov_type'] = 'robust'
            
            # Fit model
            results = model.fit(**cluster_args)
            
            print(colored("✓ Model fitted successfully", "green"))
            
            # Extract results
            panel_results = {
                "model_type": model_type,
                "dependent_variable": dependent_var,
                "independent_variables": independent_vars,
                "entity_effects": entity_effects,
                "time_effects": time_effects,
                "observations": int(results.nobs),
                "entities": len(y_clean.index.get_level_values(0).unique()),
                "time_periods": len(y_clean.index.get_level_values(1).unique()),
                "r_squared": float(results.rsquared),
                "r_squared_within": float(results.rsquared_within) if hasattr(results, 'rsquared_within') else None,
                "r_squared_between": float(results.rsquared_between) if hasattr(results, 'rsquared_between') else None,
                "r_squared_overall": float(results.rsquared_overall) if hasattr(results, 'rsquared_overall') else None,
                "f_statistic": float(results.f_statistic.stat),
                "f_pvalue": float(results.f_statistic.pval),
                "coefficients": {},
                "clustering": cluster_args
            }
            
            # Extract coefficient information
            for param in results.params.index:
                panel_results["coefficients"][param] = {
                    "coefficient": float(results.params[param]),
                    "std_error": float(results.std_errors[param]),
                    "t_statistic": float(results.tstats[param]),
                    "p_value": float(results.pvalues[param]),
                    "conf_int_lower": float(results.conf_int().loc[param, 'lower']),
                    "conf_int_upper": float(results.conf_int().loc[param, 'upper'])
                }
            
            # Store fitted model
            self.fitted_model = results
            
            # Get AI interpretation
            interpretation = await self._interpret_panel_results(panel_results)
            panel_results["ai_interpretation"] = interpretation
            
            self.analysis_results["panel_data"] = panel_results
            
            print(colored("✅ Panel data analysis completed", "green"))
            return panel_results
            
        except Exception as e:
            error_msg = f"Error in panel data analysis: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def hausman_test(self, 
                          dependent_var: str,
                          independent_vars: List[str]) -> Dict[str, Any]:
        """
        Perform Hausman test to choose between fixed and random effects
        
        Args:
            dependent_var: Dependent variable name
            independent_vars: Independent variable names
            
        Returns:
            Hausman test results
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        if self.entity_col is None or self.time_col is None:
            raise ValueError("Panel structure not set")
        
        print(colored("🔍 Performing Hausman test (Fixed vs Random Effects)...", "cyan"))
        
        try:
            # Prepare data
            y = self.current_data[dependent_var]
            X = self.current_data[independent_vars]
            
            # Remove missing values
            data_clean = pd.concat([y, X], axis=1).dropna()
            y_clean = data_clean[dependent_var]
            X_clean = data_clean[independent_vars]
            
            # Fit both models
            fe_model = PanelOLS(y_clean, X_clean, entity_effects=True)
            fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)
            
            re_model = RandomEffects(y_clean, X_clean)
            re_results = re_model.fit(cov_type='clustered', cluster_entity=True)
            
            # Perform Hausman test
            hausman_stat = hausman(fe_results, re_results)
            
            results = {
                "test": "hausman",
                "statistic": float(hausman_stat.stat),
                "p_value": float(hausman_stat.pval),
                "degrees_of_freedom": int(hausman_stat.df),
                "critical_value_5pct": 3.84 if hausman_stat.df == 1 else None,
                "interpretation": "Fixed Effects preferred" if hausman_stat.pval < 0.05 else "Random Effects preferred",
                "recommendation": "Use Fixed Effects model" if hausman_stat.pval < 0.05 else "Use Random Effects model"
            }
            
            print(colored(f"  Hausman statistic: {hausman_stat.stat:.4f}", "blue"))
            print(colored(f"  P-value: {hausman_stat.pval:.4f}", "blue"))
            print(colored(f"  Recommendation: {results['recommendation']}", "green"))
            
            # Get AI interpretation
            interpretation = await self._interpret_hausman_test(results)
            results["ai_interpretation"] = interpretation
            
            self.analysis_results["hausman_test"] = results
            return results
            
        except Exception as e:
            error_msg = f"Error in Hausman test: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    async def model_comparison(self,
                             dependent_var: str,
                             independent_vars: List[str]) -> Dict[str, Any]:
        """
        Compare different panel data models
        
        Args:
            dependent_var: Dependent variable name
            independent_vars: Independent variable names
            
        Returns:
            Model comparison results
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        print(colored("🔍 Comparing panel data models...", "cyan"))
        
        try:
            # Prepare data
            y = self.current_data[dependent_var]
            X = self.current_data[independent_vars]
            
            # Remove missing values
            data_clean = pd.concat([y, X], axis=1).dropna()
            y_clean = data_clean[dependent_var]
            X_clean = data_clean[independent_vars]
            
            models_to_compare = ["pooled_ols", "fixed_effects", "random_effects"]
            comparison_results = {}
            
            for model_type in models_to_compare:
                print(colored(f"  Fitting {model_type}...", "yellow"))
                
                try:
                    if model_type == "pooled_ols":
                        model = PooledOLS(y_clean, X_clean)
                        results = model.fit(cov_type='robust')
                    elif model_type == "fixed_effects":
                        model = PanelOLS(y_clean, X_clean, entity_effects=True)
                        results = model.fit(cov_type='clustered', cluster_entity=True)
                    elif model_type == "random_effects":
                        model = RandomEffects(y_clean, X_clean)
                        results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    comparison_results[model_type] = {
                        "r_squared": float(results.rsquared),
                        "r_squared_within": float(results.rsquared_within) if hasattr(results, 'rsquared_within') else None,
                        "r_squared_between": float(results.rsquared_between) if hasattr(results, 'rsquared_between') else None,
                        "f_statistic": float(results.f_statistic.stat),
                        "f_pvalue": float(results.f_statistic.pval),
                        "observations": int(results.nobs),
                        "loglik": float(results.loglik) if hasattr(results, 'loglik') else None
                    }
                    
                except Exception as e:
                    comparison_results[model_type] = {"error": str(e)}
            
            # Perform Hausman test if both FE and RE were successful
            hausman_result = None
            if ("fixed_effects" in comparison_results and "random_effects" in comparison_results and
                "error" not in comparison_results["fixed_effects"] and "error" not in comparison_results["random_effects"]):
                
                try:
                    hausman_result = await self.hausman_test(dependent_var, independent_vars)
                except:
                    hausman_result = {"error": "Hausman test failed"}
            
            results = {
                "dependent_variable": dependent_var,
                "independent_variables": independent_vars,
                "model_comparison": comparison_results,
                "hausman_test": hausman_result
            }
            
            # Get AI interpretation
            interpretation = await self._interpret_model_comparison(results)
            results["ai_interpretation"] = interpretation
            
            self.analysis_results["model_comparison"] = results
            
            print(colored("✅ Model comparison completed", "green"))
            return results
            
        except Exception as e:
            error_msg = f"Error in model comparison: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    def get_panel_summary(self) -> Dict[str, Any]:
        """
        Get summary of panel data structure
        
        Returns:
            Panel data summary
        """
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        if self.entity_col is None or self.time_col is None:
            return {"error": "Panel structure not set"}
        
        entities = self.current_data.index.get_level_values(0).unique()
        time_periods = self.current_data.index.get_level_values(1).unique()
        
        # Check for balanced panel
        expected_obs = len(entities) * len(time_periods)
        actual_obs = len(self.current_data)
        is_balanced = expected_obs == actual_obs
        
        # Calculate observations per entity
        obs_per_entity = self.current_data.groupby(level=0).size()
        
        summary = {
            "entities": len(entities),
            "time_periods": len(time_periods),
            "total_observations": actual_obs,
            "expected_observations": expected_obs,
            "is_balanced": is_balanced,
            "balance_ratio": actual_obs / expected_obs,
            "min_obs_per_entity": int(obs_per_entity.min()),
            "max_obs_per_entity": int(obs_per_entity.max()),
            "avg_obs_per_entity": float(obs_per_entity.mean()),
            "entity_range": [str(entities.min()), str(entities.max())],
            "time_range": [str(time_periods.min()), str(time_periods.max())]
        }
        
        return summary
    
    async def _interpret_panel_results(self, results: Dict[str, Any]) -> str:
        """AI interpretation of panel data results"""
        try:
            system_prompt = """You are an expert panel data econometrician. Analyze the panel data results and provide:
            1. Model specification assessment
            2. Interpretation of coefficients in panel context
            3. Within vs between vs overall R-squared interpretation
            4. Fixed/random effects implications
            5. Policy and economic interpretation
            6. Recommendations for robustness checks
            
            Focus on panel data specific insights and econometric validity."""
            
            user_prompt = f"""
            Panel Data Results:
            - Model: {results['model_type']}
            - Entities: {results['entities']}, Time periods: {results['time_periods']}
            - Observations: {results['observations']}
            - R-squared: {results['r_squared']:.4f}
            - Within R-squared: {results.get('r_squared_within', 'N/A')}
            - Between R-squared: {results.get('r_squared_between', 'N/A')}
            - F-statistic p-value: {results['f_pvalue']:.4f}
            
            Coefficients:
            {self._format_coefficients_for_ai(results['coefficients'])}
            
            Please provide a comprehensive interpretation of these panel data results.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await self._call_openai_api(messages)
            return interpretation
            
        except Exception as e:
            return f"Error generating panel data interpretation: {str(e)}"
    
    async def _interpret_hausman_test(self, results: Dict[str, Any]) -> str:
        """AI interpretation of Hausman test"""
        try:
            system_prompt = """You are an expert panel data econometrician. Analyze the Hausman test results and provide:
            1. Test interpretation and implications
            2. Fixed vs Random effects choice rationale
            3. Assumptions and limitations
            4. Economic interpretation of the choice
            5. Recommendations for further analysis
            
            Focus on the econometric theory behind the test and practical implications."""
            
            user_prompt = f"""
            Hausman Test Results:
            - Statistic: {results['statistic']:.4f}
            - P-value: {results['p_value']:.4f}
            - Degrees of freedom: {results['degrees_of_freedom']}
            - Interpretation: {results['interpretation']}
            - Recommendation: {results['recommendation']}
            
            Please explain the Hausman test results and their implications.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await self._call_openai_api(messages)
            return interpretation
            
        except Exception as e:
            return f"Error generating Hausman test interpretation: {str(e)}"
    
    async def _interpret_model_comparison(self, results: Dict[str, Any]) -> str:
        """AI interpretation of model comparison"""
        try:
            system_prompt = """You are an expert panel data econometrician. Analyze the model comparison results and provide:
            1. Comparison of model performance
            2. Strengths and weaknesses of each approach
            3. Recommendation for best model choice
            4. Economic interpretation implications
            5. Robustness considerations
            
            Focus on helping choose the most appropriate panel data model."""
            
            comparison_summary = []
            for model, stats in results['model_comparison'].items():
                if "error" not in stats:
                    comparison_summary.append(f"{model}: R² = {stats['r_squared']:.4f}, F p-value = {stats['f_pvalue']:.4f}")
                else:
                    comparison_summary.append(f"{model}: Error - {stats['error']}")
            
            user_prompt = f"""
            Panel Data Model Comparison:
            {chr(10).join(comparison_summary)}
            
            Hausman Test: {results.get('hausman_test', {}).get('recommendation', 'Not available')}
            
            Please analyze these results and recommend the best model choice.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await self._call_openai_api(messages)
            return interpretation
            
        except Exception as e:
            return f"Error generating model comparison interpretation: {str(e)}"
    
    def _format_coefficients_for_ai(self, coefficients: Dict) -> str:
        """Format coefficients for AI interpretation"""
        formatted = []
        for var, stats in coefficients.items():
            significance = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
            formatted.append(f"{var}: {stats['coefficient']:.4f}{significance} (SE: {stats['std_error']:.4f}, p: {stats['p_value']:.4f})")
        return "\n".join(formatted)
    
    def print_summary(self):
        """Print a summary of the latest panel data analysis"""
        if "panel_data" not in self.analysis_results:
            print(colored("No panel data analysis results available", "yellow"))
            return
        
        results = self.analysis_results["panel_data"]
        
        print(colored(f"\n📊 {results['model_type'].upper().replace('_', ' ')} Panel Data Summary", "cyan"))
        print(colored("=" * 60, "cyan"))
        
        print(f"Dependent Variable: {results['dependent_variable']}")
        print(f"Entities: {results['entities']}")
        print(f"Time Periods: {results['time_periods']}")
        print(f"Total Observations: {results['observations']}")
        
        print(f"\nR-squared: {results['r_squared']:.4f}")
        if results.get('r_squared_within'):
            print(f"Within R-squared: {results['r_squared_within']:.4f}")
        if results.get('r_squared_between'):
            print(f"Between R-squared: {results['r_squared_between']:.4f}")
        if results.get('r_squared_overall'):
            print(f"Overall R-squared: {results['r_squared_overall']:.4f}")
        
        print(f"F-statistic p-value: {results['f_pvalue']:.4f}")
        
        print(colored("\nCoefficients:", "yellow"))
        for var, stats in results['coefficients'].items():
            significance = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
            print(f"  {var}: {stats['coefficient']:.4f}{significance} (SE: {stats['std_error']:.4f})")
        
        print(colored("\n" + "=" * 60, "cyan"))

