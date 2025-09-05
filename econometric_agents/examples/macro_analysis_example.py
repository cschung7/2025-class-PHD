"""
Macroeconomic Analysis Example
Specialized example focusing on macroeconomic modeling and interpretation
"""
import asyncio
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.macro_agent import MacroEconomicAgent
from termcolor import colored


async def create_realistic_macro_data():
    """Create realistic macroeconomic data with proper relationships"""
    print(colored("📊 Creating realistic macroeconomic dataset...", "cyan"))
    
    # Monthly data from 1990 to 2023
    dates = pd.date_range(start='1990-01-01', end='2023-12-31', freq='M')
    n_obs = len(dates)
    
    np.random.seed(123)  # For reproducible results
    
    # Create realistic macro variables with proper relationships
    
    # 1. Core inflation (trend + cycles + shocks)
    trend_inflation = 2.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_obs))  # Long-term cycles
    inflation_shocks = np.random.normal(0, 0.8, n_obs)
    core_inflation = trend_inflation + inflation_shocks
    
    # 2. Unemployment rate (NAIRU + cyclical component)
    nairu = 5.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, n_obs))  # Time-varying NAIRU
    unemployment_cycle = 2.0 * np.sin(np.linspace(0, 8*np.pi, n_obs)) + np.random.normal(0, 0.5, n_obs)
    unemployment_rate = nairu + unemployment_cycle
    unemployment_rate = np.clip(unemployment_rate, 2.0, 12.0)
    
    # 3. Phillips Curve relationship (with some nonlinearity)
    phillips_inflation = 3.0 - 0.4 * (unemployment_rate - 5.5) + 0.02 * (unemployment_rate - 5.5)**2
    inflation_expectations = 0.7 * core_inflation + 0.3 * phillips_inflation
    
    # 4. Federal Funds Rate (Taylor Rule + policy shocks)
    natural_rate = 2.0
    inflation_gap = core_inflation - 2.0
    output_gap = -(unemployment_rate - nairu)  # Okun's law approximation
    
    taylor_rate = natural_rate + inflation_gap + 1.5 * inflation_gap + 0.5 * output_gap
    policy_shocks = np.random.normal(0, 0.3, n_obs)
    federal_funds_rate = taylor_rate + policy_shocks
    federal_funds_rate = np.clip(federal_funds_rate, 0.0, 20.0)
    
    # 5. GDP Growth (related to unemployment via Okun's Law)
    potential_growth = 2.5
    gdp_growth = potential_growth - 0.5 * (unemployment_rate - nairu) + np.random.normal(0, 1.0, n_obs)
    
    # 6. Money Supply Growth (related to inflation with lags)
    money_growth_trend = 3.0 + 0.8 * core_inflation + np.random.normal(0, 1.5, n_obs)
    
    # 7. Long-term bond yields (expectations + term premium)
    term_premium = 1.5 + 0.5 * np.sin(np.linspace(0, 6*np.pi, n_obs))
    long_term_yield = inflation_expectations + natural_rate + term_premium + np.random.normal(0, 0.4, n_obs)
    
    # 8. Exchange rate (purchasing power parity + shocks)
    exchange_rate = 100 * np.exp(np.cumsum((core_inflation - 2.0) * 0.01 + np.random.normal(0, 0.02, n_obs)))
    
    # 9. Consumer confidence (related to economic conditions)
    confidence_trend = 100 - 2 * (unemployment_rate - 5.5) + 0.5 * gdp_growth
    consumer_confidence = confidence_trend + np.random.normal(0, 5, n_obs)
    consumer_confidence = np.clip(consumer_confidence, 30, 150)
    
    # 10. Industrial production growth
    industrial_production = gdp_growth * 1.2 + np.random.normal(0, 1.5, n_obs)
    
    # Create DataFrame
    macro_data = pd.DataFrame({
        'date': dates,
        'core_inflation': core_inflation,
        'cpi_inflation': core_inflation + np.random.normal(0, 0.3, n_obs),  # CPI slightly more volatile
        'unemployment_rate': unemployment_rate,
        'federal_funds_rate': federal_funds_rate,
        'gdp_growth': gdp_growth,
        'money_supply_growth': money_growth_trend,
        'long_term_yield': long_term_yield,
        'exchange_rate': exchange_rate,
        'consumer_confidence': consumer_confidence,
        'industrial_production': industrial_production,
        'inflation_expectations': inflation_expectations,
        'output_gap': output_gap,
        'nairu': nairu
    })
    
    print(colored(f"✅ Created realistic macro dataset: {macro_data.shape[0]} observations", "green"))
    return macro_data


async def comprehensive_macro_analysis():
    """Run comprehensive macroeconomic analysis"""
    print(colored("🏛️  Starting Comprehensive Macroeconomic Analysis", "cyan"))
    print("=" * 60)
    
    # Initialize macro agent
    macro_agent = MacroEconomicAgent()
    
    # Create and load data
    macro_data = await create_realistic_macro_data()
    macro_agent.load_data(macro_data, "Realistic macroeconomic data (1990-2023)")
    
    # 1. Variable Identification
    print(colored("\n🔍 Step 1: Identifying Macroeconomic Variables", "yellow"))
    var_results = await macro_agent.identify_macro_variables()
    
    # 2. Phillips Curve Analysis
    print(colored("\n📈 Step 2: Phillips Curve Analysis", "yellow"))
    phillips_results = await macro_agent.phillips_curve_analysis(
        inflation_var="core_inflation",
        unemployment_var="unemployment_rate",
        expectations_var="inflation_expectations"
    )
    
    # 3. Taylor Rule Analysis
    print(colored("\n🏦 Step 3: Taylor Rule Analysis", "yellow"))
    taylor_results = await macro_agent.taylor_rule_analysis(
        policy_rate_var="federal_funds_rate",
        inflation_var="core_inflation",
        output_gap_var="output_gap",
        target_inflation=2.0
    )
    
    # 4. Inflation Dynamics Analysis
    print(colored("\n💰 Step 4: Inflation Dynamics Analysis", "yellow"))
    inflation_results = await macro_agent.inflation_analysis(
        ["core_inflation", "cpi_inflation"]
    )
    
    # 5. Monetary Policy Analysis
    print(colored("\n🏦 Step 5: Monetary Policy Analysis", "yellow"))
    policy_results = await macro_agent.monetary_policy_analysis(
        ["federal_funds_rate", "money_supply_growth", "long_term_yield"]
    )
    
    # 6. Comprehensive Economic Interpretation
    print(colored("\n🎯 Step 6: Comprehensive Economic Interpretation", "yellow"))
    
    # Combine all results for interpretation
    combined_results = {
        "phillips_curve": phillips_results,
        "taylor_rule": taylor_results,
        "inflation_analysis": inflation_results,
        "monetary_policy": policy_results
    }
    
    economic_interpretation = await macro_agent.economic_interpretation(
        combined_results,
        policy_context="Analysis of U.S. macroeconomic relationships from 1990-2023, covering multiple business cycles and policy regimes."
    )
    
    # Print comprehensive summary
    print(colored("\n📊 COMPREHENSIVE MACROECONOMIC SUMMARY", "cyan"))
    print("=" * 60)
    macro_agent.print_macro_summary()
    
    print(colored("\n🎯 ECONOMIC INTERPRETATION", "cyan"))
    print("-" * 60)
    print(economic_interpretation)
    
    # Save all results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "data_period": "1990-2023",
        "variable_identification": var_results,
        "phillips_curve": phillips_results,
        "taylor_rule": taylor_results,
        "inflation_analysis": inflation_results,
        "monetary_policy": policy_results,
        "economic_interpretation": economic_interpretation
    }
    
    macro_agent.save_results(all_results, f"comprehensive_macro_analysis_{timestamp}.json")
    
    return all_results


async def policy_scenario_analysis():
    """Demonstrate policy scenario analysis"""
    print(colored("\n🎭 POLICY SCENARIO ANALYSIS", "cyan"))
    print("=" * 60)
    
    macro_agent = MacroEconomicAgent()
    
    # Create data with different policy scenarios
    base_data = await create_realistic_macro_data()
    
    # Scenario 1: Higher inflation target (3% instead of 2%)
    scenario_data = base_data.copy()
    scenario_data['federal_funds_rate'] = scenario_data['federal_funds_rate'] + 1.0  # More aggressive policy
    
    macro_agent.load_data(scenario_data, "Policy scenario: Higher inflation target")
    
    # Analyze Taylor Rule under new scenario
    scenario_taylor = await macro_agent.taylor_rule_analysis(
        policy_rate_var="federal_funds_rate",
        inflation_var="core_inflation",
        output_gap_var="output_gap",
        target_inflation=3.0  # Higher target
    )
    
    print(colored("📊 Policy Scenario Results:", "green"))
    print(f"Taylor Rule inflation response: {scenario_taylor['inflation_response']:.3f}")
    print(f"Taylor Principle satisfied: {scenario_taylor['taylor_principle']['satisfied']}")
    
    # Get economic interpretation
    scenario_interpretation = await macro_agent.economic_interpretation(
        {"taylor_rule_scenario": scenario_taylor},
        policy_context="Analysis of alternative monetary policy with 3% inflation target"
    )
    
    print(colored("\n🎯 Scenario Interpretation:", "blue"))
    print(scenario_interpretation)


async def main():
    """Main function to run macroeconomic analysis examples"""
    try:
        print(colored("🚀 Macroeconomic Analysis Examples", "green"))
        print("=" * 60)
        
        # Run comprehensive analysis
        results = await comprehensive_macro_analysis()
        
        # Run policy scenario analysis
        await policy_scenario_analysis()
        
        print(colored("\n✅ All macroeconomic analyses completed successfully!", "green"))
        
        # Summary of key findings
        print(colored("\n📋 KEY FINDINGS SUMMARY:", "cyan"))
        print("-" * 30)
        
        if 'phillips_curve' in results:
            pc = results['phillips_curve']
            print(f"• Phillips Curve slope: {pc.get('slope_coefficient', 'N/A'):.4f}")
            print(f"• Phillips relationship: {pc.get('phillips_slope_interpretation', 'N/A')}")
        
        if 'taylor_rule' in results:
            tr = results['taylor_rule']
            print(f"• Taylor Rule inflation response: {tr.get('inflation_response', 'N/A'):.4f}")
            print(f"• Taylor Principle: {tr.get('taylor_principle', {}).get('interpretation', 'N/A')}")
        
        if 'inflation_analysis' in results:
            inf = results['inflation_analysis']
            core_stats = inf.get('descriptive_stats', {}).get('core_inflation', {})
            print(f"• Average inflation: {core_stats.get('mean', 'N/A'):.2f}%")
            print(f"• Inflation volatility: {core_stats.get('std', 'N/A'):.2f}%")
        
        print(colored("\n🎯 For more detailed analysis, check the saved results files!", "blue"))
        
    except Exception as e:
        print(colored(f"❌ Error in macro analysis: {str(e)}", "red"))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
