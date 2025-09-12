"""
Basic Usage Examples for Econometric Agents
Demonstrates how to use the econometric agents system
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import EconometricOrchestrator


async def create_sample_data():
    """Create sample economic data for demonstration"""
    print("📊 Creating sample economic data...")
    
    # Create time series data
    dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='Q')
    n_obs = len(dates)
    
    # Generate realistic economic data with relationships
    np.random.seed(42)
    
    # Generate base series
    gdp_growth = np.random.normal(2.5, 1.5, n_obs)  # GDP growth
    unemployment = 6.0 + np.random.normal(0, 1.0, n_obs)  # Unemployment rate
    
    # Phillips Curve relationship: inflation inversely related to unemployment
    inflation = 2.0 - 0.3 * (unemployment - 6.0) + np.random.normal(0, 0.8, n_obs)
    
    # Taylor Rule: policy rate responds to inflation and output
    policy_rate = 2.0 + 1.5 * (inflation - 2.0) + 0.5 * gdp_growth + np.random.normal(0, 0.5, n_obs)
    
    # Other macro variables
    money_supply_growth = np.random.normal(3.0, 2.0, n_obs)
    stock_returns = np.random.normal(8.0, 15.0, n_obs)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'gdp_growth': gdp_growth,
        'unemployment_rate': unemployment,
        'inflation_rate': inflation,
        'federal_funds_rate': policy_rate,
        'money_supply_growth': money_supply_growth,
        'stock_returns': stock_returns,
        'consumer_confidence': 100 + np.random.normal(0, 10, n_obs),
        'industrial_production': np.random.normal(1.0, 2.0, n_obs)
    })
    
    # Ensure realistic bounds
    data['unemployment_rate'] = np.clip(data['unemployment_rate'], 2.0, 15.0)
    data['inflation_rate'] = np.clip(data['inflation_rate'], -2.0, 8.0)
    data['federal_funds_rate'] = np.clip(data['federal_funds_rate'], 0.0, 20.0)
    
    print(f"✅ Created sample data: {data.shape[0]} observations, {data.shape[1]} variables")
    return data


async def demo_regression_analysis(orchestrator, data):
    """Demonstrate regression analysis"""
    print("\n" + "="*60)
    print("🔍 REGRESSION ANALYSIS DEMO")
    print("="*60)
    
    # Get regression agent
    regression_agent = orchestrator.get_agent("regression")
    
    # Load data
    regression_agent.load_data(data, "Sample macroeconomic data")
    
    # Run regression analysis
    results = await regression_agent.perform_analysis(
        dependent_var="inflation_rate",
        independent_vars=["unemployment_rate", "gdp_growth"],
        run_diagnostics=True
    )
    
    # Print summary
    regression_agent.print_summary()
    
    return results


async def demo_time_series_analysis(orchestrator, data):
    """Demonstrate time series analysis"""
    print("\n" + "="*60)
    print("📈 TIME SERIES ANALYSIS DEMO")
    print("="*60)
    
    # Get time series agent
    ts_agent = orchestrator.get_agent("time_series")
    
    # Load data and set time index
    ts_agent.load_data(data, "Sample macroeconomic time series")
    ts_agent.set_time_index("date", "Q")
    
    # Test stationarity
    stationarity_results = await ts_agent.stationarity_test(["inflation_rate", "unemployment_rate"])
    
    # Run ARIMA analysis
    arima_results = await ts_agent.perform_analysis(
        model_type="arima",
        variables="inflation_rate"
    )
    
    # Generate forecast
    forecast_results = await ts_agent.forecast(steps=8)
    
    print(f"📮 Forecast generated for next 8 quarters")
    
    return {"stationarity": stationarity_results, "arima": arima_results, "forecast": forecast_results}


async def demo_macro_analysis(orchestrator, data):
    """Demonstrate macroeconomic analysis"""
    print("\n" + "="*60)
    print("🏛️  MACROECONOMIC ANALYSIS DEMO")
    print("="*60)
    
    # Get macro agent
    macro_agent = orchestrator.get_agent("macro")
    
    # Load data
    macro_agent.load_data(data, "Sample macroeconomic data for macro analysis")
    
    # Identify macro variables
    var_identification = await macro_agent.identify_macro_variables()
    
    # Phillips Curve analysis
    phillips_results = await macro_agent.phillips_curve_analysis(
        inflation_var="inflation_rate",
        unemployment_var="unemployment_rate"
    )
    
    # Taylor Rule analysis
    taylor_results = await macro_agent.taylor_rule_analysis(
        policy_rate_var="federal_funds_rate",
        inflation_var="inflation_rate",
        output_gap_var="gdp_growth"  # Using GDP growth as proxy for output gap
    )
    
    # Inflation analysis
    inflation_results = await macro_agent.inflation_analysis("inflation_rate")
    
    # Print macro summary
    macro_agent.print_macro_summary()
    
    return {
        "variables": var_identification,
        "phillips": phillips_results,
        "taylor": taylor_results,
        "inflation": inflation_results
    }


async def demo_automated_pipeline(orchestrator, data):
    """Demonstrate automated analysis pipeline"""
    print("\n" + "="*60)
    print("🔄 AUTOMATED ANALYSIS PIPELINE DEMO")
    print("="*60)
    
    research_question = """
    What is the relationship between inflation and unemployment in the macroeconomy?
    How does monetary policy respond to economic conditions?
    """
    
    pipeline_results = await orchestrator.auto_analysis_pipeline(
        research_question=research_question,
        data=data,
        data_description="Quarterly macroeconomic data from 2000-2023"
    )
    
    print("\n📋 COMPREHENSIVE INTERPRETATION:")
    print("-" * 40)
    print(pipeline_results.get("comprehensive_interpretation", "No interpretation available"))
    
    return pipeline_results


async def demo_interactive_features(orchestrator):
    """Demonstrate interactive features (non-blocking demo)"""
    print("\n" + "="*60)
    print("💬 INTERACTIVE FEATURES DEMO")
    print("="*60)
    
    # Show system status
    orchestrator.print_system_status()
    
    # List all agents
    orchestrator.list_agents()
    
    # Get recommendations for a research question
    recommendations = await orchestrator.recommend_analysis_approach(
        "How does monetary policy affect inflation expectations and economic growth?"
    )
    
    print("\n🎯 ANALYSIS RECOMMENDATIONS:")
    print("-" * 30)
    print(recommendations.get("recommendation", "No recommendations available"))


async def main():
    """Main demonstration function"""
    print("🚀 Econometric Agents System - Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = EconometricOrchestrator()
        
        # Create sample data
        data = await create_sample_data()
        
        # Run demonstrations
        print("\n🎭 Running comprehensive demonstrations...")
        
        # 1. Regression Analysis
        regression_results = await demo_regression_analysis(orchestrator, data)
        
        # 2. Time Series Analysis
        ts_results = await demo_time_series_analysis(orchestrator, data)
        
        # 3. Macroeconomic Analysis
        macro_results = await demo_macro_analysis(orchestrator, data)
        
        # 4. Automated Pipeline
        pipeline_results = await demo_automated_pipeline(orchestrator, data)
        
        # 5. Interactive Features
        await demo_interactive_features(orchestrator)
        
        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n📊 SUMMARY OF RESULTS:")
        print("-" * 30)
        print(f"• Regression R²: {regression_results.get('r_squared', 'N/A'):.4f}")
        print(f"• Phillips Curve slope: {macro_results['phillips'].get('slope_coefficient', 'N/A'):.4f}")
        print(f"• Taylor Rule inflation response: {macro_results['taylor'].get('inflation_response', 'N/A'):.4f}")
        print(f"• ARIMA model AIC: {ts_results['arima'].get('aic', 'N/A'):.2f}")
        
        print("\n🎯 NEXT STEPS:")
        print("- Try the interactive chat: orchestrator.interactive_chat()")
        print("- Load your own data: orchestrator.load_data_to_all(your_data)")
        print("- Explore specific agents: orchestrator.get_agent('macro')")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())

