"""
Econometric Agents Orchestrator
Main interface for managing and coordinating econometric agents
"""
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from termcolor import colored

from agents.regression_agent import RegressionAgent
from agents.time_series_agent import TimeSeriesAgent
from agents.panel_data_agent import PanelDataAgent
from agents.macro_agent import MacroEconomicAgent
from config.config import RESULTS_DIR, CHARTS_DIR


class EconometricOrchestrator:
    """
    Main orchestrator for managing econometric agents
    """
    
    def __init__(self):
        """Initialize the orchestrator with all available agents"""
        print(colored("🚀 Initializing Econometric Agents System...", "cyan"))
        
        # Initialize agents
        self.agents = {}
        
        try:
            self.agents["regression"] = RegressionAgent()
            self.agents["time_series"] = TimeSeriesAgent()
            self.agents["panel_data"] = PanelDataAgent()
            self.agents["macro"] = MacroEconomicAgent()
            
            print(colored(f"✅ Initialized {len(self.agents)} specialized agents", "green"))
            
        except Exception as e:
            print(colored(f"❌ Error initializing agents: {str(e)}", "red"))
            raise
        
        # Create output directories
        self._create_directories()
        
        # Current active agent
        self.active_agent = None
        self.current_data = None
        
        print(colored("🎯 Econometric Agents System ready!", "green"))
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [RESULTS_DIR, CHARTS_DIR]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(colored("📁 Output directories created", "blue"))
    
    def list_agents(self):
        """List all available agents and their capabilities"""
        print(colored("\n🤖 Available Econometric Agents:", "cyan"))
        print(colored("=" * 50, "cyan"))
        
        for agent_type, agent in self.agents.items():
            print(colored(f"\n{agent_type.upper()} AGENT:", "yellow"))
            print(f"  Name: {agent.agent_name}")
            print(f"  Specialization: {agent.specialization}")
            print(f"  Methods: {', '.join(agent.get_available_methods())}")
        
        print(colored("\n" + "=" * 50, "cyan"))
    
    def get_agent(self, agent_type: str):
        """
        Get a specific agent by type
        
        Args:
            agent_type: Type of agent ("regression", "time_series", "panel_data", "macro")
            
        Returns:
            The requested agent instance
        """
        if agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not available. Available: {list(self.agents.keys())}")
        
        return self.agents[agent_type]
    
    def set_active_agent(self, agent_type: str):
        """
        Set the active agent for subsequent operations
        
        Args:
            agent_type: Type of agent to activate
        """
        if agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not available")
        
        self.active_agent = self.agents[agent_type]
        print(colored(f"🎯 Active agent set to: {self.active_agent.agent_name}", "green"))
    
    def load_data_to_all(self, data, data_description: str = ""):
        """
        Load data to all agents
        
        Args:
            data: Data to load (DataFrame, file path, or dict)
            data_description: Description of the data
        """
        print(colored("📊 Loading data to all agents...", "cyan"))
        
        for agent_type, agent in self.agents.items():
            try:
                agent.load_data(data, data_description)
                print(colored(f"  ✓ Data loaded to {agent_type} agent", "blue"))
            except Exception as e:
                print(colored(f"  ❌ Failed to load data to {agent_type} agent: {str(e)}", "red"))
        
        self.current_data = data
        print(colored("✅ Data loading completed", "green"))
    
    def load_data_to_agent(self, agent_type: str, data, data_description: str = ""):
        """
        Load data to a specific agent
        
        Args:
            agent_type: Type of agent
            data: Data to load
            data_description: Description of the data
        """
        if agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not available")
        
        agent = self.agents[agent_type]
        agent.load_data(data, data_description)
        
        print(colored(f"✓ Data loaded to {agent_type} agent", "green"))
    
    async def analyze_data_context_all(self) -> Dict[str, str]:
        """
        Get data context analysis from all agents
        
        Returns:
            Dictionary with analysis from each agent
        """
        print(colored("🔍 Analyzing data context with all agents...", "cyan"))
        
        analyses = {}
        
        for agent_type, agent in self.agents.items():
            if agent.current_data is not None:
                try:
                    print(colored(f"  Analyzing with {agent_type} agent...", "yellow"))
                    analysis = await agent.analyze_data_context()
                    analyses[agent_type] = analysis
                    print(colored(f"  ✓ {agent_type} analysis completed", "blue"))
                except Exception as e:
                    analyses[agent_type] = f"Error: {str(e)}"
                    print(colored(f"  ❌ {agent_type} analysis failed: {str(e)}", "red"))
            else:
                analyses[agent_type] = "No data loaded"
        
        return analyses
    
    async def recommend_analysis_approach(self, research_question: str) -> Dict[str, Any]:
        """
        Get recommendations for analysis approach based on research question
        
        Args:
            research_question: The research question or objective
            
        Returns:
            Recommendations from AI about analysis approach
        """
        print(colored("🎯 Getting analysis recommendations...", "cyan"))
        
        # Use the regression agent's OpenAI client for this general recommendation
        regression_agent = self.agents["regression"]
        
        system_prompt = """You are an expert econometrician and data scientist. Based on the research question provided, recommend:
        1. Which type of econometric analysis is most appropriate (regression, time series, panel data)
        2. Specific methods or models to consider
        3. Data requirements and preprocessing steps
        4. Potential challenges and solutions
        5. Expected outputs and interpretation
        
        Consider the strengths and limitations of different econometric approaches."""
        
        user_prompt = f"""
        Research Question: {research_question}
        
        Available Analysis Types:
        - Regression Analysis: Cross-sectional regression, model selection, diagnostics
        - Time Series Analysis: ARIMA, VAR, VECM, stationarity tests, forecasting
        - Panel Data Analysis: Fixed effects, random effects, Hausman test
        - Macroeconomic Analysis: Phillips Curve, Taylor Rule, inflation analysis, monetary policy
        
        Please provide detailed recommendations for analyzing this research question.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            recommendation = await regression_agent._call_openai_api(messages)
            
            result = {
                "research_question": research_question,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }
            
            print(colored("✅ Analysis recommendations generated", "green"))
            print(colored("\n📋 RECOMMENDATIONS:", "cyan"))
            print(recommendation)
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            print(colored(error_msg, "red"))
            return {"error": error_msg}
    
    async def auto_analysis_pipeline(self, 
                                   research_question: str,
                                   data,
                                   data_description: str = "") -> Dict[str, Any]:
        """
        Automated analysis pipeline that:
        1. Analyzes the research question
        2. Loads data to appropriate agents
        3. Runs recommended analyses
        4. Compiles results
        
        Args:
            research_question: The research question
            data: Data for analysis
            data_description: Description of the data
            
        Returns:
            Comprehensive analysis results
        """
        print(colored("🔄 Starting automated analysis pipeline...", "cyan"))
        
        pipeline_results = {
            "research_question": research_question,
            "data_description": data_description,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        try:
            # Step 1: Get recommendations
            print(colored("📋 Step 1: Getting analysis recommendations...", "yellow"))
            recommendations = await self.recommend_analysis_approach(research_question)
            pipeline_results["steps"]["recommendations"] = recommendations
            
            # Step 2: Load data
            print(colored("📊 Step 2: Loading data to agents...", "yellow"))
            self.load_data_to_all(data, data_description)
            pipeline_results["steps"]["data_loaded"] = True
            
            # Step 3: Get data context from all agents
            print(colored("🔍 Step 3: Analyzing data context...", "yellow"))
            context_analyses = await self.analyze_data_context_all()
            pipeline_results["steps"]["context_analyses"] = context_analyses
            
            # Step 4: Run basic analyses with each agent (if data is suitable)
            print(colored("⚙️  Step 4: Running analyses...", "yellow"))
            analysis_results = {}
            
            # Try regression analysis
            try:
                regression_agent = self.agents["regression"]
                if regression_agent.current_data is not None and len(regression_agent.current_data.columns) >= 2:
                    numeric_cols = regression_agent.current_data.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 2:
                        dependent_var = numeric_cols[0]  # First numeric column as dependent
                        independent_vars = numeric_cols[1:3]  # Next 1-2 as independent
                        
                        print(colored(f"  Running regression: {dependent_var} ~ {' + '.join(independent_vars)}", "blue"))
                        regression_results = await regression_agent.perform_analysis(
                            dependent_var=dependent_var,
                            independent_vars=independent_vars,
                            run_diagnostics=True
                        )
                        analysis_results["regression"] = regression_results
            except Exception as e:
                analysis_results["regression"] = {"error": str(e)}
            
            # Try time series analysis if time-indexed data
            try:
                ts_agent = self.agents["time_series"]
                if ts_agent.current_data is not None:
                    # Look for date columns
                    date_cols = ts_agent.current_data.select_dtypes(include=['datetime64', 'object']).columns.tolist()
                    numeric_cols = ts_agent.current_data.select_dtypes(include=['number']).columns.tolist()
                    
                    if date_cols and numeric_cols:
                        # Try to set time index and run stationarity test
                        date_col = date_cols[0]
                        numeric_col = numeric_cols[0]
                        
                        print(colored(f"  Testing stationarity of {numeric_col}", "blue"))
                        # Note: This would require proper time index setup
                        # For now, just test stationarity if data looks time-series-like
                        if len(ts_agent.current_data) > 20:  # Reasonable time series length
                            stationarity_results = await ts_agent.stationarity_test(numeric_col)
                            analysis_results["time_series"] = stationarity_results
            except Exception as e:
                analysis_results["time_series"] = {"error": str(e)}
            
            # Try macroeconomic analysis
            try:
                macro_agent = self.agents["macro"]
                if macro_agent.current_data is not None:
                    print(colored("  Identifying macroeconomic variables", "blue"))
                    macro_vars = await macro_agent.identify_macro_variables()
                    analysis_results["macro"] = macro_vars
                    
                    # Try Phillips Curve if inflation and unemployment data available
                    if (macro_vars["identified_variables"]["inflation"] and 
                        macro_vars["identified_variables"]["employment"]):
                        print(colored("  Running Phillips Curve analysis", "blue"))
                        phillips_results = await macro_agent.phillips_curve_analysis()
                        analysis_results["phillips_curve"] = phillips_results
                        
            except Exception as e:
                analysis_results["macro"] = {"error": str(e)}
            
            pipeline_results["steps"]["analyses"] = analysis_results
            
            # Step 5: Generate comprehensive interpretation
            print(colored("📝 Step 5: Generating comprehensive interpretation...", "yellow"))
            interpretation = await self._generate_pipeline_interpretation(pipeline_results)
            pipeline_results["comprehensive_interpretation"] = interpretation
            
            # Save results
            self._save_pipeline_results(pipeline_results)
            
            print(colored("✅ Automated analysis pipeline completed!", "green"))
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Error in analysis pipeline: {str(e)}"
            print(colored(error_msg, "red"))
            pipeline_results["error"] = error_msg
            return pipeline_results
    
    async def _generate_pipeline_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive interpretation of pipeline results"""
        try:
            regression_agent = self.agents["regression"]
            
            system_prompt = """You are an expert econometrician providing a comprehensive analysis report. 
            Synthesize all the analysis results and provide:
            1. Executive summary of findings
            2. Key econometric insights
            3. Answers to the research question
            4. Limitations and caveats
            5. Recommendations for further analysis
            
            Be thorough but accessible, focusing on practical implications."""
            
            # Format results for AI
            formatted_results = f"""
            Research Question: {results['research_question']}
            
            Recommendations: {results['steps'].get('recommendations', {}).get('recommendation', 'N/A')}
            
            Analysis Results:
            {self._format_pipeline_results_for_ai(results['steps'].get('analyses', {}))}
            """
            
            user_prompt = formatted_results
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            interpretation = await regression_agent._call_openai_api(messages, max_tokens=2000)
            return interpretation
            
        except Exception as e:
            return f"Error generating comprehensive interpretation: {str(e)}"
    
    def _format_pipeline_results_for_ai(self, analyses: Dict[str, Any]) -> str:
        """Format analysis results for AI interpretation"""
        formatted = []
        
        for analysis_type, result in analyses.items():
            if isinstance(result, dict) and "error" not in result:
                if analysis_type == "regression":
                    formatted.append(f"Regression Analysis: R² = {result.get('r_squared', 'N/A'):.4f}, F p-value = {result.get('f_pvalue', 'N/A')}")
                elif analysis_type == "time_series":
                    formatted.append(f"Time Series Analysis: Stationarity tests completed")
                else:
                    formatted.append(f"{analysis_type}: Analysis completed")
            else:
                formatted.append(f"{analysis_type}: {result.get('error', 'Unknown error')}")
        
        return "\n".join(formatted) if formatted else "No successful analyses"
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_results_{timestamp}.json"
            filepath = os.path.join(RESULTS_DIR, filename)
            
            # Use the base agent's save_results method
            regression_agent = self.agents["regression"]
            regression_agent.save_results(results, filename)
            
        except Exception as e:
            print(colored(f"Warning: Could not save pipeline results: {str(e)}", "yellow"))
    
    async def interactive_chat(self, agent_type: str = None):
        """
        Start an interactive chat session with agents
        
        Args:
            agent_type: Specific agent to chat with, or None for orchestrator-level chat
        """
        if agent_type and agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not available")
        
        agent = self.agents[agent_type] if agent_type else None
        
        print(colored(f"\n💬 Interactive Chat Started", "cyan"))
        print(colored(f"Agent: {agent.agent_name if agent else 'Orchestrator'}", "blue"))
        print(colored("Type 'quit' to exit, 'help' for commands", "yellow"))
        print(colored("=" * 50, "cyan"))
        
        while True:
            try:
                user_input = input(colored("\nYou: ", "green")).strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(colored("Chat session ended.", "yellow"))
                    break
                
                if user_input.lower() == 'help':
                    self._print_chat_help()
                    continue
                
                if not user_input:
                    continue
                
                if agent:
                    # Chat with specific agent
                    response = await agent.chat(user_input)
                    print(colored(f"\n{agent.agent_name}: ", "cyan") + response)
                else:
                    # Orchestrator-level chat
                    response = await self._orchestrator_chat(user_input)
                    print(colored(f"\nOrchestrator: ", "cyan") + response)
                
            except KeyboardInterrupt:
                print(colored("\nChat session interrupted.", "yellow"))
                break
            except Exception as e:
                print(colored(f"Error: {str(e)}", "red"))
    
    def _print_chat_help(self):
        """Print chat help commands"""
        help_text = """
Available Commands:
- quit/exit/q: End chat session
- help: Show this help message

You can ask questions like:
- "What analysis should I run for my research question?"
- "How do I interpret these regression results?"
- "What's the difference between fixed and random effects?"
- "Run a regression analysis on my data"
        """
        print(colored(help_text, "blue"))
    
    async def _orchestrator_chat(self, user_message: str) -> str:
        """Handle orchestrator-level chat"""
        # Use regression agent's OpenAI client for orchestrator chat
        regression_agent = self.agents["regression"]
        
        system_prompt = """You are an econometric analysis orchestrator. You manage specialized agents for:
        - Regression Analysis (cross-sectional, model selection, diagnostics)
        - Time Series Analysis (ARIMA, VAR, VECM, forecasting)
        - Panel Data Analysis (fixed effects, random effects, Hausman test)
        - Macroeconomic Analysis (Phillips Curve, Taylor Rule, inflation analysis, monetary policy)
        
        Help users understand which agent to use, how to structure their analysis, and provide general econometric guidance.
        Be helpful in directing them to the right specialized agent and analysis approach."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await regression_agent._call_openai_api(messages)
            return response
        except Exception as e:
            return f"Error in orchestrator chat: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "agents_initialized": len(self.agents),
            "active_agent": self.active_agent.agent_name if self.active_agent else None,
            "data_loaded": self.current_data is not None,
            "agents_with_data": []
        }
        
        for agent_type, agent in self.agents.items():
            if agent.current_data is not None:
                status["agents_with_data"].append({
                    "agent": agent_type,
                    "data_shape": agent.current_data.shape,
                    "analysis_results": list(agent.analysis_results.keys())
                })
        
        return status
    
    def print_system_status(self):
        """Print current system status"""
        status = self.get_system_status()
        
        print(colored("\n📊 System Status", "cyan"))
        print(colored("=" * 30, "cyan"))
        print(f"Agents initialized: {status['agents_initialized']}")
        print(f"Active agent: {status['active_agent'] or 'None'}")
        print(f"Data loaded: {status['data_loaded']}")
        
        if status["agents_with_data"]:
            print(colored("\nAgents with data:", "yellow"))
            for agent_info in status["agents_with_data"]:
                print(f"  {agent_info['agent']}: {agent_info['data_shape']} - Results: {agent_info['analysis_results']}")
        
        print(colored("=" * 30, "cyan"))
