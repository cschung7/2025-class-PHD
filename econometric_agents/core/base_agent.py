"""
Base Econometric Agent Class
Provides core functionality for all specialized econometric agents
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from termcolor import colored

from config.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MODEL_MINI,
    MAX_TOKENS, TEMPERATURE, MAX_RETRIES,
    RESULTS_DIR, LOGS_DIR
)


class BaseEconometricAgent(ABC):
    """
    Abstract base class for all econometric agents
    """
    
    def __init__(self, 
                 agent_name: str,
                 specialization: str,
                 model_name: str = OPENAI_MODEL,
                 temperature: float = TEMPERATURE):
        """
        Initialize the base econometric agent
        
        Args:
            agent_name: Name identifier for the agent
            specialization: Area of econometric specialization
            model_name: OpenAI model to use
            temperature: Sampling temperature for responses
        """
        self.agent_name = agent_name
        self.specialization = specialization
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Data storage
        self.current_data = None
        self.analysis_results = {}
        
        print(colored(f"✓ Initialized {agent_name} - {specialization}", "green"))
    
    def _setup_logging(self):
        """Setup logging for the agent"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        log_filename = f"{self.agent_name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = os.path.join(LOGS_DIR, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.agent_name)
    
    async def _call_openai_api(self, 
                              messages: List[Dict[str, str]], 
                              max_tokens: int = MAX_TOKENS,
                              temperature: float = None) -> str:
        """
        Make an API call to OpenAI with retry logic
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            temperature: Override default temperature
            
        Returns:
            Response content from OpenAI
        """
        if temperature is None:
            temperature = self.temperature
        
        for attempt in range(MAX_RETRIES):
            try:
                print(colored(f"🤖 {self.agent_name} thinking...", "yellow"))
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                self.logger.info(f"OpenAI API call successful (attempt {attempt + 1})")
                
                return content
                
            except Exception as e:
                self.logger.error(f"OpenAI API call failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to get response after {MAX_RETRIES} attempts: {str(e)}")
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def load_data(self, data: Union[pd.DataFrame, str, Dict], data_description: str = ""):
        """
        Load data for analysis
        
        Args:
            data: DataFrame, file path, or data dictionary
            data_description: Description of the data
        """
        try:
            if isinstance(data, str):
                # Assume it's a file path
                if data.endswith('.csv'):
                    self.current_data = pd.read_csv(data, encoding='utf-8')
                elif data.endswith('.xlsx'):
                    self.current_data = pd.read_excel(data)
                else:
                    raise ValueError(f"Unsupported file format: {data}")
                    
            elif isinstance(data, pd.DataFrame):
                self.current_data = data.copy()
                
            elif isinstance(data, dict):
                self.current_data = pd.DataFrame(data)
                
            else:
                raise ValueError("Data must be DataFrame, file path, or dictionary")
            
            print(colored(f"✓ Data loaded: {self.current_data.shape[0]} rows, {self.current_data.shape[1]} columns", "green"))
            
            if data_description:
                self.data_description = data_description
                print(colored(f"📊 Data description: {data_description}", "blue"))
            
            self.logger.info(f"Data loaded successfully: {self.current_data.shape}")
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of current data
        
        Returns:
            Dictionary containing data summary
        """
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        summary = {
            "shape": self.current_data.shape,
            "columns": list(self.current_data.columns),
            "dtypes": self.current_data.dtypes.to_dict(),
            "missing_values": self.current_data.isnull().sum().to_dict(),
            "numeric_summary": self.current_data.describe().to_dict() if len(self.current_data.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        return summary
    
    async def analyze_data_context(self) -> str:
        """
        Use OpenAI to analyze and understand the data context
        
        Returns:
            Analysis of data context and suggestions
        """
        if self.current_data is None:
            return "No data available for analysis"
        
        data_summary = self.get_data_summary()
        
        system_prompt = f"""You are an expert econometrician specializing in {self.specialization}. 
        Analyze the provided data summary and provide insights about:
        1. Data structure and quality
        2. Potential econometric modeling approaches
        3. Variables that might serve as dependent/independent variables
        4. Potential issues or concerns
        5. Recommended preprocessing steps
        
        Be specific and practical in your recommendations."""
        
        user_prompt = f"""
        Data Summary:
        - Shape: {data_summary['shape']}
        - Columns: {data_summary['columns']}
        - Data Types: {data_summary['dtypes']}
        - Missing Values: {data_summary['missing_values']}
        
        {f"Numeric Summary: {data_summary['numeric_summary']}" if data_summary['numeric_summary'] else ""}
        
        Please analyze this data and provide econometric insights.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        analysis = await self._call_openai_api(messages)
        
        print(colored("📈 Data Context Analysis:", "cyan"))
        print(analysis)
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save analysis results to file
        
        Args:
            results: Results dictionary to save
            filename: Optional custom filename
        """
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.agent_name}_results_{timestamp}.json"
            
            filepath = os.path.join(RESULTS_DIR, filename)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert numpy types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            converted_results = recursive_convert(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
            print(colored(f"✓ Results saved to: {filepath}", "green"))
            self.logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            print(colored(error_msg, "red"))
            self.logger.error(error_msg)
    
    @abstractmethod
    async def perform_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method for performing specialized analysis
        Must be implemented by subclasses
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def get_available_methods(self) -> List[str]:
        """
        Abstract method to return available analysis methods
        Must be implemented by subclasses
        
        Returns:
            List of available method names
        """
        pass
    
    async def chat(self, user_message: str) -> str:
        """
        Interactive chat interface for the agent
        
        Args:
            user_message: User's question or request
            
        Returns:
            Agent's response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Create system prompt based on agent specialization and current data
        system_prompt = f"""You are an expert econometric agent specializing in {self.specialization}.
        
        Your capabilities include:
        - {', '.join(self.get_available_methods())}
        
        Current data status: {"Data loaded" if self.current_data is not None else "No data loaded"}
        
        Provide helpful, accurate econometric advice and analysis. If the user asks about performing
        analysis, guide them on how to use your specialized methods.
        
        Be practical and provide code examples when appropriate using statsmodels."""
        
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        
        response = await self._call_openai_api(messages)
        
        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
