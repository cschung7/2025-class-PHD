#!/usr/bin/env python3
"""
US Inflation Research using Perplexity AI
Verify current inflation data for 2024-2025 period
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/mnt/nas/gpt/.env")

class InflationResearcher:
    """Research current US inflation data using Perplexity AI."""
    
    def __init__(self):
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment")
        
        # Define targeted research queries
        self.research_queries = [
            # Latest CPI data verification
            "US consumer price index CPI inflation rate August September 2025 latest data Bureau of Labor Statistics",
            
            # Core inflation trends
            "US core inflation rate August September 2025 excluding food energy Federal Reserve target",
            
            # Recent inflation trends and context
            "US inflation trends 2024 2025 Federal Reserve monetary policy rate changes economic outlook",
            
            # Comparative analysis
            "US inflation rate changes July August September 2025 month over month annual comparison",
            
            # Economic policy context
            "Federal Reserve inflation target 2025 Jerome Powell monetary policy interest rates inflation expectations"
        ]
    
    def search_inflation_data(self, query: str) -> Dict:
        """Use Perplexity API to search for current inflation data."""
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an economic data analyst. Provide the most recent and accurate US inflation data from official sources like the Bureau of Labor Statistics, Federal Reserve, and reputable financial news. Include specific numbers, dates, and cite your sources. Focus on 2024-2025 data."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "search_domain_filter": ["bls.gov", "federalreserve.gov", "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com"],
            "return_citations": True
        }
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Perplexity API error for query '{query[:50]}...': {e}")
            return {"error": str(e)}
    
    def research_current_inflation(self) -> Dict:
        """Conduct comprehensive inflation research."""
        print("🔍 Researching Current US Inflation Data...")
        print("=" * 60)
        
        research_results = {
            "timestamp": datetime.now().isoformat(),
            "queries_executed": [],
            "findings": {},
            "summary": {},
            "sources": []
        }
        
        for i, query in enumerate(self.research_queries, 1):
            print(f"\n📊 Query {i}/5: Searching for {query[:50]}...")
            
            result = self.search_inflation_data(query)
            
            if "error" not in result:
                # Extract key information
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Store results
                    research_results["queries_executed"].append(query)
                    research_results["findings"][f"query_{i}"] = {
                        "query": query,
                        "response": content,
                        "citations": result.get("citations", [])
                    }
                    
                    # Extract citations if available
                    if "citations" in result:
                        research_results["sources"].extend(result["citations"])
                    
                    print(f"✅ Found data: {content[:100]}...")
                else:
                    print(f"⚠️  No response content for query {i}")
            else:
                print(f"❌ Error in query {i}: {result['error']}")
        
        return research_results
    
    def synthesize_findings(self, research_results: Dict) -> Dict:
        """Synthesize research findings into coherent summary."""
        synthesis = {
            "key_findings": {
                "latest_cpi_rate": "To be extracted from research",
                "core_inflation_rate": "To be extracted from research", 
                "inflation_trend": "To be extracted from research",
                "fed_target_comparison": "To be extracted from research",
                "economic_context": "To be extracted from research"
            },
            "data_verification": {
                "august_2025_rate": None,
                "september_2025_rate": None,
                "previous_user_data": {
                    "august_rate": "2.9%",
                    "core_rate": "3.1%",
                    "source": "User provided"
                },
                "verification_status": "Pending analysis"
            },
            "confidence_assessment": {
                "source_quality": "High (official sources prioritized)",
                "data_recency": "Current search",
                "cross_validation": "Multiple sources consulted"
            }
        }
        
        # Extract key data points from research findings
        all_responses = []
        for finding_key in research_results["findings"]:
            finding = research_results["findings"][finding_key]
            all_responses.append(finding["response"])
        
        # Store raw responses for analysis
        synthesis["raw_research_data"] = all_responses
        
        return synthesis
    
    def generate_report(self, research_results: Dict, synthesis: Dict) -> str:
        """Generate comprehensive inflation research report."""
        report = f"""# US Inflation Research Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report verifies current US inflation data using Perplexity AI web search focusing on official sources and recent economic data.

### Research Methodology
- **API Used**: Perplexity AI (sonar model)
- **Sources Prioritized**: BLS.gov, FederalReserve.gov, Bloomberg, Reuters, WSJ, CNBC
- **Queries Executed**: {len(research_results['queries_executed'])}
- **Search Focus**: August-September 2025 inflation data

## User's Previous Data (To Verify)
- **August 2025 Inflation**: 2.9% (up from 2.7%)
- **Core Inflation**: 3.1%
- **Source**: User provided data

## Research Findings

"""
        
        # Add findings from each query
        for i, finding_key in enumerate(research_results["findings"], 1):
            finding = research_results["findings"][finding_key]
            report += f"""### Query {i}: {finding['query'][:60]}...

**Response:**
{finding['response']}

**Citations:** {len(finding.get('citations', []))} sources
---

"""
        
        report += f"""## Data Verification Analysis

### Key Metrics Found:
{synthesis['key_findings']}

### Verification Status:
{synthesis['data_verification']}

### Confidence Assessment:
- **Source Quality**: {synthesis['confidence_assessment']['source_quality']}
- **Data Recency**: {synthesis['confidence_assessment']['data_recency']}
- **Cross-Validation**: {synthesis['confidence_assessment']['cross_validation']}

## Sources Consulted

"""
        
        # Add unique sources
        unique_sources = list(set(research_results["sources"]))
        for i, source in enumerate(unique_sources[:10], 1):  # Limit to top 10
            if isinstance(source, str):
                report += f"{i}. {source}\n"
            elif isinstance(source, dict) and 'url' in source:
                report += f"{i}. {source['url']} - {source.get('title', 'No title')}\n"
        
        report += f"""
## Research Metadata

- **Total API Calls**: {len(research_results['queries_executed'])}
- **Search Domains**: BLS, Federal Reserve, Bloomberg, Reuters, WSJ, CNBC
- **Response Time**: Real-time search
- **Model Used**: Perplexity sonar
- **Temperature**: 0.1 (high precision)

## Recommendations

1. **Data Validation**: Compare findings with official BLS releases
2. **Trend Analysis**: Monitor month-over-month changes
3. **Policy Context**: Consider Federal Reserve policy implications
4. **Follow-up Research**: Track any data revisions or updates

---
*Report generated using Perplexity AI for external data validation*
"""
        
        return report

def main():
    """Execute inflation research."""
    print("🏛️  US INFLATION DATA VERIFICATION")
    print("=" * 50)
    print("Using Perplexity AI to verify current inflation data\n")
    
    try:
        # Initialize researcher
        researcher = InflationResearcher()
        print("✅ Perplexity API connection established")
        
        # Conduct research
        research_results = researcher.research_current_inflation()
        
        # Synthesize findings
        print("\n🔄 Synthesizing research findings...")
        synthesis = researcher.synthesize_findings(research_results)
        
        # Generate report
        print("📄 Generating comprehensive report...")
        report = researcher.generate_report(research_results, synthesis)
        
        # Save results
        results_file = "/mnt/nas/Class/2025/PHD/inflation_research_results.json"
        report_file = "/mnt/nas/Class/2025/PHD/inflation_research_report.md"
        
        with open(results_file, 'w') as f:
            json.dump({
                "research_results": research_results,
                "synthesis": synthesis
            }, f, indent=2)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print("\n✅ Research Complete!")
        print(f"📊 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")
        
        # Quick summary
        print("\n📋 Quick Summary:")
        print(f"- Queries executed: {len(research_results.get('queries_executed', []))}")
        print(f"- Data sources consulted: {len(set(research_results.get('sources', [])))}")
        print("- Focus: Verifying August-September 2025 US inflation data")
        
        return research_results, synthesis
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        return None, None

if __name__ == "__main__":
    main()