import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import requests
import subprocess
import sys
from datetime import datetime

class DataInsightsAgent:
    def __init__(self, df: pd.DataFrame, backend: str = "ollama", model: str = "llama3.1"):
        """
        Initialize AI agent for data insights.
        
        Args:
            df: The dataset to analyze
            backend: AI backend ('ollama', 'huggingface', 'openai-compatible')
            model: Model name/identifier
        """
        self.df = df
        self.backend = backend.lower()
        self.model = model
        self.data_context = self._build_data_context()
        
    def _build_data_context(self) -> Dict[str, Any]:
        """Build comprehensive context about the dataset."""
        try:
            # Basic info
            context = {
                "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": self.df.isnull().sum().to_dict(),
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
            # Statistical summary for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                context["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
            
            # Categorical info
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            context["categorical_info"] = {}
            for col in cat_cols[:10]:  # Limit to avoid huge context
                unique_count = self.df[col].nunique()
                context["categorical_info"][col] = {
                    "unique_values": unique_count,
                    "top_values": self.df[col].value_counts().head(5).to_dict() if unique_count < 100 else f"{unique_count} unique values"
                }
            
            # Data quality metrics
            context["data_quality"] = {
                "completeness": f"{((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100):.1f}%",
                "duplicate_rows": self.df.duplicated().sum(),
                "columns_with_nulls": self.df.isnull().any().sum()
            }
            
            return context
        except Exception as e:
            logger.error(f"Error building data context: {e}")
            return {"error": "Could not build data context", "basic_info": {"rows": len(self.df), "columns": len(self.df.columns)}}

    def _create_system_prompt(self) -> str:
        """Create system prompt with data context."""
        return f"""You are an expert data analyst AI assistant. You have access to a dataset with the following characteristics:

Dataset Overview:
- Shape: {self.data_context['shape']['rows']} rows, {self.data_context['shape']['columns']} columns
- Memory usage: {self.data_context.get('memory_usage', 'Unknown')}
- Data completeness: {self.data_context.get('data_quality', {}).get('completeness', 'Unknown')}

Columns and Types:
{json.dumps(self.data_context['dtypes'], indent=2)}

Missing Values:
{json.dumps(self.data_context['missing_values'], indent=2)}

Statistical Summary (numeric columns):
{json.dumps(self.data_context.get('numeric_summary', {}), indent=2) if self.data_context.get('numeric_summary') else 'No numeric columns'}

Categorical Information:
{json.dumps(self.data_context.get('categorical_info', {}), indent=2)}

Data Quality:
{json.dumps(self.data_context.get('data_quality', {}), indent=2)}

Your role is to:
1. Answer questions about this dataset intelligently
2. Provide data insights, trends, and patterns
3. Suggest data analysis approaches
4. Recommend visualizations
5. Identify potential data quality issues
6. Propose feature engineering ideas
7. Suggest machine learning approaches when relevant

Always provide actionable, specific insights based on the actual data characteristics shown above.
"""

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API."""
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "system": self._create_system_prompt()
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.json().get("response", "No response generated")
            
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"Error querying Ollama: {str(e)}"

    def _query_huggingface(self, prompt: str) -> str:
        """Query using Hugging Face Transformers (local inference)."""
        try:
            from transformers import pipeline, AutoTokenizer
            
            # Use a smaller model for local inference
            model_name = self.model if self.model != "llama3.1" else "microsoft/DialoGPT-medium"
            
            generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            full_prompt = f"{self._create_system_prompt()}\n\nUser Question: {prompt}\n\nAnswer:"
            
            result = generator(full_prompt, max_new_tokens=300, pad_token_id=50256)
            return result[0]["generated_text"].split("Answer:")[-1].strip()
            
        except ImportError:
            return "Error: transformers library not installed. Run: pip install transformers torch"
        except Exception as e:
            logger.error(f"Hugging Face query error: {e}")
            return f"Error with Hugging Face model: {str(e)}"

    def _query_openai_compatible(self, prompt: str) -> str:
        """Query OpenAI-compatible API (e.g., LocalAI, vLLM)."""
        try:
            url = "http://localhost:8080/v1/chat/completions"  # Adjust as needed
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            return f"Error querying OpenAI-compatible API: {str(e)}"

    def query(self, user_question: str) -> Dict[str, Any]:
        """Main method to query the AI agent."""
        logger.info(f"Processing query: {user_question[:100]}...")
        
        start_time = datetime.now()
        
        # Route to appropriate backend
        if self.backend == "ollama":
            response = self._query_ollama(user_question)
        elif self.backend == "huggingface":
            response = self._query_huggingface(user_question)
        elif self.backend == "openai-compatible":
            response = self._query_openai_compatible(user_question)
        else:
            response = f"Error: Unsupported backend '{self.backend}'"
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        result = {
            "query": user_question,
            "response": response,
            "backend": self.backend,
            "model": self.model,
            "response_time_seconds": response_time,
            "timestamp": end_time.isoformat(),
            "data_context_summary": {
                "dataset_shape": self.data_context["shape"],
                "columns_analyzed": len(self.data_context["columns"])
            }
        }
        
        logger.info(f"Query processed in {response_time:.2f}s using {self.backend}/{self.model}")
        return result

    def suggest_analyses(self) -> List[str]:
        """Suggest relevant analyses based on dataset characteristics."""
        suggestions = []
        
        # Based on data types
        numeric_cols = len([col for col, dtype in self.data_context['dtypes'].items() 
                          if 'int' in dtype or 'float' in dtype])
        cat_cols = len([col for col, dtype in self.data_context['dtypes'].items() 
                       if 'object' in dtype])
        
        if numeric_cols > 1:
            suggestions.append("What are the correlations between numeric variables?")
            suggestions.append("Which columns have outliers that need attention?")
            suggestions.append("What statistical patterns do you see in the data?")
        
        if cat_cols > 0:
            suggestions.append("What insights can you provide about the categorical variables?")
            suggestions.append("Are there any data quality issues I should address?")
        
        # Based on missing values
        missing_pct = sum(self.data_context['missing_values'].values()) / (len(self.df) * len(self.df.columns))
        if missing_pct > 0.1:
            suggestions.append("How should I handle the missing values in this dataset?")
        
        # Based on size
        if len(self.df) > 10000:
            suggestions.append("What sampling strategies would work well for this large dataset?")
        
        suggestions.append("What machine learning approaches would be suitable for this data?")
        suggestions.append("What visualizations would best represent this data?")
        
        return suggestions[:5]  # Return top 5 suggestions
