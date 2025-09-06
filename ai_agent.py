import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import requests
from datetime import datetime

class DataInsightsAgent:
    def __init__(self, df: pd.DataFrame, backend: str = "ollama", model: str = "llama3.1"):
        """
        Initialize AI agent for data insights.
        
        Args:
            df: The dataset to analyze
            backend: AI backend ('ollama', 'huggingface')
            model: Model name/identifier
        """
        self.df = df
        self.backend = backend.lower()
        self.model = model
        self.data_context = self._build_data_context()
        
    def _convert_to_json_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
        
    def _build_data_context(self) -> Dict[str, Any]:
        """Build comprehensive context about the dataset."""
        try:
            # Basic info
            context = {
                "shape": {"rows": int(len(self.df)), "columns": int(len(self.df.columns))},
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": {col: int(count) for col, count in self.df.isnull().sum().items()},
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
            # Statistical summary for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                desc = self.df[numeric_cols].describe()
                # Convert to JSON serializable
                context["numeric_summary"] = {}
                for col in desc.columns:
                    context["numeric_summary"][col] = {}
                    for stat in desc.index:
                        val = desc.loc[stat, col]
                        if pd.isna(val):
                            context["numeric_summary"][col][stat] = None
                        else:
                            context["numeric_summary"][col][stat] = float(val)
            
            # Categorical info
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            context["categorical_info"] = {}
            for col in cat_cols[:10]:  # Limit to avoid huge context
                unique_count = int(self.df[col].nunique())
                context["categorical_info"][col] = {
                    "unique_values": unique_count
                }
                
                if unique_count < 100:
                    # Convert value counts to JSON serializable
                    top_values = {}
                    for val, count in self.df[col].value_counts().head(5).items():
                        top_values[str(val)] = int(count)
                    context["categorical_info"][col]["top_values"] = top_values
                else:
                    context["categorical_info"][col]["top_values"] = f"{unique_count} unique values"
            
            # Data quality metrics
            total_cells = len(self.df) * len(self.df.columns)
            missing_cells = self.df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100
            
            context["data_quality"] = {
                "completeness": f"{completeness:.1f}%",
                "duplicate_rows": int(self.df.duplicated().sum()),
                "columns_with_nulls": int(self.df.isnull().any().sum())
            }
            
            # Convert entire context to be JSON serializable
            context = self._convert_to_json_serializable(context)
            return context
            
        except Exception as e:
            logger.error(f"Error building data context: {e}")
            return {
                "error": "Could not build data context", 
                "basic_info": {
                    "rows": int(len(self.df)), 
                    "columns": int(len(self.df.columns))
                }
            }

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
        """Query Ollama API with proper JSON serialization."""
        try:
            url = "http://localhost:11434/api/generate"
            
            # Ensure all data is JSON serializable
            payload = {
                "model": self.model,
                "prompt": str(prompt),  # Ensure string
                "stream": False,
                "system": self._create_system_prompt()
            }
            
            # Test JSON serialization before sending
            json.dumps(payload)
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
        except json.JSONEncoder as e:
            logger.error(f"JSON serialization error: {e}")
            return f"Error: Data serialization issue - {str(e)}"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"Error querying Ollama: {str(e)}"

    def _query_huggingface(self, prompt: str) -> str:
        """Query using Hugging Face Transformers (local inference) with tabular support."""
        # Check if this should be a tabular response first
        if self._detect_tabular_query(prompt):
            return self._generate_tabular_response(prompt)
            
        try:
            from transformers import pipeline
            
            # Use a smaller model for local inference
            model_name = "microsoft/DialoGPT-medium"
            
            generator = pipeline(
                "text-generation",
                model=model_name,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            
            # Create a focused prompt
            context_summary = f"Dataset: {self.data_context['shape']['rows']} rows, {self.data_context['shape']['columns']} columns. Columns: {', '.join(self.data_context['columns'][:5])}{'...' if len(self.data_context['columns']) > 5 else ''}"
            full_prompt = f"{context_summary}\nQuestion: {prompt}\nAnswer:"
            
            result = generator(full_prompt, max_new_tokens=200)
            response = result[0]["generated_text"].split("Answer:")[-1].strip()
            
            # If response is too short or unclear, provide a basic analysis
            if len(response) < 20:
                return self._generate_basic_insight(prompt)
            
            return response
            
        except ImportError:
            return "Error: transformers library not installed. Run: pip install transformers torch"
        except Exception as e:
            logger.error(f"Hugging Face query error: {e}")
            return self._generate_basic_insight(prompt)

    def _generate_basic_insight(self, prompt: str) -> str:
        """Generate basic insights without LLM when AI fails."""
        insights = []
        
        # Basic dataset info
        insights.append(f"This dataset contains {self.data_context['shape']['rows']} rows and {self.data_context['shape']['columns']} columns.")
        
        # Missing values insight
        missing_total = sum(self.data_context['missing_values'].values())
        if missing_total > 0:
            insights.append(f"There are {missing_total} missing values across the dataset.")
        
        # Column types
        numeric_cols = len([col for col, dtype in self.data_context['dtypes'].items() if 'int' in dtype or 'float' in dtype])
        cat_cols = len([col for col, dtype in self.data_context['dtypes'].items() if 'object' in dtype])
        
        if numeric_cols > 0:
            insights.append(f"The dataset has {numeric_cols} numeric columns suitable for statistical analysis.")
        if cat_cols > 0:
            insights.append(f"There are {cat_cols} categorical columns that could be analyzed for patterns.")
        
        # Specific insights based on query
        query_lower = prompt.lower()
        
        if 'high' in query_lower and any(col in query_lower for col in ['bilirubin', 'creatinine', 'value']):
            # Look for outliers in numeric columns
            numeric_insights = []
            for col in self.df.select_dtypes(include=[np.number]).columns:
                if any(term in col.lower() for term in ['bilirubin', 'creatinine']):
                    q75 = self.df[col].quantile(0.75)
                    q95 = self.df[col].quantile(0.95)
                    high_values = self.df[self.df[col] > q95]
                    if len(high_values) > 0:
                        numeric_insights.append(f"Found {len(high_values)} records with high {col} values (>{q95:.2f})")
            
            if numeric_insights:
                insights.extend(numeric_insights)
        
        # Suggestions based on query content
        if any(word in query_lower for word in ['correlation', 'relationship', 'pattern']):
            insights.append("For correlation analysis, focus on the numeric columns and look for linear relationships.")
        elif any(word in query_lower for word in ['clean', 'quality', 'missing']):
            insights.append("Consider handling missing values and checking for duplicates in your data preprocessing.")
        elif any(word in query_lower for word in ['visualize', 'plot', 'chart']):
            insights.append("Histograms work well for numeric data, bar charts for categorical data, and scatter plots for relationships.")
        
        return " ".join(insights)

    def query(self, user_question: str) -> Dict[str, Any]:
        """Main method to query the AI agent."""
        logger.info(f"Processing query: {user_question[:100]}...")
        
        start_time = datetime.now()
        
        # Route to appropriate backend
        if self.backend == "ollama":
            response = self._query_ollama(user_question)
        elif self.backend == "huggingface":
            response = self._query_huggingface(user_question)
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
        total_missing = sum(self.data_context['missing_values'].values())
        total_cells = self.data_context['shape']['rows'] * self.data_context['shape']['columns']
        missing_pct = total_missing / total_cells if total_cells > 0 else 0
        
        if missing_pct > 0.1:
            suggestions.append("How should I handle the missing values in this dataset?")
        
        # Based on size
        if self.data_context['shape']['rows'] > 10000:
            suggestions.append("What sampling strategies would work well for this large dataset?")
        
        suggestions.append("What machine learning approaches would be suitable for this data?")
        suggestions.append("What visualizations would best represent this data?")
        
        return suggestions[:5]  # Return top 5 suggestions