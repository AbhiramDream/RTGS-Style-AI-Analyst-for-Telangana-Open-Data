import argparse
import sys
import os
import pickle
from pathlib import Path
from loguru import logger

# Import pipeline components
from data_loader import DataLoader
from data_cleaner import DataCleaner
from imputer import DataImputer
from analyzer import Analyzer
from reporter import Reporter
from ai_agent import DataInsightsAgent


class ModularDataPipeline:
    def __init__(self):
        self.data_dir = Path("data_cache")
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        logger.remove()
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>"
        )
        logger.add(
            self.log_dir / "pipeline.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB"
        )

    def _save_dataframe(self, df, stage_name: str):
        """Save DataFrame to cache for next stage."""
        cache_file = self.data_dir / f"{stage_name}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Data cached at stage: {stage_name}")

    def _load_dataframe(self, stage_name: str):
        """Load DataFrame from cache."""
        cache_file = self.data_dir / f"{stage_name}.pkl"
        if not cache_file.exists():
            logger.error(f"No cached data found for stage: {stage_name}")
            logger.info("Available stages: " + ", ".join([f.stem for f in self.data_dir.glob("*.pkl")]))
            return None
        
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded data from stage: {stage_name}")
        return df

    def load(self, file_path: str):
        """Load data from CSV file."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            loader = DataLoader(file_path)
            df = loader.load()
            self._save_dataframe(df, "loaded")
            logger.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def clean(self):
        """Clean the loaded data."""
        df = self._load_dataframe("loaded")
        if df is None:
            return False
        
        try:
            cleaner = DataCleaner(df)
            cleaned_df = cleaner.clean()
            self._save_dataframe(cleaned_df, "cleaned")
            logger.success("Data cleaned successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            return False

    def impute(self, drop_threshold: float = 0.95):
        """Impute missing values in cleaned data."""
        df = self._load_dataframe("cleaned")
        if df is None:
            logger.info("No cleaned data found, trying loaded data...")
            df = self._load_dataframe("loaded")
            if df is None:
                return False
        
        try:
            imputer = DataImputer(df, drop_threshold=drop_threshold)
            imputed_df = imputer.impute()
            self._save_dataframe(imputed_df, "imputed")
            logger.success("Data imputation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to impute data: {e}")
            return False

    def analyze(self):
        """Analyze the processed data."""
        # Try to get the most processed version available
        df = None
        for stage in ["imputed", "cleaned", "loaded"]:
            df = self._load_dataframe(stage)
            if df is not None:
                logger.info(f"Using {stage} data for analysis")
                break
        
        if df is None:
            logger.error("No data available for analysis")
            return False
        
        try:
            analyzer = Analyzer(df)
            results = analyzer.run_analysis()
            
            # Save analysis results
            analysis_file = self.log_dir / "analysis_results.json"
            import json
            with open(analysis_file, 'w') as f:
                # Convert numpy types to JSON serializable
                def convert_types(obj):
                    if hasattr(obj, 'item'):
                        return obj.item()
                    elif hasattr(obj, 'tolist'):
                        return obj.tolist()
                    else:
                        return obj
                
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[key] = {k: convert_types(v) for k, v in value.items()}
                    else:
                        json_results[key] = convert_types(value)
                
                json.dump(json_results, f, indent=4, default=str)
            
            logger.success(f"Analysis completed and saved to {analysis_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to analyze data: {e}")
            return False

    def query(self, question: str = None, backend: str = "ollama", model: str = "llama3.1", interactive: bool = False):
        """Query the data using AI agent."""
        # Try to get the most processed version available
        df = None
        stage_used = None
        for stage in ["imputed", "cleaned", "loaded"]:
            df = self._load_dataframe(stage)
            if df is not None:
                stage_used = stage
                break
        
        if df is None:
            logger.error("No data available for querying. Please load data first.")
            return False
        
        logger.info(f"Using {stage_used} data for AI queries")
        
        try:
            agent = DataInsightsAgent(df, backend=backend, model=model)
            
            if interactive:
                self._interactive_mode(agent)
            elif question:
                result = agent.query(question)
                print(f"\n🤖 AI Response:")
                print("=" * 60)
                print(result['response'])
                print("=" * 60)
                print(f"Backend: {result['backend']}/{result['model']} | Time: {result['response_time_seconds']:.2f}s")
            else:
                # Show suggestions
                suggestions = agent.suggest_analyses()
                print("\n💡 Suggested questions for your dataset:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion}")
                print(f"\nUse --query 'your question' or --interactive for AI chat mode.")
            
            return True
        except Exception as e:
            logger.error(f"Failed to query data: {e}")
            return False

    def _interactive_mode(self, agent):
        """Interactive query mode."""
        print("\n" + "="*60)
        print("🤖 AI Data Insights - Interactive Mode")
        print("="*60)
        print(f"Dataset: {agent.data_context['shape']['rows']} rows, {agent.data_context['shape']['columns']} columns")
        print(f"AI Backend: {agent.backend}/{agent.model}")
        print("Commands: 'quit' to exit, 'suggestions' for ideas, 'info' for dataset info")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("💬 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'suggestions':
                    suggestions = agent.suggest_analyses()
                    print("\n💡 Suggested questions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                    continue
                elif query.lower() == 'info':
                    print(f"\n📊 Dataset Info:")
                    print(f"Shape: {agent.data_context['shape']['rows']} rows × {agent.data_context['shape']['columns']} columns")
                    print(f"Columns: {', '.join(agent.data_context['columns'][:5])}{'...' if len(agent.data_context['columns']) > 5 else ''}")
                    print(f"Data completeness: {agent.data_context['data_quality']['completeness']}")
                    continue
                elif not query:
                    continue
                
                print("\n Analyzing...")
                result = agent.query(query)
                
                print(f"\n Response ({result['response_time_seconds']:.1f}s):")
                print("-" * 50)
                print(result['response'])
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f" Error: {e}")

    def status(self):
        """Show pipeline status."""
        print("\n Pipeline Status:")
        print("=" * 40)
        
        stages = ["loaded", "cleaned", "imputed"]
        for stage in stages:
            cache_file = self.data_dir / f"{stage}.pkl"
            if cache_file.exists():
                # Get file info
                size = cache_file.stat().st_size / 1024  # KB
                modified = cache_file.stat().st_mtime
                from datetime import datetime
                mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")
                print(f" {stage.capitalize()}: {size:.1f} KB (modified: {mod_time})")
            else:
                print(f" {stage.capitalize()}: Not available")
        
        # Check if analysis results exist
        analysis_file = self.log_dir / "analysis_results.json"
        if analysis_file.exists():
            print(f" Analysis: Available")
        else:
            print(f" Analysis: Not run")
        
        print("=" * 40)


def main():
    """Main CLI entry point with individual commands."""
    parser = argparse.ArgumentParser(
        description="Modular AI-Powered Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load data
  python cli.py load data.csv
  
  # Clean loaded data  
  python cli.py clean
  
  # Impute missing values
  python cli.py impute --drop-threshold 0.9
  
  # Analyze processed data
  python cli.py analyze
  
  # Query with AI (interactive)
  python cli.py query --interactive
  
  # Query with AI (single question)  
  python cli.py query --question "What insights do you see?"
  
  # Check pipeline status
  python cli.py status
  
  # Full pipeline
  python cli.py load data.csv && python cli.py clean && python cli.py impute && python cli.py analyze
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load CSV data')
    load_parser.add_argument('file_path', help='Path to CSV file')
    
    # Clean command
    subparsers.add_parser('clean', help='Clean loaded data')
    
    # Impute command
    impute_parser = subparsers.add_parser('impute', help='Impute missing values')
    impute_parser.add_argument('--drop-threshold', type=float, default=0.95,
                              help='Drop columns with missing ratio > threshold (default: 0.95)')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Analyze processed data')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query data with AI')
    query_parser.add_argument('--question', help='Single question to ask')
    query_parser.add_argument('--interactive', action='store_true', help='Interactive query mode')
    query_parser.add_argument('--backend', choices=['ollama', 'huggingface'], default='ollama',
                             help='AI backend (default: ollama)')
    query_parser.add_argument('--model', default='llama3.1', help='AI model (default: llama3.1)')
    
    # Status command
    subparsers.add_parser('status', help='Show pipeline status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = ModularDataPipeline()
    
    # Execute command
    success = False
    if args.command == 'load':
        success = pipeline.load(args.file_path)
    elif args.command == 'clean':
        success = pipeline.clean()
    elif args.command == 'impute':
        success = pipeline.impute(args.drop_threshold)
    elif args.command == 'analyze':
        success = pipeline.analyze()
    elif args.command == 'query':
        success = pipeline.query(
            question=getattr(args, 'question', None),
            backend=getattr(args, 'backend', 'ollama'),
            model=getattr(args, 'model', 'llama3.1'),
            interactive=getattr(args, 'interactive', False)
        )
    elif args.command == 'status':
        pipeline.status()
        success = True
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()