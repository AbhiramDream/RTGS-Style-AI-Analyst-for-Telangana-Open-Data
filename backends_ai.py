import subprocess
import sys
import os
import requests
import time
from pathlib import Path

def install_ollama():
    """Install and setup Ollama."""
    print(" Setting up Ollama...")
    
    try:
        # Check if ollama is already installed
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        print(f" Ollama already installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print(" Installing Ollama...")
        
        if sys.platform.startswith('linux'):
            subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], 
                         stdout=subprocess.PIPE)
        elif sys.platform == 'darwin':  # macOS
            print("Please install Ollama manually from https://ollama.com")
            return False
        elif sys.platform.startswith('win'):
            print("Please install Ollama manually from https://ollama.com")
            return False
    
    # Pull model
    print(" Pulling llama3.1 model (this may take a while)...")
    try:
        subprocess.run(["ollama", "pull", "llama3.1"], check=True)
        print(" llama3.1 model ready!")
    except subprocess.CalledProcessError:
        print(" Failed to pull llama3.1, trying llama2...")
        subprocess.run(["ollama", "pull", "llama2"], check=True)
        print(" llama2 model ready!")
    
    return True

def install_huggingface():
    """Install Hugging Face transformers."""
    print("🔧 Setting up Hugging Face Transformers...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "transformers", "torch", "tokenizers"], check=True)
        print(" Hugging Face Transformers installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to install Hugging Face: {e}")
        return False

def test_backends():
    """Test all available backends."""
    print("\n Testing AI backends...")
    
    # Test Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f" Ollama: {len(models)} models available")
        else:
            print(" Ollama: Server not responding (run 'ollama serve')")
    except:
        print(" Ollama: Not available")
    
    # Test Hugging Face
    try:
        import transformers
        print(f" Hugging Face Transformers: v{transformers.__version__}")
    except ImportError:
        print(" Hugging Face Transformers: Not installed")

def main():
    """Main setup function."""
    print(" Setting up AI backends for Data Pipeline...")
    
    backends = input("Which backends to setup? (ollama/huggingface/both) [both]: ").lower() or "both"
    
    if backends in ["ollama", "both"]:
        install_ollama()
    
    if backends in ["huggingface", "both"]:
        install_huggingface()
    
    test_backends()
    
    print("\n Setup complete!")
    print("\nNext steps:")
    print("1. Start Ollama server: ollama serve")
    print("2. Run your pipeline: python -m pipeline.cli your_data.csv --interactive")

if __name__ == "__main__":
    main()