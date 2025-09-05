"""
Install and setup script for enhanced attention visualization and episodic memory
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for attention visualization"""
    
    required_packages = [
        "plotly>=5.0.0",
        "seaborn>=0.11.0", 
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "bertviz",  # Optional - for BertViz integration
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "wandb>=0.13.0"
    ]
    
    print("🚀 Installing required packages for attention visualization...")
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"⚠️  Warning: {package} installation had issues: {result.stderr}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "./attention_visualizations",
        "./memory_attention_analysis", 
        "./external_episodic_memory",
        "./logs/attention_logs",
        "./checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run training with: python train_unified.py configs/bitmar_with_enhanced_memory_attention.yaml")
    print("2. Monitor attention visualizations in ./attention_visualizations/")
    print("3. Check external memory storage in ./external_episodic_memory/")
    print("4. View WandB dashboard for real-time analysis")

if __name__ == "__main__":
    install_packages()
