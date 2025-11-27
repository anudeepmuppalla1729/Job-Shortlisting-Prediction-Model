import pandas as pd
import joblib
import os
import sys

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import decisionTreeTraining
import linearRegressionTraining
import randomForestTraining
import svrTraining
import gradientBoostingTraining

def main():
    print("Loading dataset...")
    # Load dataset using one of the modules (they all have the same load_dataset)
    try:
        df = decisionTreeTraining.load_dataset()
    except FileNotFoundError:
        # Try adjusting path if running from root
        print("Dataset not found with default path. Trying adjusted path...")
        df = decisionTreeTraining.load_dataset("data/processed/feature_extracted_dataset.csv")

    results = {}
    
    print("\n--- Training Decision Tree ---")
    results["Decision Tree"] = decisionTreeTraining.train_model(df)
    
    print("\n--- Training Linear Regression ---")
    results["Linear Regression"] = linearRegressionTraining.train_model(df)
    
    print("\n--- Training Random Forest ---")
    results["Random Forest"] = randomForestTraining.train_model(df)
    
    print("\n--- Training SVR ---")
    results["SVR"] = svrTraining.train_model(df)
    
    print("\n--- Training Gradient Boosting ---")
    results["Gradient Boosting"] = gradientBoostingTraining.train_model(df)
    
    # Compare
    best_model_name = None
    best_mse = float("inf")
    
    print("\n\n--- Model Comparison ---")
    for name, res in results.items():
        mse = res["mse"]
        r2 = res["r2"]
        print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")
        
        # Criteria: Lowest MSE
        if mse < best_mse:
            best_mse = mse
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with MSE={best_mse:.4f}")
    
    # Save best model
    best_model = results[best_model_name]["model"]
    
    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)
    
    save_path = "../models/best_model.joblib"
    joblib.dump(best_model, save_path)
    print(f"Best model saved to {save_path}")

if __name__ == "__main__":
    main()
