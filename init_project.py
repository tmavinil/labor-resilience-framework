import os

def create_project_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    subfolders = [
        "core_engine",
        "forecasting",
        "benchmarking",
        "intelligence_layer",
        "interpretability",
        "optimization",
        "api",
        "examples"
    ]
    
    print(f"Initializing project structure in: {base_dir}")
    
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create an __init__.py in each folder to make them valid Python packages
        init_file = os.path.join(folder_path, "__init__.py")
        with open(init_file, "w") as f:
            pass
            
        print(f"Created: {folder}/")

if __name__ == "__main__":
    create_project_structure()
    print("Project directory structure generated successfully.")
