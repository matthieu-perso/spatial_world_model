import argparse
import traceback
import time
import sys
from pathlib import Path

# Import your modules
from src.data.data import generate_single_sentence_dataset, extract_embeddings
# Import other modules

def parse_args():
    parser = argparse.ArgumentParser(description="Debug spatial world model")
    parser.add_argument("--small", action="store_true", help="Run with small dataset")
    parser.add_argument("--layer", type=int, default=24, help="Layer to analyze")
    return parser.parse_args()

def main(args):
    try:
        print("Step 1: Initializing...")
        # Your initialization code here
        
        print("Step 2: Loading objects and relations...")
        # Use small subset if requested
        if args.small:
            objects = ["book", "cup", "lamp", "phone", "remote"]
            relations = ["above", "below", "to the left of", "to the right of"]
        else:
            # Your full objects and relations loading
            pass
            
        print("Step 3: Generating dataset...")
        # Your dataset generation code with try/except blocks
        
        # Continue with other steps, each in their own try/except block
        
        print("✅ All steps completed successfully")
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    exit_code = main(args)
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")
    sys.exit(exit_code)
