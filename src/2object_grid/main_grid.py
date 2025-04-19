import os
import torch
import numpy as np
import argparse
from tqdm.auto import tqdm

from data_grid import (
    setup_language_model, 
    generate_diverse_spatial_sentences,
    extract_and_save_activations, 
    load_activations,
    prepare_data_from_saved,
    GRID_SIZE,
    OBJECTS
)

from prob_grid import (
    LinearSpatialProbe,
    train_model,
    save_probe_model,
    load_saved_probe,
    test_opposite_relationships,
    run_comprehensive_tests
)

from compositionality_grid import run_pca_analysis

def main(args):
    """Main function with options to extract new activations and train new model."""
    # Setup paths
    activation_file = args.activation_file
    model_file = args.model_file
    
    # Step 1: Get activations (extract new or load saved)
    if args.extract_new:
        print("Step 1: Generating new training data...")
        # Setup LLM
        print(f"  Setting up language model: {args.model_name}")
        model, tokenizer = setup_language_model(args.model_name, token=args.hf_token)
        
        print(f"  Generating {args.num_sentences} sentences for spatial relationships...")
        sentences = generate_diverse_spatial_sentences(num_sentences=args.num_sentences)
        
        print(f"  Extracting activations and saving to {activation_file}...")
        data_dict = extract_and_save_activations(
            sentences, model, tokenizer, save_path=activation_file
        )
    else:
        print(f"Step 1: Loading saved activations from {activation_file}...")
        data_dict = load_activations(load_path=activation_file)

    # Step 2: Train probe (train new or load saved)
    if args.train_new:
        print("\nStep 2: Setting up and training the probe...")
        # Setup LLM if not already done
        if not args.extract_new:
            print(f"  Setting up language model: {args.model_name}")
            model, tokenizer = setup_language_model(args.model_name, token=args.hf_token)
        
        # Prepare data
        print("  Preparing data for training...")
        dataloader = prepare_data_from_saved(data_dict, batch_size=args.batch_size)

        # Initialize model
        print("  Initializing probe model...")
        d_model = model.config.hidden_size
        probe = LinearSpatialProbe(d_model, GRID_SIZE, len(OBJECTS) + 1).cuda()

        # Train model
        print(f"  Training probe for {args.epochs} epochs...")
        probe = train_model(probe, dataloader, num_epochs=args.epochs)

        # Save model with metadata
        print(f"  Saving trained probe to {model_file}...")
        save_probe_model(probe, d_model, GRID_SIZE, OBJECTS, save_path=model_file)
    else:
        # Load existing model
        print(f"\nStep 2: Loading saved probe from {model_file}...")
        probe, model_info = load_saved_probe(model_file)

    # Step 3: Testing
    if args.run_tests:
        print("\nStep 3: Testing the probe...")
        # Setup LLM if not already done
        if not args.extract_new and not args.train_new:
            print(f"  Setting up language model: {args.model_name}")
            model, tokenizer = setup_language_model(args.model_name, token=args.hf_token)
            
        # Test opposite relationships
        if args.test_opposites:
            print("  Testing opposite spatial relationships...")
            test_opposite_relationships(probe, tokenizer, model, save_plots=False)
        
        # Run comprehensive tests
        if args.test_comprehensive:
            print("  Running comprehensive test suite...")
            test_results = run_comprehensive_tests(probe, tokenizer, model)

    # Step 4: Compositionality Analysis
    if args.run_pca:
        print("\nStep 4: Running PCA analysis on spatial relationships...")
        pca_results = run_pca_analysis(model_path=model_file)
        print("PCA analysis complete.")

    print("\nAll tasks completed successfully!")
    return probe, data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Spatial Relationship Grid World Model")
    
    # Data generation arguments
    parser.add_argument("--extract_new", action="store_true", help="Extract new activations")
    parser.add_argument("--num_sentences", type=int, default=15000, help="Number of sentences to generate")
    parser.add_argument("--activation_file", type=str, default="spatial_activations.pt", help="Path to save/load activations")
    
    # Model training arguments
    parser.add_argument("--train_new", action="store_true", help="Train a new probe model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--model_file", type=str, default="spatial_3d_probe.pt", help="Path to save/load model")
    
    # Language model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B", help="HuggingFace model ID")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    
    # Testing arguments
    parser.add_argument("--run_tests", action="store_true", help="Run tests on the model")
    parser.add_argument("--test_opposites", action="store_true", help="Test opposite relationships")
    parser.add_argument("--test_comprehensive", action="store_true", help="Run comprehensive test suite")
    
    # Analysis arguments
    parser.add_argument("--run_pca", action="store_true", help="Run PCA analysis")
    
    args = parser.parse_args()
    
    # Set default behavior if no actions specified
    if not any([args.extract_new, args.train_new, args.run_tests, args.run_pca]):
        print("No actions specified, defaulting to loading model and running tests.")
        args.run_tests = True
        args.test_comprehensive = True
    
    # Run main function
    probe, data = main(args)
