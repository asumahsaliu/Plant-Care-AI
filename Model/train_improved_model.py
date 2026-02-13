"""
Quick Start Training Script
===========================

This script provides a simple interface to train the improved model.

Usage:
    python train_improved_model.py --data_dir Dataset --model resnet50 --epochs 50

Requirements:
    - Dataset folder with plant disease images
    - GPU recommended (but works on CPU)
    - ~8GB RAM minimum
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from improved_model import (
    ImprovedPlantDiseaseModel,
    create_data_loaders,
    train_model,
    evaluate_model,
    plot_training_history
)


def main():
    parser = argparse.ArgumentParser(description='Train Improved Plant Disease Detection Model')
    
    parser.add_argument('--data_dir', type=str, default='Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--fine_tune_epoch', type=int, default=20,
                       help='Epoch to start fine-tuning all layers')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("\nPlease download the PlantVillage dataset from:")
        print("https://data.mendeley.com/datasets/tywbtsjrjv/1")
        print("\nOr update --data_dir to point to your dataset location")
        sys.exit(1)
    
    print("="*80)
    print("IMPROVED PLANT DISEASE DETECTION - TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Model: {args.model}")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Initial LR: {args.lr}")
    print(f"  Fine-tune Epoch: {args.fine_tune_epoch}")
    print(f"  Output Directory: {args.output_dir}")
    print("\n" + "="*80 + "\n")
    
    # Create data loaders
    print("📊 Loading and preparing data...")
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            args.data_dir,
            batch_size=args.batch_size
        )
        print(f"✓ Found {len(class_names)} classes")
        print(f"✓ Train samples: {len(train_loader.sampler)}")
        print(f"✓ Validation samples: {len(val_loader.sampler)}")
        print(f"✓ Test samples: {len(test_loader.sampler)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Create model
    print(f"\n🧠 Building {args.model} model...")
    try:
        model = ImprovedPlantDiseaseModel(
            num_classes=len(class_names),
            model_type=args.model,
            pretrained=True
        )
        print(f"✓ Model created successfully")
        print(f"✓ Using device: {model.device}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        sys.exit(1)
    
    # Train model
    print(f"\n🚀 Starting training...")
    print("="*80)
    try:
        history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            initial_lr=args.lr,
            fine_tune_epoch=args.fine_tune_epoch
        )
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Partial results may be available")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Plot training history
    print("\n📈 Generating training plots...")
    try:
        plot_training_history(history)
        print("✓ Training history saved")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
    
    # Evaluate on test set
    print("\n🎯 Evaluating on test set...")
    print("="*80)
    try:
        cm, report = evaluate_model(model, test_loader, class_names)
        print("\n✓ Evaluation completed!")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("  ✓ best_plant_model.pt - Model weights (use this for deployment)")
    print("  ✓ confusion_matrix.png - Visual performance analysis")
    print("  ✓ training_history.png - Training/validation curves")
    print("  ✓ classification_report.json - Detailed metrics per class")
    
    print("\n📝 Next steps:")
    print("  1. Review confusion_matrix.png to identify problem classes")
    print("  2. Check classification_report.json for per-class metrics")
    print("  3. If accuracy is low, consider:")
    print("     - Training for more epochs")
    print("     - Collecting more data for poorly performing classes")
    print("     - Adjusting augmentation parameters")
    print("  4. Deploy best_plant_model.pt to your Flask app")
    
    print("\n💡 To use the trained model in Flask:")
    print("  1. Copy best_plant_model.pt to 'Flask Deployed App' folder")
    print("  2. Update app.py to load the improved model")
    print("  3. Restart the Flask server")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
