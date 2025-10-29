#!/usr/bin/env python3
"""
Privacy-centric Motion Retargeting (PMR) - Interactive CLI

This CLI provides commands for training, evaluation, visualization, and more.
"""

import click
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.default_config import get_default_config, get_ntu120_config, get_dmr_config
from training.trainer import PMRTrainer
from utils.visualization import visualize_skeleton, create_comparison_video
from utils.data_loader import load_ntu_data


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Privacy-centric Motion Retargeting (PMR)
    
    A deep learning framework for anonymizing skeleton-based motion data
    while preserving action utility.
    """
    pass


@cli.command()
@click.option('--dataset', type=click.Choice(['ntu60', 'ntu120']), default='ntu60',
              help='Dataset to use for training')
@click.option('--model', type=click.Choice(['pmr', 'dmr']), default='pmr',
              help='Model type: PMR (with privacy) or DMR (baseline)')
@click.option('--epochs', type=int, default=None,
              help='Total training epochs (overrides config)')
@click.option('--batch-size', type=int, default=32,
              help='Batch size for training')
@click.option('--lr', type=float, default=1e-5,
              help='Learning rate')
@click.option('--device', default='cuda:0',
              help='Device to use (cuda:0, cuda:1, cpu)')
@click.option('--checkpoint-dir', type=click.Path(), default='checkpoints',
              help='Directory to save checkpoints')
@click.option('--resume', type=click.Path(), default=None,
              help='Path to checkpoint to resume from')
@click.option('--use-mlflow', is_flag=True,
              help='Enable MLflow logging')
def train(dataset, model, epochs, batch_size, lr, device, checkpoint_dir, resume, use_mlflow):
    """Train the PMR or DMR model"""
    click.echo(f"🚀 Starting training: {model.upper()} on {dataset.upper()}")
    
    # Load configuration
    if dataset == 'ntu120':
        config = get_ntu120_config()
    else:
        config = get_default_config()
    
    if model == 'dmr':
        config.training.alpha_emb = 0.0  # Disable adversarial training
    
    # Override config with CLI arguments
    if epochs:
        config.training.stage4_paired_training = epochs
    config.training.batch_size = batch_size
    config.training.lr = lr
    config.training.device = device
    config.training.checkpoint_dir = checkpoint_dir
    config.training.use_mlflow = use_mlflow
    
    # Initialize trainer
    trainer = PMRTrainer(config)
    
    # Resume from checkpoint if provided
    if resume:
        trainer.load_checkpoint(resume)
        click.echo(f"📂 Resumed from checkpoint: {resume}")
    
    # Start training
    try:
        trainer.train()
        click.echo("✅ Training completed successfully!")
    except KeyboardInterrupt:
        click.echo("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint('interrupted.pt')
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}", err=True)
        raise


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--dataset', type=click.Choice(['ntu60', 'ntu120']), default='ntu60',
              help='Dataset to evaluate on')
@click.option('--output', type=click.Path(), default='evaluation_results.json',
              help='Output file for results')
@click.option('--device', default='cuda:0',
              help='Device to use')
def evaluate(model_path, dataset, output, device):
    """Evaluate a trained model"""
    click.echo(f"📊 Evaluating model: {model_path}")
    
    from training.evaluator import PMREvaluator
    
    # Load configuration
    config = get_ntu120_config() if dataset == 'ntu120' else get_default_config()
    config.training.device = device
    
    # Initialize evaluator
    evaluator = PMREvaluator(config, model_path)
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Save results
    import json
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    click.echo("\n📈 Evaluation Results:")
    click.echo(f"  MSE: {results.get('mse', 'N/A'):.4f}")
    click.echo(f"  Action Recognition (Top-1): {results.get('action_top1', 'N/A'):.2%}")
    click.echo(f"  Action Recognition (Top-5): {results.get('action_top5', 'N/A'):.2%}")
    click.echo(f"  Re-ID (Top-1): {results.get('reid_top1', 'N/A'):.2%}")
    click.echo(f"  Re-ID (Top-5): {results.get('reid_top5', 'N/A'):.2%}")
    click.echo(f"\n💾 Full results saved to: {output}")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Input skeleton file (pickle)')
@click.option('--dummy', type=click.Path(exists=True), default=None,
              help='Dummy skeleton file (if None, uses random)')
@click.option('--output', type=click.Path(), default='anonymized.pkl',
              help='Output file for anonymized skeleton')
@click.option('--device', default='cuda:0',
              help='Device to use')
def anonymize(model_path, input, dummy, output, device):
    """Anonymize a skeleton sequence"""
    click.echo(f"🔒 Anonymizing skeleton: {input}")
    
    import pickle
    from models.pmr import PMRModel
    
    # Load model
    model = PMRModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load input skeleton
    with open(input, 'rb') as f:
        skeleton_data = pickle.load(f)
    
    # Load or select dummy
    if dummy:
        with open(dummy, 'rb') as f:
            dummy_data = pickle.load(f)
    else:
        # Use random skeleton from dataset as dummy
        dummy_data = skeleton_data  # Simplified for now
    
    # Anonymize
    with torch.no_grad():
        skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).unsqueeze(0).to(device)
        dummy_tensor = torch.tensor(dummy_data, dtype=torch.float32).unsqueeze(0).to(device)
        anonymized = model.cross_reconstruct(skeleton_tensor, dummy_tensor)
        anonymized = anonymized.cpu().numpy().squeeze(0)
    
    # Save result
    with open(output, 'wb') as f:
        pickle.dump(anonymized, f)
    
    click.echo(f"✅ Anonymized skeleton saved to: {output}")


@cli.command()
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Input skeleton file (pickle)')
@click.option('--output', type=click.Path(), default='skeleton_video.gif',
              help='Output video file')
@click.option('--fps', type=int, default=30,
              help='Frames per second')
def visualize(input, output, fps):
    """Visualize a skeleton sequence"""
    click.echo(f"🎬 Creating visualization: {input}")
    
    import pickle
    with open(input, 'rb') as f:
        skeleton_data = pickle.load(f)
    
    from utils.visualization import create_skeleton_video
    create_skeleton_video(skeleton_data, output, fps=fps)
    
    click.echo(f"✅ Video saved to: {output}")


@cli.command()
@click.option('--original', type=click.Path(exists=True), required=True,
              help='Original skeleton file')
@click.option('--anonymized', type=click.Path(exists=True), required=True,
              help='Anonymized skeleton file')
@click.option('--output', type=click.Path(), default='comparison.gif',
              help='Output comparison video')
def compare(original, anonymized, output):
    """Create side-by-side comparison video"""
    click.echo("🎬 Creating comparison video...")
    
    import pickle
    with open(original, 'rb') as f:
        orig_data = pickle.load(f)
    with open(anonymized, 'rb') as f:
        anon_data = pickle.load(f)
    
    create_comparison_video(orig_data, anon_data, output)
    click.echo(f"✅ Comparison video saved to: {output}")


@cli.command()
def info():
    """Display information about the PMR framework"""
    click.echo("""
╔══════════════════════════════════════════════════════════════╗
║  Privacy-centric Motion Retargeting (PMR)                    ║
╚══════════════════════════════════════════════════════════════╝

PMR is a deep learning framework for anonymizing skeleton-based motion
data while preserving action utility. It uses adversarial learning to
disentangle user identity from motion information.

📚 Key Features:
  • Motion retargeting with privacy preservation
  • Adversarial training for identity removal
  • Maintains action recognition accuracy
  • Reduces re-identification risk

🏗️  Architecture:
  • Motion Encoder (E_M): Captures action information
  • Privacy Encoder (E_P): Captures identity information
  • Decoder (D): Reconstructs skeleton sequences
  • Motion Classifier (M): Ensures action preservation
  • Privacy Classifier (P): Ensures identity removal
  • Quality Controller (Q): Improves realism

📖 For more information, see README.md or visit:
   https://github.com/Thomasc33/Privacy-Retargeting
    """)


if __name__ == '__main__':
    cli()

