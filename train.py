"""
Main training script for 2048 DQN agent.

Usage:
    python train.py                    # Train from scratch
    python train.py --resume           # Resume from checkpoint
    python train.py --episodes 5000    # Train for specific number of episodes
    python train.py --eval             # Evaluate existing model
"""

import argparse
import os
import torch

from training.trainer import Trainer
from training.config import config
from agent.dqn_agent import DQNAgent


def main():
    """Main training function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN agent for 2048')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train (default: from config)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate existing model instead of training')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render games during evaluation')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to specific model checkpoint to load')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("🎮 2048 DQN Training")
    print("="*60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        config.DEVICE = "cuda"
    else:
        print("⚠️  No GPU detected, using CPU (will be slower)")
        config.DEVICE = "cpu"
    
    # Create or load agent
    agent = DQNAgent(config)
    
    if args.resume or args.load:
        # Load checkpoint
        if args.load:
            checkpoint_path = args.load
        else:
            checkpoint_path = os.path.join(config.MODEL_DIR, "model_latest.pth")
        
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"✅ Resumed from: {checkpoint_path}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Starting fresh training...")
    
    # Create trainer
    trainer = Trainer(config, agent=agent)
    
    if args.eval:
        # Evaluation mode
        print("\n🧪 Evaluation Mode")
        print(f"Episodes: {args.eval_episodes}")
        print(f"Render: {args.render}")
        print()
        
        trainer.evaluate(num_episodes=args.eval_episodes, render=args.render)
    else:
        # Training mode
        num_episodes = args.episodes if args.episodes else config.MAX_EPISODES
        
        print(f"\n🏋️ Training Configuration:")
        print(f"   Episodes: {num_episodes}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Learning rate: {config.LEARNING_RATE}")
        print(f"   Epsilon: {agent.epsilon:.3f} → {config.EPSILON_MIN}")
        print(f"   Replay memory: {config.REPLAY_MEMORY_SIZE}")
        print(f"   Device: {config.DEVICE}")
        print()
        
        input("Press Enter to start training (Ctrl+C to cancel)...")
        print()
        
        try:
            trainer.train(num_episodes=num_episodes, verbose=True)
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("Saving checkpoint before exit...")
            trainer._save_checkpoint(agent.episodes_trained)
            print("✅ Checkpoint saved")
    
    print("\n" + "="*60)
    print("✨ Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()