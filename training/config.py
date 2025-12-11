"""
Hyperparameters for training the 2048 DQN agent.

Key concepts:
- Learning rate: How much to adjust weights after each batch
- Gamma (discount): How much to value future rewards vs immediate rewards
- Epsilon: Exploration vs exploitation tradeoff
- Batch size: Number of experiences to learn from at once
- Replay memory: How many past experiences to remember
"""


class Config:
    """Training configuration and hyperparameters."""
    
    # === Environment ===
    BOARD_SIZE = 4
    ACTION_SPACE = 4  # Up, Right, Down, Left
    
    # === Neural Network Architecture ===
    # Input: 4x4 board → Conv layers → Dense layers → 4 Q-values (one per action)
    INPUT_SHAPE = (1, BOARD_SIZE, BOARD_SIZE)  # (channels, height, width)
    
    # CNN architecture
    CONV1_FILTERS = 128
    CONV1_KERNEL = 2
    CONV2_FILTERS = 128
    CONV2_KERNEL = 2
    DENSE_UNITS = 256
    
    # === Training Hyperparameters ===
    
    # Learning rate: How fast the network learns
    # Too high (0.01) → unstable, bounces around
    # Too low (0.00001) → learns too slowly
    # 0.001 is the sweet spot for most problems
    LEARNING_RATE = 0.001
    
    # Gamma (discount factor): How much we value future rewards
    # 0.99 = "I care about long-term strategy"
    # 0.5 = "I only care about immediate rewards"
    # For 2048, we want long-term thinking → 0.99
    GAMMA = 0.99
    
    # Batch size: How many experiences to train on at once
    # Larger = more stable but slower
    # 512 is good for 2048 (enough variety, not too slow)
    BATCH_SIZE = 128
    
    # Replay memory size: How many past experiences to remember
    # 10,000 games worth of experience
    # Helps prevent "forgetting" old strategies
    REPLAY_MEMORY_SIZE = 10000
    
    # === Exploration vs Exploitation (Epsilon-Greedy) ===
    # Key concept: Agent needs to balance:
    #   - Exploration: Try random moves to discover new strategies
    #   - Exploitation: Use learned policy to maximize score
    
    # Start exploring 100% (completely random)
    EPSILON_START = 1.0
    
    # End at 1% exploration (mostly use learned policy)
    EPSILON_MIN = 0.01
    
    # Decay rate: How fast we go from exploring → exploiting
    # 0.995 means epsilon *= 0.995 after each episode
    # Reaches ~0.01 after about 1000 episodes
    EPSILON_DECAY = 0.997
    
    # === Target Network ===
    # DQN trick: Use two networks:
    #   - Main network: Updates every batch
    #   - Target network: Updates every N episodes (more stable)
    # This prevents the "chasing a moving target" problem
    TARGET_UPDATE_FREQUENCY = 100  # Update target network every 100 episodes
    
    # === Training Settings ===
    MAX_EPISODES = 10000  # Total games to play during training
    
    # Save model checkpoints
    SAVE_FREQUENCY = 100  # Save every 100 episodes
    MODEL_DIR = "models"
    LOG_DIR = "logs"
    
    # === Reward Shaping ===
    # How we give feedback to the agent
    # This is CRITICAL - bad rewards = agent learns nothing
    
    # Reward for merging tiles (gets the merged value)
    # e.g., merge two 8s → reward = 16
    MERGE_REWARD_MULTIPLIER = 1.0
    
    # Bonus for reaching 2048 (the goal!)
    WIN_REWARD = 10000
    
    # Penalty for invalid moves (trying to move when nothing moves)
    INVALID_MOVE_PENALTY = -1
    
    # Small penalty per move (encourages efficiency)
    # Agent learns to win in fewer moves
    MOVE_PENALTY = -0.1
    
    # Game over penalty (losing is bad!)
    GAME_OVER_PENALTY = -20
    
    # === Logging ===
    PRINT_FREQUENCY = 10  # Print stats every 10 episodes
    
    # === Device (CPU vs GPU) ===
    # Will be set automatically in training code
    DEVICE = "cpu"  # or "cuda" if GPU available


# Create global config instance
config = Config()


# Quick sanity check
if __name__ == "__main__":
    print("🔧 Configuration Settings")
    print("=" * 50)
    print(f"Board size: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
    print(f"Action space: {config.ACTION_SPACE} moves")
    print(f"\n📚 Training:")
    print(f"  Episodes: {config.MAX_EPISODES}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Gamma: {config.GAMMA}")
    print(f"\n🎯 Exploration:")
    print(f"  Epsilon start: {config.EPSILON_START}")
    print(f"  Epsilon min: {config.EPSILON_MIN}")
    print(f"  Epsilon decay: {config.EPSILON_DECAY}")
    print(f"\n🧠 Network:")
    print(f"  Conv layers: 2 ({config.CONV1_FILTERS}, {config.CONV2_FILTERS} filters)")
    print(f"  Dense units: {config.DENSE_UNITS}")
    print(f"\n💾 Memory:")
    print(f"  Replay size: {config.REPLAY_MEMORY_SIZE} transitions")
    print(f"  Target update freq: {config.TARGET_UPDATE_FREQUENCY} episodes")
    print("\n✅ Config loaded successfully!")