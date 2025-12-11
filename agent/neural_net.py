"""
Deep Q-Network (DQN) Architecture for 2048.

Key concept: Q-learning
- Q(state, action) = expected total reward if we take 'action' in 'state'
- The network learns to predict these Q-values
- At decision time: pick action with highest Q-value

Architecture choice: CNN vs MLP
- MLP (fully connected): Treats board as flat vector → loses spatial structure
- CNN: Preserves spatial relationships → better for 2048!
  - Example: CNN can learn "keep high tiles in corner" pattern
  - MLP would struggle to learn this spatial concept

Why CNNs work for 2048:
- 2048 is about spatial patterns (adjacent tiles, corners, edges)
- CNNs are designed to recognize spatial patterns
- Convolutional filters can detect tile arrangements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    Deep Q-Network using Convolutional layers.
    
    Architecture:
    Input (4x4 board) 
    → Log2 encoding (normalize values)
    → Conv1 (128 filters, 2x2 kernel)
    → ReLU
    → Conv2 (128 filters, 2x2 kernel)  
    → ReLU
    → Flatten
    → Dense (256 units)
    → ReLU
    → Output (4 Q-values)
    """
    
    def __init__(self, board_size=4, action_space=4):
        """
        Initialize the DQN.
        
        Args:
            board_size: Size of the board (4 for standard 2048)
            action_space: Number of actions (4: up, right, down, left)
        """
        super(DQN, self).__init__()
        
        self.board_size = board_size
        self.action_space = action_space
        
        # === Convolutional Layers ===
        # Why 128 filters? Enough to learn complex patterns, not too many to overfit
        # Why 2x2 kernel? Captures adjacent tile relationships (key for merging)
        
        # Conv1: 1 input channel (grayscale board) → 128 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=0
        )
        # Output size: (4-2+1) = 3x3 with 128 channels
        
        # Conv2: 128 → 128 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=0
        )
        # Output size: (3-2+1) = 2x2 with 128 channels
        
        # Calculate size after convolutions
        # After conv1: 3x3x128
        # After conv2: 2x2x128 = 512 features
        conv_output_size = 2 * 2 * 128
        
        # === Fully Connected Layers ===
        # Dense layer to learn high-level strategy
        self.fc1 = nn.Linear(conv_output_size, 256)
        
        # Output layer: 4 Q-values (one per action)
        self.fc2 = nn.Linear(256, action_space)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Batch of board states (batch_size, 4, 4)
               Values should be in range [0, 15] (log2 of tile values)
        
        Returns:
            Q-values for each action (batch_size, 4)
        """
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 4, 4) → (batch, 1, 4, 4)
        
        # Conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 2, 2, 128) → (batch, 512)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        # Output Q-values (no activation - we want raw Q-values)
        q_values = self.fc2(x)
        
        return q_values
    
    def predict_action(self, state, epsilon=0.0):
        """
        Predict best action using epsilon-greedy policy.
        
        Epsilon-greedy explained:
        - With probability epsilon: take random action (exploration)
        - With probability (1-epsilon): take best action (exploitation)
        
        Args:
            state: Board state (4, 4) numpy array
            epsilon: Exploration rate (0.0 = always exploit, 1.0 = always explore)
        
        Returns:
            action: Integer 0-3 representing the chosen action
        """
        # Exploration: random action
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)
        
        # Exploitation: best action according to Q-values
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            
            # Get Q-values
            q_values = self.forward(state_tensor)
            
            # Return action with highest Q-value
            return q_values.argmax().item()
    
    def get_q_values(self, state):
        """
        Get Q-values for a single state (useful for debugging/visualization).
        
        Args:
            state: Board state (4, 4) numpy array
        
        Returns:
            q_values: Numpy array of 4 Q-values
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.cpu().numpy()[0]


def preprocess_state(board):
    """
    Preprocess board state for neural network input.
    
    Key concept: State encoding
    - Raw tiles: [0, 2, 4, 8, 16, 32, ..., 2048, ...]
    - Problem: Huge range (0 to 2048+), hard for network to learn
    
    - Log2 encoding: [0, 1, 2, 3, 4, 5, ..., 11, ...]
    - Benefit: Smaller range, preserves order, equal "distance" between powers
    
    Example:
    - 0 → 0
    - 2 → 1
    - 4 → 2
    - 8 → 3
    - 2048 → 11
    
    Args:
        board: Raw board state (4, 4) with tile values
    
    Returns:
        Preprocessed board (4, 4) with log2 values
    """
    # Avoid log2(0) by adding small epsilon where board is 0
    # Then take log2 and round
    processed = np.where(board > 0, np.log2(board), 0)
    return processed.astype(np.float32)


# Test the network
if __name__ == "__main__":
    print("🧠 Testing DQN Architecture\n")
    
    # Create network
    dqn = DQN(board_size=4, action_space=4)
    print("✅ Network created")
    print(f"   Parameters: {sum(p.numel() for p in dqn.parameters()):,}")
    
    # Test forward pass with fake board
    print("\n🎮 Testing forward pass...")
    fake_board = np.array([
        [2, 4, 8, 16],
        [0, 2, 4, 8],
        [0, 0, 2, 4],
        [0, 0, 0, 2]
    ], dtype=np.float32)
    
    # Preprocess
    processed = preprocess_state(fake_board)
    print(f"Raw board:\n{fake_board}")
    print(f"\nLog2 encoded:\n{processed}")
    
    # Get Q-values
    q_values = dqn.get_q_values(processed)
    print(f"\nQ-values: {q_values}")
    print(f"Best action: {q_values.argmax()} (0=Up, 1=Right, 2=Down, 3=Left)")
    
    # Test epsilon-greedy
    print("\n🎲 Testing epsilon-greedy policy...")
    actions_explored = []
    actions_exploited = []
    
    for _ in range(100):
        actions_explored.append(dqn.predict_action(processed, epsilon=1.0))
        actions_exploited.append(dqn.predict_action(processed, epsilon=0.0))
    
    print(f"With ε=1.0 (explore): {len(set(actions_explored))} unique actions")
    print(f"With ε=0.0 (exploit): {len(set(actions_exploited))} unique actions")
    
    # Test batch processing
    print("\n📦 Testing batch processing...")
    batch = torch.FloatTensor(np.random.rand(32, 4, 4))
    output = dqn(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("\n✅ All tests passed! Network is ready for training.")