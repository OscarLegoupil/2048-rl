"""
Experience Replay Memory for DQN.

Key concept: Store past experiences and randomly sample from them during training.

Why is this crucial?
1. Breaks correlation between consecutive experiences
   - Without this: agent sees (state1 → state2 → state3) in sequence
   - Problem: consecutive states are highly correlated → overfitting
   - With replay: agent sees random mix of experiences → better generalization

2. Reuses experiences multiple times
   - Each game is expensive (time to play)
   - Replay lets us learn from same experience multiple times
   - Improves sample efficiency

3. Stabilizes training
   - Random sampling smooths out the learning process
   - Prevents catastrophic forgetting (overwriting old knowledge)
"""

import numpy as np
import random
from collections import deque


class ReplayMemory:
    """
    Circular buffer to store game experiences.
    
    Each experience is a tuple: (state, action, reward, next_state, done)
    - state: Board before action (4x4 array)
    - action: What move was made (0-3)
    - reward: Points earned from this action
    - next_state: Board after action (4x4 array)
    - done: Is game over? (True/False)
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
                     Once full, oldest experiences are removed (FIFO)
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: Current board state (numpy array)
            action: Action taken (int 0-3)
            reward: Reward received (float)
            next_state: Resulting board state (numpy array)
            done: Whether episode ended (bool)
        """
        # Store as tuple
        # We'll convert to numpy arrays when sampling for efficiency
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Randomly sample a batch of experiences.
        
        This is the key to breaking correlation!
        Instead of learning from sequential experiences,
        we learn from a random mix.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of numpy arrays:
            - states: (batch_size, 4, 4)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - next_states: (batch_size, 4, 4)
            - dones: (batch_size,)
        """
        # Random sample without replacement
        batch = random.sample(self.memory, batch_size)
        
        # Unzip the batch into separate arrays
        # This converts list of tuples → tuple of lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays for PyTorch
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of memory."""
        return len(self.memory)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if we have enough experiences to sample a batch.
        
        We need at least batch_size experiences before we can start training.
        This ensures the agent has seen some variety before learning.
        """
        return len(self.memory) >= batch_size
    
    def clear(self):
        """Clear all experiences from memory."""
        self.memory.clear()
    
    def get_stats(self):
        """Get statistics about the replay memory."""
        if len(self.memory) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0
            }
        
        # Calculate utilization
        utilization = len(self.memory) / self.capacity * 100
        
        # Sample some experiences to get reward stats
        sample_size = min(1000, len(self.memory))
        sample = random.sample(self.memory, sample_size)
        rewards = [exp[2] for exp in sample]  # exp[2] is reward
        
        return {
            'size': len(self.memory),
            'capacity': self.capacity,
            'utilization': utilization,
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }


# Test the replay memory
if __name__ == "__main__":
    print("🧪 Testing Replay Memory\n")
    
    memory = ReplayMemory(capacity=1000)
    
    # Add some fake experiences
    print("Adding experiences...")
    for i in range(50):
        state = np.random.randint(0, 16, size=(4, 4))
        action = np.random.randint(0, 4)
        reward = np.random.randint(-10, 100)
        next_state = np.random.randint(0, 16, size=(4, 4))
        done = np.random.random() < 0.1
        
        memory.push(state, action, reward, next_state, done)
    
    print(f"Memory size: {len(memory)}")
    print(f"Ready to train? {memory.is_ready(batch_size=32)}")
    
    # Sample a batch
    print("\nSampling batch of 10...")
    states, actions, rewards, next_states, dones = memory.sample(10)
    
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Sample rewards: {rewards[:5]}")
    
    # Get stats
    print("\n📊 Memory stats:")
    stats = memory.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Replay memory working correctly!")