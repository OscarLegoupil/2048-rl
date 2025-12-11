"""
Deep Q-Network Agent for 2048.

Implements:
- Double DQN target computation
- Replay memory
- Epsilon-greedy action selection with invalid move masking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path so imports work when called from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.neural_net import DQN, preprocess_state
from agent.replay_memory import ReplayMemory
from training.config import config


class DQNAgent:
    """
    DQN Agent that learns to play 2048 using Double DQN.
    
    Networks:
    - policy_net: updated every batch
    - target_net: updated every N episodes (more stable targets)
    """

    def __init__(self, cfg=config):
        """
        Initialize the DQN agent.

        Args:
            cfg: Configuration object with hyperparameters
        """
        self.config = cfg

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        # === Networks ===
        self.policy_net = DQN(
            board_size=cfg.BOARD_SIZE,
            action_space=cfg.ACTION_SPACE,
        ).to(self.device)

        self.target_net = DQN(
            board_size=cfg.BOARD_SIZE,
            action_space=cfg.ACTION_SPACE,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # === Optimizer & loss ===
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=cfg.LEARNING_RATE,
        )
        self.criterion = nn.MSELoss()

        # === Replay Memory ===
        self.memory = ReplayMemory(capacity=cfg.REPLAY_MEMORY_SIZE)

        # === Exploration parameters ===
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_MIN
        self.epsilon_decay = cfg.EPSILON_DECAY

        # === Training stats ===
        self.episodes_trained = 0
        self.total_steps = 0
        self.losses = []

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def select_action(self, state, valid_actions=None):
        """
        Select action using epsilon-greedy policy with invalid-move masking.

        Args:
            state: Raw board state (4,4) numpy array (tile values)
            valid_actions: list of valid action indices, or None

        Returns:
            int: chosen action in {0,1,2,3}
        """
        # Fallback: if env doesn't give valid_actions, assume all
        if not valid_actions:
            valid_actions = list(range(self.config.ACTION_SPACE))

        # Exploration
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Exploitation
        processed_state = preprocess_state(state)
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            )  # (1,4,4)
            q_values = self.policy_net(state_tensor)[0].cpu().numpy()

        # Mask invalid actions
        mask = np.ones_like(q_values, dtype=np.float32) * -1e9
        mask[valid_actions] = 0.0
        q_values = q_values + mask

        return int(np.argmax(q_values))

    # -------------------------------------------------------------------------
    # Replay memory interaction
    # -------------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        States are preprocessed (log2 encoding) before storage to keep
        representation consistent between training and inference.
        """
        processed_state = preprocess_state(state)
        processed_next_state = preprocess_state(next_state)

        self.memory.push(
            processed_state,
            action,
            reward,
            processed_next_state,
            done,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train_step(self):
        """
        Perform one training step (one minibatch update).

        Uses Double DQN target:
            a* = argmax_a Q_policy(next_state, a)
            y  = r + γ * Q_target(next_state, a*)
        """
        if not self.memory.is_ready(self.config.BATCH_SIZE):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.BATCH_SIZE
        )

        # To tensors
        states = torch.FloatTensor(states).to(self.device)       # (B,4,4)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)      # (B,)
        rewards = torch.FloatTensor(rewards).to(self.device)     # (B,)
        dones = torch.FloatTensor(dones).to(self.device)         # (B,)

        # Q(s,a) from policy_net
        q_values = self.policy_net(states)                       # (B,4)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN targets
        with torch.no_grad():
            # a* = argmax_a Q_policy(next_state, a)
            next_q_policy = self.policy_net(next_states)         # (B,4)
            next_actions = next_q_policy.argmax(dim=1)           # (B,)

            # Q_target(next_state, a*)
            next_q_target = self.target_net(next_states)         # (B,4)
            next_q = next_q_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)                                         # (B,)

            # y = r + γ * (1 - done) * next_q
            target_q = rewards + (1.0 - dones) * self.config.GAMMA * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.total_steps += 1

        return loss.item()

    def update_target_network(self):
        """Synchronize target_net with policy_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def episode_end(self):
        """
        Call at the end of each episode.

        - Decays epsilon
        - Periodically updates target network
        """
        self.episodes_trained += 1
        self.decay_epsilon()

        if self.episodes_trained % self.config.TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()
            print(f"🎯 Target network updated (episode {self.episodes_trained})")

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, filepath):
        """
        Save model weights and training state.

        Args:
            filepath: path for .pth checkpoint
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episodes_trained": self.episodes_trained,
                "total_steps": self.total_steps,
            },
            filepath,
        )
        print(f"💾 Model saved to {filepath}")

    def load(self, filepath):
        """
        Load model weights and training state.

        Args:
            filepath: path for .pth checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.episodes_trained = checkpoint.get("episodes_trained", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

        print(f"📂 Model loaded from {filepath}")

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def get_stats(self):
        """
        Return training statistics for logging.

        Returns:
            dict with keys:
                - episodes
                - total_steps
                - epsilon
                - avg_loss (last 100 steps)
                - memory_size
        """
        avg_loss = float(np.mean(self.losses[-100:])) if self.losses else 0.0

        return {
            "episodes": self.episodes_trained,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "avg_loss": avg_loss,
            "memory_size": len(self.memory),
        }


if __name__ == "__main__":
    # Simple smoke test
    agent = DQNAgent(config)
    fake_state = np.zeros((4, 4), dtype=np.float32)
    action = agent.select_action(fake_state, valid_actions=[0, 1, 2, 3])
    print("Test action:", action)
