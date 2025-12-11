"""
Training Loop for 2048 DQN Agent.

Key concept: Episode-based training
- Episode = One complete game of 2048 (from start to game over)
- Each episode:
  1. Reset environment
  2. Loop until game over:
     - Agent selects action
     - Environment executes action
     - Agent receives reward
     - Store experience
     - Train on batch
  3. Update stats and save checkpoints

The agent learns by playing thousands of games!
"""

import numpy as np
import time
import os
from collections import deque
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board2048
from agent.dqn_agent import DQNAgent
from training.config import config


class Trainer:
    """
    Orchestrates training of the DQN agent.
    
    Responsibilities:
    - Play episodes
    - Compute rewards
    - Track statistics
    - Save checkpoints
    - Print progress
    """
    
    def __init__(self, config, agent=None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            agent: Pre-initialized agent (or None to create new)
        """
        self.config = config
        
        # Create agent if not provided
        self.agent = agent if agent else DQNAgent(config)
        
        # Create environment
        self.env = Board2048(size=config.BOARD_SIZE)
        
        # Create directories
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # === Training statistics ===
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_steps = []
        
        # Running averages (last 100 episodes)
        self.recent_rewards = deque(maxlen=100)
        self.recent_scores = deque(maxlen=100)
        self.recent_max_tiles = deque(maxlen=100)
        
        # Best performance tracking
        self.best_score = 0
        self.best_max_tile = 0
        
        # Tile achievement tracking
        self.tile_achievements = {
            128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0
        }
    
    def compute_reward(self, points, game_over, valid_move):
        reward = 0

        # merging reward (kept)
        reward += points * self.config.MERGE_REWARD_MULTIPLIER

        # invalid move → small penalty, do NOT apply move penalty
        if not valid_move:
            return -1

        # small per-move penalty
        reward += -0.1

        if game_over:
            reward += -20

        return reward
 
    def play_episode(self, render=False):
        """
        Play one complete episode (game).
        
        Args:
            render: Whether to print board state (for debugging)
        
        Returns:
            episode_stats: Dict with episode statistics
        """
        # Reset environment
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        while not self.env.game_over:
            # Get valid actions
            valid_actions = self.env.get_valid_actions()
            
            # Agent selects action
            action = self.agent.select_action(state, valid_actions)
            
            # Execute action
            valid_move, points = self.env.move(action)
            next_state = self.env.get_state()
            game_over = self.env.game_over
            
            # Compute reward
            reward = self.compute_reward(points, game_over, valid_move)
            
            # Store experience
            self.agent.store_transition(state, action, reward, next_state, game_over)
            
            # Train on batch
            loss = self.agent.train_step()
            
            # Update state and stats
            state = next_state
            total_reward += reward
            steps += 1
            
            # Optional rendering
            if render:
                print(f"\nStep {steps}")
                print(self.env)
                print(f"Action: {['Up', 'Right', 'Down', 'Left'][action]}")
                print(f"Reward: {reward:.2f}")
                time.sleep(0.1)
        
        # Episode finished
        self.agent.episode_end()
        
        # Collect stats
        episode_stats = {
            'reward': total_reward,
            'score': self.env.score,
            'max_tile': self.env.get_max_tile(),
            'steps': steps,
        }
        
        return episode_stats
    
    def train(self, num_episodes=None, verbose=True):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train (None = use config)
            verbose: Whether to print progress
        """
        if num_episodes is None:
            num_episodes = self.config.MAX_EPISODES
        
        print("🚀 Starting Training!")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Replay memory: {self.config.REPLAY_MEMORY_SIZE}")
        print(f"Epsilon: {self.agent.epsilon:.3f} → {self.config.EPSILON_MIN}")
        print("=" * 60)
        print()
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Play one episode
            stats = self.play_episode()
            
            # Update statistics
            self.episode_rewards.append(stats['reward'])
            self.episode_scores.append(stats['score'])
            self.episode_max_tiles.append(stats['max_tile'])
            self.episode_steps.append(stats['steps'])
            
            self.recent_rewards.append(stats['reward'])
            self.recent_scores.append(stats['score'])
            self.recent_max_tiles.append(stats['max_tile'])
            
            # Track best performance
            if stats['score'] > self.best_score:
                self.best_score = stats['score']
            if stats['max_tile'] > self.best_max_tile:
                self.best_max_tile = stats['max_tile']
            
            # Track tile achievements
            for tile in self.tile_achievements.keys():
                if stats['max_tile'] >= tile:
                    self.tile_achievements[tile] += 1
            
            # Print progress
            if verbose and episode % self.config.PRINT_FREQUENCY == 0:
                self._print_progress(episode, num_episodes, start_time)
            
            # Save checkpoint
            if episode % self.config.SAVE_FREQUENCY == 0:
                self._save_checkpoint(episode)
        
        print("\n🎉 Training Complete!")
        self._print_final_stats()
    
    def _print_progress(self, episode, total_episodes, start_time):
        """Print training progress."""
        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed
        eta = (total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
        
        avg_reward = np.mean(self.recent_rewards)
        avg_score = np.mean(self.recent_scores)
        avg_max_tile = np.mean(self.recent_max_tiles)
        
        agent_stats = self.agent.get_stats()
        
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{total_episodes} ({episode/total_episodes*100:.1f}%)")
        print(f"{'='*60}")
        print(f"⏱️  Time: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | Speed: {eps_per_sec:.2f} eps/s")
        print(f"🎯 Epsilon: {agent_stats['epsilon']:.4f}")
        print(f"📊 Avg Reward (100): {avg_reward:.2f}")
        print(f"🎮 Avg Score (100): {avg_score:.0f}")
        print(f"🏆 Avg Max Tile (100): {avg_max_tile:.0f}")
        print(f"⭐ Best Score: {self.best_score}")
        print(f"🎖️  Best Max Tile: {self.best_max_tile}")
        print(f"💾 Memory: {agent_stats['memory_size']}/{self.config.REPLAY_MEMORY_SIZE}")
        if agent_stats['avg_loss'] > 0:
            print(f"📉 Avg Loss: {agent_stats['avg_loss']:.4f}")
        
        # Tile achievements
        print(f"\n🎯 Tile Achievements:")
        for tile, count in sorted(self.tile_achievements.items()):
            percentage = count / episode * 100
            print(f"   {tile:4d}: {count:4d} ({percentage:5.1f}%)")
    
    def _print_final_stats(self):
        """Print final training statistics."""
        print(f"\n{'='*60}")
        print("📊 Final Statistics")
        print(f"{'='*60}")
        print(f"Total Episodes: {len(self.episode_scores)}")
        print(f"Best Score: {self.best_score}")
        print(f"Best Max Tile: {self.best_max_tile}")
        print(f"Avg Score (all): {np.mean(self.episode_scores):.0f}")
        print(f"Avg Score (last 100): {np.mean(self.recent_scores):.0f}")
        print(f"\n🎯 Final Tile Achievements:")
        for tile, count in sorted(self.tile_achievements.items()):
            percentage = count / len(self.episode_scores) * 100
            print(f"   {tile:4d}: {count:4d} ({percentage:5.1f}%)")
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint."""
        filepath = os.path.join(
            self.config.MODEL_DIR,
            f"model_episode_{episode}.pth"
        )
        self.agent.save(filepath)
        
        # Also save as "latest"
        latest_path = os.path.join(self.config.MODEL_DIR, "model_latest.pth")
        self.agent.save(latest_path)
    
    def evaluate(self, num_episodes=100, render=False):
        """
        Evaluate trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render games
        
        Returns:
            stats: Dictionary of evaluation statistics
        """
        print(f"\n🧪 Evaluating agent over {num_episodes} episodes...")
        
        # Save current epsilon and set to 0 (no exploration)
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        scores = []
        max_tiles = []
        
        for i in range(num_episodes):
            stats = self.play_episode(render=render)
            scores.append(stats['score'])
            max_tiles.append(stats['max_tile'])
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_episodes}")
        
        # Restore epsilon
        self.agent.epsilon = old_epsilon
        
        # Compute statistics
        eval_stats = {
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores),
            'avg_max_tile': np.mean(max_tiles),
            'max_tile_achieved': np.max(max_tiles),
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Avg Score: {eval_stats['avg_score']:.0f}")
        print(f"   Max Score: {eval_stats['max_score']:.0f}")
        print(f"   Avg Max Tile: {eval_stats['avg_max_tile']:.0f}")
        print(f"   Max Tile Achieved: {eval_stats['max_tile_achieved']:.0f}")
        
        # Count tile achievements
        tile_counts = {}
        for tile in [128, 256, 512, 1024, 2048, 4096]:
            count = sum(1 for t in max_tiles if t >= tile)
            percentage = count / num_episodes * 100
            tile_counts[tile] = count
            print(f"   Reached {tile:4d}: {count:3d} ({percentage:5.1f}%)")
        
        return eval_stats


# Quick test
if __name__ == "__main__":
    print("🏋️ Testing Trainer\n")
    
    # Reduce episodes for quick test
    config.MAX_EPISODES = 5
    config.PRINT_FREQUENCY = 1
    
    trainer = Trainer(config)
    
    print("Playing 5 test episodes...\n")
    trainer.train(num_episodes=5, verbose=True)
    
    print("\n✅ Trainer test complete!")