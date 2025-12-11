"""
Watch a trained agent play 2048.

Usage:
    python play.py                     # Load latest model, play with pygame
    python play.py --model path.pth    # Load specific model
    python play.py --games 10          # Play multiple games
    python play.py --no-render         # No pygame visualization
"""

import argparse
import os
import time
import numpy as np

from game.board import Board2048
from agent.dqn_agent import DQNAgent
from training.config import config

# Try to import pygame (optional)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Pygame not available, using terminal display only")


def play_game_terminal(agent, render_delay=0.3):
    """
    Play one game and render in terminal.
    
    Args:
        agent: Trained DQN agent
        render_delay: Delay between moves (seconds)
    
    Returns:
        stats: Game statistics
    """
    env = Board2048()
    state = env.reset()
    
    print("\n" + "="*60)
    print("🎮 New Game Starting!")
    print("="*60)
    print(env)
    
    step = 0
    
    while not env.game_over:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Agent selects action (no exploration)
        action = agent.select_action(state, valid_actions)
        
        # Execute action
        valid_move, points = env.move(action)
        
        if valid_move:
            step += 1
            action_names = ['↑ Up', '→ Right', '↓ Down', '← Left']
            print(f"\nStep {step}: {action_names[action]} (+{points} points)")
            print(env)
            time.sleep(render_delay)
        
        state = env.get_state()
    
    print("\n" + "="*60)
    print("💀 Game Over!")
    print("="*60)
    
    stats = {
        'score': env.score,
        'max_tile': env.get_max_tile(),
        'steps': step
    }
    
    return stats


def play_game_pygame(agent, cell_size=120):
    """
    Play one game with pygame visualization.
    
    Args:
        agent: Trained DQN agent
        cell_size: Size of each cell in pixels
    
    Returns:
        stats: Game statistics
    """
    if not PYGAME_AVAILABLE:
        print("⚠️  Pygame not available, falling back to terminal")
        return play_game_terminal(agent)
    
    # Import pygame visualization
    from visualization.pygame_view import Game2048UI, COLORS
    
    # Create custom game UI for agent
    pygame.init()
    
    size = 4
    padding = 18
    header_height = 140
    board_size = size * cell_size + (size + 1) * padding
    width = board_size
    height = board_size + header_height
    
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2048 - AI Agent Playing")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.Font(None, 60)
    font_medium = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 30)
    
    # Game
    env = Board2048()
    state = env.reset()
    
    step = 0
    running = True
    
    while running and not env.game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Agent selects action
        action = agent.select_action(state, valid_actions)
        
        # Execute action
        valid_move, points = env.move(action)
        
        if valid_move:
            step += 1
        
        state = env.get_state()
        
        # Render
        screen.fill(COLORS['background'])
        
        # Header
        score_text = font_medium.render(f"Score: {env.score}", True, (119, 110, 101))
        screen.blit(score_text, (20, 20))
        
        max_tile_text = font_small.render(f"Max: {env.get_max_tile()}", True, (119, 110, 101))
        screen.blit(max_tile_text, (20, 60))
        
        step_text = font_small.render(f"Steps: {step}", True, (119, 110, 101))
        screen.blit(step_text, (20, 95))
        
        ai_text = font_small.render("🤖 AI Playing (Q to Quit)", True, (119, 110, 101))
        ai_rect = ai_text.get_rect(right=width - 20, centery=40)
        screen.blit(ai_text, ai_rect)
        
        # Draw board
        for row in range(size):
            for col in range(size):
                x = padding + col * (cell_size + padding)
                y = header_height + padding + row * (cell_size + padding)
                
                value = env.board[row, col]
                color = COLORS.get(value, COLORS[0])
                
                pygame.draw.rect(
                    screen,
                    color,
                    (x, y, cell_size, cell_size),
                    border_radius=8
                )
                
                if value != 0:
                    text_color = (119, 110, 101) if value <= 4 else (255, 255, 255)
                    
                    if value >= 1000:
                        font = font_small
                    elif value >= 100:
                        font = font_medium
                    else:
                        font = font_large
                    
                    text = font.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                    screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(5)  # 5 moves per second
    
    # Game over
    if running:
        time.sleep(1)
    
    pygame.quit()
    
    stats = {
        'score': env.score,
        'max_tile': env.get_max_tile(),
        'steps': step
    }
    
    return stats


def main():
    """Main function to watch trained agent play."""
    
    parser = argparse.ArgumentParser(description='Watch trained 2048 agent play')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: latest)')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable pygame rendering (terminal only)')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between moves in terminal mode (seconds)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 2048 AI Agent - Watch Mode")
    print("="*60)
    
    # Load model
    agent = DQNAgent(config)
    agent.epsilon = 0.0  # No exploration
    
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(config.MODEL_DIR, "model_latest.pth")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Train a model first with: python train.py")
        return
    
    agent.load(model_path)
    print(f"✅ Model loaded: {model_path}")
    print(f"   Episodes trained: {agent.episodes_trained}")
    print()
    
    # Play games
    all_scores = []
    all_max_tiles = []
    
    for game_num in range(1, args.games + 1):
        print(f"\n{'='*60}")
        print(f"Game {game_num}/{args.games}")
        print(f"{'='*60}")
        
        if args.no_render or not PYGAME_AVAILABLE:
            stats = play_game_terminal(agent, render_delay=args.delay)
        else:
            stats = play_game_pygame(agent)
        
        all_scores.append(stats['score'])
        all_max_tiles.append(stats['max_tile'])
        
        print(f"\n📊 Game {game_num} Results:")
        print(f"   Score: {stats['score']}")
        print(f"   Max Tile: {stats['max_tile']}")
        print(f"   Steps: {stats['steps']}")
    
    # Summary
    if args.games > 1:
        print(f"\n{'='*60}")
        print(f"📊 Summary ({args.games} games)")
        print(f"{'='*60}")
        print(f"Avg Score: {np.mean(all_scores):.0f}")
        print(f"Max Score: {np.max(all_scores)}")
        print(f"Avg Max Tile: {np.mean(all_max_tiles):.0f}")
        print(f"Best Max Tile: {np.max(all_max_tiles)}")
        
        # Count tile achievements
        for tile in [128, 256, 512, 1024, 2048, 4096]:
            count = sum(1 for t in all_max_tiles if t >= tile)
            percentage = count / args.games * 100
            print(f"Reached {tile:4d}: {count:2d}/{args.games} ({percentage:.0f}%)")
    
    print(f"\n{'='*60}")
    print("✨ Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()