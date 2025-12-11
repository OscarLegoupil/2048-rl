import pygame
import sys
import os

# Add parent directory to path so we can import game module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board2048


# Color scheme (modern 2048 colors)
COLORS = {
    'background': (187, 173, 160),
    'empty': (205, 193, 180),
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
}

TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    # All others use white text
}


class Game2048UI:
    """Interactive 2048 game with Pygame."""
    
    def __init__(self, size=4, cell_size=120):
        pygame.init()
        
        self.size = size
        self.cell_size = cell_size
        self.padding = 18
        self.header_height = 140
        
        # Calculate window size
        board_size = size * cell_size + (size + 1) * self.padding
        self.width = board_size
        self.height = board_size + self.header_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2048 - Use Arrow Keys")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 30)
        
        # Game
        self.game = Board2048(size=size)
        self.clock = pygame.time.Clock()
        
        # Animation
        self.animating = False
    
    def draw_cell(self, row, col, value):
        """Draw a single cell."""
        x = self.padding + col * (self.cell_size + self.padding)
        y = self.header_height + self.padding + row * (self.cell_size + self.padding)
        
        # Cell background
        color = COLORS.get(value, COLORS[0])
        pygame.draw.rect(
            self.screen, 
            color, 
            (x, y, self.cell_size, self.cell_size),
            border_radius=8
        )
        
        # Cell text
        if value != 0:
            text_color = TEXT_COLORS.get(value, (255, 255, 255))
            
            # Adjust font size based on number of digits
            if value >= 1000:
                font = self.font_small
            elif value >= 100:
                font = self.font_medium
            else:
                font = self.font_large
            
            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)
    
    def draw_board(self):
        """Draw the entire game board."""
        # Background
        self.screen.fill(COLORS['background'])
        
        # Header with score
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, (119, 110, 101))
        self.screen.blit(score_text, (20, 20))
        
        max_tile_text = self.font_small.render(f"Max: {self.game.get_max_tile()}", True, (119, 110, 101))
        self.screen.blit(max_tile_text, (20, 60))
        
        # Instructions
        instructions = self.font_small.render("Arrow Keys to Move | R to Restart | Q to Quit", True, (119, 110, 101))
        inst_rect = instructions.get_rect(right=self.width - 20, centery=40)
        self.screen.blit(instructions, inst_rect)
        
        # Draw cells
        for row in range(self.size):
            for col in range(self.size):
                value = self.game.board[row, col]
                self.draw_cell(row, col, value)
        
        # Game over overlay
        if self.game.game_over:
            overlay = pygame.Surface((self.width, self.height))
            overlay.set_alpha(200)
            overlay.fill((255, 255, 255))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, (119, 110, 101))
            text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 30))
            self.screen.blit(game_over_text, text_rect)
            
            score_text = self.font_medium.render(f"Final Score: {self.game.score}", True, (119, 110, 101))
            score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 + 30))
            self.screen.blit(score_text, score_rect)
            
            restart_text = self.font_small.render("Press R to restart", True, (119, 110, 101))
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 80))
            self.screen.blit(restart_text, restart_rect)
    
    def handle_input(self):
        """Handle keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                
                if event.key == pygame.K_r:
                    self.game.reset()
                    return True
                
                # Arrow keys for moves
                # Fixed mapping: actions in board.py are 0=Up, 1=Right, 2=Down, 3=Left
                action = None
                if event.key == pygame.K_UP:
                    action = 0  # Up
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Right
                elif event.key == pygame.K_DOWN:
                    action = 2  # Down
                elif event.key == pygame.K_LEFT:
                    action = 3  # Left
                
                if action is not None and not self.game.game_over:
                    valid, points = self.game.move(action)
                    if valid:
                        print(f"Move: {['Up', 'Right', 'Down', 'Left'][action]} | Points: +{points} | Score: {self.game.score}")
        
        return True
    
    def run(self):
        """Main game loop."""
        print("🎮 2048 Game Started!")
        print("Controls:")
        print("  ↑ ↓ ← → : Move tiles")
        print("  R : Restart game")
        print("  Q : Quit")
        print()
        
        running = True
        while running:
            running = self.handle_input()
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print(f"\n📊 Final Score: {self.game.score}")
        print(f"   Max Tile: {self.game.get_max_tile()}")


if __name__ == "__main__":
    game_ui = Game2048UI(size=4, cell_size=120)
    game_ui.run()