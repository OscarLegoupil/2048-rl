import numpy as np
import random
from typing import Tuple, Optional


class Board2048:
    """
    Core 2048 game logic.
    
    State representation:
    - 4x4 numpy array
    - 0 = empty cell
    - Powers of 2 for tiles (2, 4, 8, 16, ...)
    
    Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    """
    
    def __init__(self, size: int = 4):
        """Initialize empty board."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.game_over = False
        
        # Spawn 2 initial tiles
        self._add_random_tile()
        self._add_random_tile()
    
    def reset(self) -> np.ndarray:
        """Reset board to initial state."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        
        self._add_random_tile()
        self._add_random_tile()
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Return current board state as numpy array."""
        return self.board.copy()
    
    def _add_random_tile(self) -> bool:
        """
        Add a random tile (2 or 4) to an empty cell.
        Returns True if tile was added, False if board is full.
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        
        if not empty_cells:
            return False
        
        # 90% chance of 2, 10% chance of 4 (official 2048 rules)
        value = 2 if random.random() < 0.9 else 4
        row, col = random.choice(empty_cells)
        self.board[row, col] = value
        
        return True
    
    def move(self, action: int) -> Tuple[bool, int]:
        """
        Perform a move action.
        
        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left
        
        Returns:
            (move_valid, points_earned)
            - move_valid: True if board changed
            - points_earned: Points from merges in this move
        """
        if self.game_over:
            return False, 0
        
        board_before = self.board.copy()
        points_earned = 0
        
        # Map action to rotation: rotate so desired direction becomes "left"
        # np.rot90 with k=1 rotates counter-clockwise, k=-1 rotates clockwise
        # Action 0 (Up): rotate counter-clockwise (k=1) → up becomes left
        # Action 1 (Right): rotate 180° (k=2) → right becomes left  
        # Action 2 (Down): rotate clockwise (k=-1) → down becomes left
        # Action 3 (Left): no rotation (k=0) → left stays left
        rotation_map = {
            0: 1,   # Up: counter-clockwise
            1: 2,   # Right: 180°
            2: -1,  # Down: clockwise
            3: 0    # Left: no rotation
        }
        
        rotations = rotation_map[action]
        self.board = np.rot90(self.board, k=rotations)
        
        # Perform left move with merging
        points_earned = self._move_left()
        
        # Rotate back
        self.board = np.rot90(self.board, k=-rotations)
        
        # Check if board actually changed
        move_valid = not np.array_equal(board_before, self.board)
        
        if move_valid:
            # Add new random tile
            self._add_random_tile()
            self.score += points_earned
            
            # Check game over
            if self._is_game_over():
                self.game_over = True
        
        return move_valid, points_earned
    
    def _move_left(self) -> int:
        """
        Move and merge tiles to the left.
        Returns points earned from merges.
        """
        points = 0
        
        for i in range(self.size):
            # Extract non-zero values from row
            row = self.board[i, :]
            non_zero = row[row != 0]
            
            # Merge adjacent equal tiles
            merged = []
            skip = False
            
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                
                # Check if can merge with next tile
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged_value = non_zero[j] * 2
                    merged.append(merged_value)
                    points += merged_value
                    skip = True
                else:
                    merged.append(non_zero[j])
            
            # Pad with zeros
            merged = merged + [0] * (self.size - len(merged))
            self.board[i, :] = merged
        
        return points
    
    def _is_game_over(self) -> bool:
        """
        Check if game is over (no valid moves remaining).
        Game over when board is full AND no adjacent tiles can merge.
        """
        # Check if any empty cells
        if np.any(self.board == 0):
            return False
        
        # Check for possible merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
        
        # Check for possible merges vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def get_valid_actions(self) -> list:
        """
        Return list of valid actions (moves that change the board).
        Used for masking invalid actions in RL.
        """
        valid = []
        
        rotation_map = {0: 1, 1: 2, 2: -1, 3: 0}
        
        for action in range(4):
            board_copy = self.board.copy()
            
            # Simulate move
            rotations = rotation_map[action]
            board_rotated = np.rot90(board_copy, k=rotations)
            
            # Try left move
            board_after = board_rotated.copy()
            for i in range(self.size):
                row = board_after[i, :]
                non_zero = row[row != 0]
                
                merged = []
                skip = False
                for j in range(len(non_zero)):
                    if skip:
                        skip = False
                        continue
                    if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                        merged.append(non_zero[j] * 2)
                        skip = True
                    else:
                        merged.append(non_zero[j])
                
                merged = merged + [0] * (self.size - len(merged))
                board_after[i, :] = merged
            
            board_rotated_back = np.rot90(board_after, k=-rotations)
            
            # Check if board changed
            if not np.array_equal(board_copy, board_rotated_back):
                valid.append(action)
        
        return valid
    
    def get_max_tile(self) -> int:
        """Return the maximum tile value on the board."""
        return int(np.max(self.board))
    
    def get_empty_cells(self) -> int:
        """Return count of empty cells."""
        return int(np.sum(self.board == 0))
    
    def __str__(self) -> str:
        """Pretty print the board."""
        lines = ["-" * 25]
        for row in self.board:
            line = "|"
            for val in row:
                if val == 0:
                    line += "    ·"
                else:
                    line += f"{val:5}"
            line += " |"
            lines.append(line)
        lines.append("-" * 25)
        lines.append(f"Score: {self.score} | Max: {self.get_max_tile()}")
        return "\n".join(lines)


# Test the game
if __name__ == "__main__":
    print("🎮 Testing 2048 Game Engine\n")
    
    game = Board2048()
    print("Initial board:")
    print(game)
    print()
    
    # Play a few random moves
    actions = ["Up", "Right", "Down", "Left"]
    
    for i in range(10):
        action = random.randint(0, 3)
        valid, points = game.move(action)
        
        if valid:
            print(f"Move {i+1}: {actions[action]} → +{points} points")
            print(game)
            print()
        
        if game.game_over:
            print("💀 Game Over!")
            break
    
    print(f"\n📊 Final Stats:")
    print(f"   Score: {game.score}")
    print(f"   Max tile: {game.get_max_tile()}")
    print(f"   Empty cells: {game.get_empty_cells()}")