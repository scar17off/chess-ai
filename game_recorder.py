import chess
import chess.pgn
from datetime import datetime
from pathlib import Path

class GameRecorder:
    def __init__(self):
        self.games_dir = Path("games")
        self.games_dir.mkdir(exist_ok=True)
    
    def save_game(self, moves, result):
        game = chess.pgn.Game()
        
        # Set game metadata
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Result"] = result
        
        # Add moves to game
        node = game
        for move in moves:
            node = node.add_variation(move)
        
        # Save to file
        filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = self.games_dir / filename
        
        with open(filepath, "w") as f:
            f.write(str(game)) 