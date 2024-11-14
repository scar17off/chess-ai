import chess
import chess.pgn
from pathlib import Path
from datetime import datetime
from chess_engine import ChessEngine
from chess_openings import CHESS_OPENINGS
import json
import concurrent.futures

# Load configuration
with open('chess_config.json', 'r') as f:
    config = json.load(f)

# Constants
GAMES_TO_GENERATE = config['DEFAULT_NUM_GAMES']
TOTAL_MOVES_MIN = config['MIN_MOVES_PER_GAME']
TOTAL_MOVES_MAX = config['MAX_MOVES_PER_GAME']
NUM_WORKERS = config['NUM_WORKERS']
WHITE_STRENGTH = 1.0
BLACK_STRENGTH = 0.3

#region Move Generation
def get_engine_move(engine, board):
    """Get a move from the chess engine"""
    return engine.get_best_move(board, depth=2)

def generate_sample_game(white_engine, black_engine, opening_name):
    board = chess.Board()
    moves = []
    
    # Set opening for white engine
    white_engine.set_opening(opening_name)
    
    # Play the game
    while not board.is_game_over() and len(moves) < TOTAL_MOVES_MAX:
        engine = white_engine if board.turn else black_engine
        move = get_engine_move(engine, board)
        
        if not move:
            break
            
        board.push(move)
        moves.append(move)
        
        if len(moves) >= TOTAL_MOVES_MIN and board.turn:
            eval = white_engine.evaluate_position(board)
            if eval > 500:
                if board.is_game_over() and board.result() == "1-0":
                    return moves, "1-0"
    
    return moves, board.result() if board.is_game_over() else None

def generate_opening_games(opening_name, games_per_opening, progress_callback=None):
    """Generate games for a specific opening"""
    white_engine = ChessEngine(strength=WHITE_STRENGTH, use_openings=True)
    black_engine = ChessEngine(strength=BLACK_STRENGTH, use_openings=False)
    
    successful_games = 0
    total_attempts = 0
    max_total_attempts = games_per_opening * 10  # Global maximum attempts
    
    while successful_games < games_per_opening and total_attempts < max_total_attempts:
        # Reset engines periodically to avoid potential state issues
        if total_attempts % 5 == 0:
            white_engine = ChessEngine(strength=WHITE_STRENGTH, use_openings=True)
            black_engine = ChessEngine(strength=BLACK_STRENGTH, use_openings=False)
        
        moves, result = generate_sample_game(white_engine, black_engine, opening_name)
        total_attempts += 1
        
        if moves and result == "1-0":
            # Save game immediately
            num_moves = save_game(moves, result, successful_games + 1, opening_name)
            successful_games += 1
            if progress_callback:
                progress_callback(successful_games)
    
    return opening_name, successful_games
#endregion

#region Game Storage
def save_game(moves, result, game_number, opening_name):
    # Only save if we have moves and the result is a win for White
    if not moves or result != "1-0":
        return 0
    
    # Find the next available game number
    games_dir = Path("games")
    existing_games = list(games_dir.glob("game_training_*.txt"))
    if existing_games:
        # Extract numbers from existing filenames and find the highest
        numbers = [int(game.stem.split('_')[-1]) for game in existing_games]
        next_number = max(numbers) + 1
    else:
        next_number = 1
    
    game = chess.pgn.Game()
    
    # Set headers
    game.headers["Event"] = f"Training Game - {opening_name}"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(next_number)
    game.headers["White"] = "Engine Strong"
    game.headers["Black"] = "Engine Weak"
    game.headers["Result"] = result
    game.headers["Opening"] = opening_name
    
    # Add moves
    node = game
    for move in moves:
        node = node.add_variation(move)
    
    # Save to file using the next available number
    filename = f"game_training_{next_number:03d}.txt"
    filepath = Path("games") / filename
    with open(filepath, "w") as f:
        f.write(str(game))
    
    return len(moves)
#endregion

#region Main Generation Logic
def generate_training_set(num_games=GAMES_TO_GENERATE, progress_callback=None, openings=None):
    Path("games").mkdir(exist_ok=True)
    
    openings_to_use = openings if openings is not None else CHESS_OPENINGS
    games_per_opening = num_games
    total_successful_games = 0
    
    print(f"\nGenerating {games_per_opening} games per opening using {NUM_WORKERS} workers:")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for opening_name in openings_to_use:
            def make_callback(opening):
                return lambda games_count: progress_callback(
                    total_successful_games + games_count
                ) if progress_callback else None
            
            future = executor.submit(
                generate_opening_games, 
                opening_name, 
                games_per_opening,
                make_callback(opening_name)
            )
            futures.append((future, opening_name))
        
        for future, opening_name in futures:
            try:
                opening_name, games_count = future.result()
                total_successful_games += games_count
                
                if games_count < games_per_opening:
                    print(f"  Warning: Could only generate {games_count} games for {opening_name}")
                    
            except Exception as e:
                print(f"  Error generating games for {opening_name}: {str(e)}")
    
    return total_successful_games
#endregion

#region Script Entry Point
if __name__ == "__main__":
    print("=" * 50)
    print("Chess Training Games Generator")
    print("=" * 50)
    
    num_games = input(f"Number of games to generate (default {GAMES_TO_GENERATE}): ")
    if num_games.isdigit():
        GAMES_TO_GENERATE = int(num_games)
    
    num_workers = input(f"Number of workers to use (default {NUM_WORKERS}): ")
    if num_workers.isdigit():
        NUM_WORKERS = int(num_workers)
    
    print("\nConfiguration:")
    print(f"Number of games: {GAMES_TO_GENERATE}")
    print(f"Games per opening: {GAMES_TO_GENERATE // len(CHESS_OPENINGS)}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Min moves per game: {TOTAL_MOVES_MIN}")
    print(f"Max moves per game: {TOTAL_MOVES_MAX}")
    
    input("\nPress Enter to start generation...")
    generate_training_set(GAMES_TO_GENERATE)
#endregion
