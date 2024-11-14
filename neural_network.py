import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import chess
import sys
from pathlib import Path

class ChessNN:
    def __init__(self):
        self.model = self._build_model()
        self.load_training_games()
        
    def print_progress_bar(self, current, total, prefix='', suffix='', length=50, fill='#'):
        percent = float(current) * 100 / total
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}% {suffix}')
        sys.stdout.flush()
        if current == total:
            print()
            
    def load_training_games(self):
        games_dir = Path("games")
        if not games_dir.exists():
            print("No training games found.")
            return
            
        game_files = list(games_dir.glob("game_training_*.txt"))
        if not game_files:
            print("No training games found.")
            return
            
        print("\nLoading training games:")
        X = []
        y = []
        
        for i, game_file in enumerate(game_files):
            self.print_progress_bar(i, len(game_files), 
                                  prefix='Loading:', 
                                  suffix=f'({i}/{len(game_files)} games)', 
                                  length=40)
            
            with open(game_file) as f:
                game = chess.pgn.read_game(f)
                result = game.headers["Result"]
                
                # Convert result to target value
                if result == "1-0":
                    target = 1.0
                elif result == "0-1":
                    target = -1.0
                else:
                    target = 0.0
                
                # Process all positions in the game
                board = game.board()
                for move in game.mainline_moves():
                    X.append(self._board_to_input(board))
                    y.append(target)
                    board.push(move)
        
        # Final progress bar
        self.print_progress_bar(len(game_files), len(game_files), 
                              prefix='Loading:', 
                              suffix=f'({len(game_files)}/{len(game_files)} games)', 
                              length=40)
        
        if X and y:
            print("\nTraining on loaded games...")
            X = np.array(X)
            y = np.array(y)
            
            # Add validation split and early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            self.model.fit(
                X, y,
                epochs=20,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            print("\nInitial training complete!")
        
    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(768,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='tanh')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def predict(self, board_state):
        # Ensure prediction is thread-safe and returns a Python float
        prediction = self.model.predict(np.array([board_state]), verbose=0)[0][0]
        return float(prediction)
    
    def train_on_game(self, moves, result):
        if result == "1-0":
            target = 1.0
        elif result == "0-1":
            target = -1.0
        else:
            target = 0.0
            
        # Create training data from game moves
        X = []
        y = []
        
        board = chess.Board()
        for move in moves:
            board_state = self._board_to_input(board)
            X.append(board_state)
            y.append(target)
            board.push(move)
            
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y, epochs=5, verbose=0)
    
    def _board_to_input(self, board):
        input_data = np.zeros((8, 8, 12))
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                input_data[i // 8][i % 8][piece_idx] = 1
        return input_data.flatten() 