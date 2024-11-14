import gradio as gr
import chess
import chess.svg
import numpy as np
from pathlib import Path
import os
import pickle
from neural_network import ChessNN
from game_recorder import GameRecorder
from generate_training_games import generate_training_set
import time
import glob
import chess.pgn
from chess_openings import CHESS_OPENINGS

class ChessWebUI:
    def __init__(self):
        # Create both models and games directories if they don't exist
        Path("models").mkdir(exist_ok=True)
        Path("games").mkdir(exist_ok=True)
        
        self.board = chess.Board()
        
        # Check for any existing models
        models = glob.glob("models/*.pkl")
        if models:
            # Load the most recently modified model
            latest_model = max(models, key=os.path.getmtime)
            print(f"Loading existing model: {os.path.basename(latest_model)}...")
            with open(latest_model, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
        else:
            print("No existing models found. Creating new model...")
            self.model = ChessNN()
            # Save the new model
            with open('models/default.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print("New model created and saved as default.pkl")
        
        self.game_recorder = GameRecorder()
        self.selected_square = None
        self.game_moves = []
        self.free_mode = False
        
        self.viewer_board = chess.Board()
        self.current_game_moves = []
        self.current_move_index = -1
        
        self.best_move_highlight = None
    
    def save_model_as_default(self):
        """Save current model as default"""
        try:
            with open('models/default.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            return "Current model saved as default"
        except Exception as e:
            return f"Error saving default model: {str(e)}"
    
    def get_available_models(self):
        models = glob.glob("models/*.pkl")
        return [os.path.basename(m) for m in models] if models else ["default.pkl"]
    
    def load_model(self, model_name):
        try:
            model_path = os.path.join("models", model_name)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                return f"Successfully loaded model: {model_name}"
            return f"Model file not found: {model_name}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def get_available_games(self):
        games = glob.glob("games/*.txt")
        return [os.path.basename(g) for g in games]
    
    def train_on_selected_games(self, selected_games):
        if not selected_games:
            return "No games selected for training"
        
        try:
            # Ensure models directory exists
            Path("models").mkdir(exist_ok=True)
            
            total_games = len(selected_games)
            print(f"\nTraining on {total_games} selected games...")
            
            X = []
            y = []
            
            for i, game_file in enumerate(selected_games):
                game_path = os.path.join("games", game_file)
                with open(game_path) as f:
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
                        X.append(self.model._board_to_input(board))
                        y.append(target)
                        board.push(move)
            
            if X and y:
                X = np.array(X)
                y = np.array(y)
                self.model.model.fit(X, y, epochs=5, batch_size=32, verbose=1)
                
                # Create models directory and save
                Path("models").mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_path = f"models/model_{timestamp}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                
                return f"Training complete! Model saved as: model_{timestamp}.pkl"
            
            return "No valid training data found in selected games"
            
        except Exception as e:
            return f"Error during training: {str(e)}"
    
    def board_to_input(self):
        input_data = np.zeros((8, 8, 12))
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                input_data[i // 8][i % 8][piece_idx] = 1
        return input_data.flatten()
    
    def get_ai_move(self):
        """Get best move with evaluations"""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None, []
        
        move_scores = []
        for move in legal_moves:
            self.board.push(move)
            board_state = self.board_to_input()
            score = float(self.model.predict(board_state))
            move_scores.append((move, score))
            self.board.pop()
        
        evaluations = [f"{move}: {score:.3f}" for move, score in move_scores]
        best_move = max(move_scores, key=lambda x: x[1])[0]
        return best_move, evaluations
    
    def make_move(self, square_name):
        try:
            square = chess.parse_square(square_name.lower())
            
            # Handle right-click (if square_name is special value)
            if square_name == "__right_click__":
                self.selected_square = None
                self.best_move_highlight = None
                return self.get_board_html(), "Selection cleared"
            
            # Handle spacebar (if square_name is special value)
            if square_name == "__spacebar__":
                move, evaluations = self.get_ai_move()
                if move:
                    self.best_move_highlight = move
                    return self.get_board_html(), f"Best move: {move}\n" + "\n".join(evaluations)
                return self.get_board_html(), "No legal moves available"
            
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece:
                    if self.free_mode:
                        # In free mode, check if the piece has legal moves
                        # Temporarily set the turn to match the piece's color
                        original_turn = self.board.turn
                        self.board.turn = piece.color
                        
                        has_legal_moves = any(
                            move.from_square == square 
                            for move in self.board.legal_moves
                        )
                        
                        # Restore original turn
                        self.board.turn = original_turn
                        
                        if has_legal_moves:
                            self.selected_square = square
                            # Set turn to piece's color to get correct legal moves
                            self.board.turn = piece.color
                            legal_moves = [
                                move.uci() 
                                for move in self.board.legal_moves 
                                if move.from_square == square
                            ]
                            return self.get_board_html(), f"Selected {square_name}. Legal moves: {legal_moves}"
                        return self.get_board_html(), "Selected piece has no legal moves"
                    else:
                        # Normal mode - only allow selecting pieces of current turn
                        if piece.color == self.board.turn:
                            self.selected_square = square
                            legal_moves = [
                                move.uci() 
                                for move in self.board.legal_moves 
                                if move.from_square == square
                            ]
                            return self.get_board_html(), f"Selected {square_name}. Legal moves: {legal_moves}"
                        return self.get_board_html(), "It's not your turn"
                    
                return self.get_board_html(), "No piece at selected square"
            else:
                # When making the move
                original_turn = self.board.turn
                piece = self.board.piece_at(self.selected_square)
                if piece and self.free_mode:
                    # Set turn to match the piece's color for move validation
                    self.board.turn = piece.color
                
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.game_moves.append(move)
                    self.selected_square = None
                    
                    if not self.free_mode:
                        # AI moves only in normal mode
                        if not self.board.is_game_over():
                            ai_move, evaluations = self.get_ai_move()
                            if ai_move:
                                self.board.push(ai_move)
                                self.game_moves.append(ai_move)
                                return (
                                    self.get_board_html(),
                                    f"AI evaluations:\n" + "\n".join(evaluations) + f"\nAI plays: {ai_move}"
                                )
                
                    if self.board.is_game_over():
                        result = self.board.result()
                        self.game_recorder.save_game(self.game_moves, result)
                        if not self.free_mode: # Only train on normal mode games
                            self.model.train_on_game(self.game_moves, result)
                            # Ensure models directory exists before saving
                            Path("models").mkdir(exist_ok=True)
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            model_path = f"models/model_{timestamp}.pkl"
                            with open(model_path, 'wb') as f:
                                pickle.dump(self.model, f)
                        return self.get_board_html(), f"Game Over! Result: {result}"
                    
                    current_turn = "White" if self.board.turn else "Black"
                    return self.get_board_html(), f"Move made. {current_turn} to move"
                
                # Restore original turn if move was invalid
                if self.free_mode:
                    self.board.turn = original_turn
                
                self.selected_square = None
                return self.get_board_html(), "Invalid move"
                
        except ValueError:
            return self.get_board_html(), "Invalid square"
    
    def get_board_html(self):
        """Generate HTML for the chess board with move highlighting"""
        squares = chess.SquareSet()
        arrows = []  # Initialize empty list for arrows
        
        # Add selected square
        if self.selected_square is not None:
            squares.add(self.selected_square)
        
        # Add legal move highlights
        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    squares.add(move.to_square)
        
        # Add best move highlight and arrow
        if self.best_move_highlight:
            squares.add(self.best_move_highlight.from_square)
            squares.add(self.best_move_highlight.to_square)
            arrows.append(chess.svg.Arrow(
                self.best_move_highlight.from_square,
                self.best_move_highlight.to_square,
                color="#0000FF80"
            ))
        
        # Generate SVG with highlights
        svg_data = chess.svg.board(
            self.board,
            squares=squares,
            arrows=arrows,
            size=400
        )
        
        html = f"""
        <div style="width: 400px;">
            <div style="width: 400px; height: 400px; border: 2px solid #333; border-radius: 5px;">
                {svg_data}
            </div>
            <div style="margin-top: 10px; font-family: monospace;">
                Turn: {'White' if self.board.turn else 'Black'}
            </div>
        </div>
        """
        return html
    
    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []
        return self.get_board_html(), "Game reset"
    
    def generate_training_games(self, num_games, selected_openings=None, progress=gr.Progress()):
        """Generate training games with progress tracking"""
        try:
            num_games = int(num_games)
            if num_games < 1:
                return "Number of games must be at least 1"
            
            if not selected_openings:
                return "Please select at least one opening"
            
            total_openings = len(selected_openings)
            total_games = num_games * total_openings
            progress(0, desc="Starting game generation...")
            
            def progress_callback(completed_games, current_opening=None, games_in_opening=None):
                progress((completed_games / total_games), 
                        desc=f"Generated {completed_games}/{total_games} games")
            
            print(f"\nGenerating {num_games} games per opening...")
            
            # Create a filtered openings dictionary
            selected_openings_dict = {k: CHESS_OPENINGS[k] for k in selected_openings}
            total_generated = generate_training_set(num_games, progress_callback, selected_openings_dict)
            
            return f"Successfully generated {total_generated} games ({num_games} per opening)"
        except Exception as e:
            return f"Error generating games: {str(e)}"
    
    def calculate_best_move(self):
        """Calculate and display the best move for the current position"""
        if self.board.is_game_over():
            return "Game is over!"
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return "No legal moves available!"
        
        move_scores = []
        for move in legal_moves:
            self.board.push(move)
            board_state = self.board_to_input()
            score = float(self.model.predict(board_state))
            move_scores.append((move, score))
            self.board.pop()
        
        # Sort moves by score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format the top 5 moves with their evaluations
        current_player = "White" if self.board.turn else "Black"
        response = f"Top moves for {current_player}:\n"
        for move, score in move_scores[:5]:
            response += f"{move}: {score:.3f}\n"
        
        best_move = move_scores[0][0]
        response += f"\nRecommended move: {best_move}"
        
        return response
    
    def toggle_free_mode(self):
        """Toggle free mode on/off"""
        self.free_mode = not self.free_mode
        return f"Free mode {'enabled' if self.free_mode else 'disabled'}"
    
    def load_game_for_viewing(self, game_file):
        try:
            game_path = os.path.join("games", game_file)
            with open(game_path) as f:
                game = chess.pgn.read_game(f)
            
            self.viewer_board = game.board()
            self.current_game_moves = list(game.mainline_moves())
            self.current_move_index = -1
            
            return (
                self.get_viewer_board_html(),
                f"Loaded game: {game_file}",
                f"Move 0/{len(self.current_game_moves)}"
            )
        except Exception as e:
            return (
                self.get_viewer_board_html(),
                f"Error loading game: {str(e)}",
                "No game loaded"
            )

    def get_viewer_board_html(self):
        """Generate HTML for the viewer chess board"""
        svg_data = chess.svg.board(self.viewer_board, size=400)
        html = f"""
        <div style="width: 400px; height: 400px; border: 2px solid #333; border-radius: 5px;">
            {svg_data}
        </div>
        """
        return html

    def viewer_next_move(self):
        if self.current_move_index + 1 < len(self.current_game_moves):
            self.current_move_index += 1
            move = self.current_game_moves[self.current_move_index]
            self.viewer_board.push(move)
            return (
                self.get_viewer_board_html(),
                f"Move {self.current_move_index + 1}/{len(self.current_game_moves)}"
            )
        return (
            self.get_viewer_board_html(),
            f"End of game - Move {self.current_move_index + 1}/{len(self.current_game_moves)}"
        )

    def viewer_prev_move(self):
        if self.current_move_index >= 0:
            self.viewer_board.pop()
            self.current_move_index -= 1
            return (
                self.get_viewer_board_html(),
                f"Move {self.current_move_index + 1}/{len(self.current_game_moves)}"
            )
        return (
            self.get_viewer_board_html(),
            "Start of game - Move 0/0"
        )

    def viewer_reset(self):
        self.viewer_board = chess.Board()
        self.current_move_index = -1
        return (
            self.get_viewer_board_html(),
            "Game reset - Move 0/0"
        )

    def calculate_and_make_ai_move(self):
        """Calculate and make the best AI move"""
        if self.board.is_game_over():
            return self.get_board_html(), "Game is over!"
        
        ai_move, evaluations = self.get_ai_move()
        if ai_move:
            self.board.push(ai_move)
            self.game_moves.append(ai_move)
            return (
                self.get_board_html(),
                f"AI evaluations:\n" + "\n".join(evaluations) + f"\nAI plays: {ai_move}"
            )
        return self.get_board_html(), "No legal moves available!"

def create_ui():
    chess_ui = ChessWebUI()
    
    with gr.Blocks(title="Chess AI with Neural Network") as interface:
        gr.Markdown("# Chess AI with Neural Network")
        
        with gr.Tabs():
            with gr.Tab("Play"):
                with gr.Row():
                    with gr.Column():
                        board_display = gr.HTML(chess_ui.get_board_html())
                        move_input = gr.Textbox(label="Enter move (e.g., 'e2' to select, then 'e4' to move)")
                        status_display = gr.Textbox(label="Status", interactive=False, lines=5)
                    
                    with gr.Column():
                        gr.Markdown("### Controls")
                        with gr.Row():
                            make_move_btn = gr.Button("Make Move")
                            calculate_btn = gr.Button("Calculate Best Move", variant="primary")
                        
                        with gr.Row():
                            reset_btn = gr.Button("Reset Game")
                            free_mode_btn = gr.Button("Toggle Free Mode")
                        
                        gr.Markdown("### Model Selection")
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=chess_ui.get_available_models(),
                                label="Select Model",
                                value=chess_ui.get_available_models()[0]
                            )
                            refresh_models_btn = gr.Button("🔄", size="sm")
                        with gr.Row():
                            load_model_btn = gr.Button("Load Selected Model")
                            save_default_btn = gr.Button("Save as Default")
                        
                        gr.Markdown("""
                        ### Game Modes
                        - Normal Mode: Play against AI
                        - Free Mode: Play both sides
                        """)
            
            with gr.Tab("Train"):
                with gr.Column():
                    gr.Markdown("### Generate New Training Games")
                    with gr.Row():
                        num_games = gr.Slider(minimum=1, maximum=10, value=3, step=1, 
                                            label="Games per Opening")
                        generate_btn = gr.Button("Generate Training Games", variant="primary")
                    generation_status = gr.Textbox(label="Generation Status", interactive=False)
                    
                    # Add openings selection
                    gr.Markdown("### Select Openings")
                    with gr.Row():
                        openings_checklist = gr.CheckboxGroup(
                            choices=list(CHESS_OPENINGS.keys()),
                            label="Select Openings for Generation",
                            value=[] # Initially no openings selected
                        )
                        with gr.Column():
                            select_all_openings_btn = gr.Button("Select All")
                            deselect_all_openings_btn = gr.Button("Deselect All")
            
                    gr.Markdown("### Train on Existing Games")
                    with gr.Row():
                        games_checklist = gr.CheckboxGroup(
                            choices=chess_ui.get_available_games(),
                            label="Select Games for Training",
                            value=[] # Initially no games selected
                        )
                        with gr.Column():
                            select_all_btn = gr.Button("Select All")
                            deselect_all_btn = gr.Button("Deselect All")
                            refresh_games_btn = gr.Button("Refresh Games List")
                    
                    train_btn = gr.Button("Train on Selected Games", variant="primary")
                    training_status = gr.Textbox(label="Training Status", interactive=False)
            
            with gr.Tab("Game Viewer"):
                with gr.Row():
                    with gr.Column():
                        viewer_board_display = gr.HTML(chess_ui.get_viewer_board_html())
                        move_counter = gr.Textbox(
                            label="Current Move",
                            value="No game loaded",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Game Controls")
                        games_dropdown = gr.Dropdown(
                            choices=chess_ui.get_available_games(),
                            label="Select Game to View",
                            value=None
                        )
                        
                        with gr.Row():
                            prev_btn = gr.Button("← Previous Move")
                            next_btn = gr.Button("Next Move →")
                        
                        with gr.Row():
                            reset_viewer_btn = gr.Button("Reset to Start")
                            refresh_games_list_btn = gr.Button("Refresh Games List")
                        
                        viewer_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

        # Event handlers
        make_move_btn.click(
            chess_ui.calculate_and_make_ai_move,
            outputs=[board_display, status_display]
        )
        
        reset_btn.click(
            chess_ui.reset_game,
            outputs=[board_display, status_display]
        )
        
        generate_btn.click(
            chess_ui.generate_training_games,
            inputs=[num_games, openings_checklist],
            outputs=[generation_status],
            show_progress=True
        )
        
        move_input.submit(
            chess_ui.make_move,
            inputs=[move_input],
            outputs=[board_display, status_display]
        )
        
        load_model_btn.click(
            chess_ui.load_model,
            inputs=[model_dropdown],
            outputs=[status_display]
        )
        
        def select_all_games():
            return gr.update(value=chess_ui.get_available_games())
        
        def deselect_all_games():
            return gr.update(value=[])
        
        select_all_btn.click(
            select_all_games,
            outputs=[games_checklist]
        )
        
        deselect_all_btn.click(
            deselect_all_games,
            outputs=[games_checklist]
        )
        
        refresh_games_btn.click(
            lambda: gr.update(choices=chess_ui.get_available_games(), value=[]),
            outputs=[games_checklist]
        )
        
        train_btn.click(
            chess_ui.train_on_selected_games,
            inputs=[games_checklist],
            outputs=[training_status]
        )

        calculate_btn.click(
            lambda: chess_ui.calculate_best_move(),
            outputs=[status_display]
        )
        
        free_mode_btn.click(
            chess_ui.toggle_free_mode,
            outputs=[status_display]
        )
        
        # Add handler for save default button
        save_default_btn.click(
            chess_ui.save_model_as_default,
            outputs=[status_display]
        )
        
        games_dropdown.change(
            chess_ui.load_game_for_viewing,
            inputs=[games_dropdown],
            outputs=[viewer_board_display, viewer_status, move_counter]
        )

        next_btn.click(
            chess_ui.viewer_next_move,
            outputs=[viewer_board_display, move_counter]
        )

        prev_btn.click(
            chess_ui.viewer_prev_move,
            outputs=[viewer_board_display, move_counter]
        )

        reset_viewer_btn.click(
            chess_ui.viewer_reset,
            outputs=[viewer_board_display, move_counter]
        )

        refresh_games_list_btn.click(
            lambda: gr.update(choices=chess_ui.get_available_games()),
            outputs=[games_dropdown]
        )

        refresh_models_btn.click(
            lambda: gr.update(choices=chess_ui.get_available_models()),
            outputs=[model_dropdown]
        )

        select_all_openings_btn.click(
            lambda: gr.update(value=list(CHESS_OPENINGS.keys())),
            outputs=[openings_checklist]
        )
        
        deselect_all_openings_btn.click(
            lambda: gr.update(value=[]),
            outputs=[openings_checklist]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.queue()  # Enable queuing
    interface.launch(share=True)