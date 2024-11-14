import chess
import random
from chess_openings import CHESS_OPENINGS

class ChessEngine:
    def __init__(self, strength=1.0, use_openings=False):
        self.strength = strength
        self.use_openings = use_openings
        self.current_opening = None
        self.opening_move_index = 0
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def set_opening(self, opening_name):
        """Set the opening to be used"""
        self.current_opening = CHESS_OPENINGS[opening_name]
        self.opening_move_index = 0

    def get_opening_move(self, board):
        """Get the next move from the current opening if available"""
        if not self.current_opening or not self.use_openings:
            return None
            
        # Get the current position's move count
        move_count = len(board.move_stack)
        
        # Check if we're at the right position to make the next opening move
        if move_count < len(self.current_opening):
            next_move = self.current_opening[move_count]
            try:
                move = chess.Move.from_uci(next_move)
                if move in board.legal_moves:
                    return move
            except ValueError:
                pass
        
        return None

    def evaluate_position(self, board):
        """Evaluate the current position"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        score = 0
        
        # Material count
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        
        # Position evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Center control bonus
            center_bonus = 10 if square in [27, 28, 35, 36] else 0
            
            # Pawn structure
            if piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    score += (square // 8) * 10  # Advance white pawns
                else:
                    score -= (7 - square // 8) * 10  # Advance black pawns
            
            # Add position bonus
            if piece.color == chess.WHITE:
                score += center_bonus
            else:
                score -= center_bonus
        
        return score

    def get_best_move(self, board, depth=2):
        """Get the best move using opening book or minimax"""
        # First try to get a move from the opening book
        opening_move = self.get_opening_move(board)
        if opening_move:
            return opening_move
            
        # If no opening move available, use minimax (previous code remains the same)
        def minimax(board, depth, alpha, beta, maximizing):
            if depth == 0 or board.is_game_over():
                return self.evaluate_position(board)
            
            if maximizing:
                max_eval = float('-inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval

        best_move = None
        best_value = float('-inf') if board.turn else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        
        for move in legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, alpha, beta, not board.turn)
            board.pop()
            
            value += random.uniform(-200, 200) * (1 - self.strength)
            
            if board.turn:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move