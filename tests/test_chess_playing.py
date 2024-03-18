import pytest
import random

from utils.chess_utils import *

def get_random_chessboard(min_moves=20, max_moves=100):
    """
    Generate a chessboard at a position created after making between min_moves and max_moves random legal moves.

    Parameters:
    - min_moves (int, optional): minimum legal moves to be made.
    - max_moves (int, optional): maximum legal moves to be made.

    Returns:
    - chess.Board: a chessboard at a random position
    """
    assert min_moves < max_moves, "min_moves should be lower than max_moves"

    num_moves = random.randint(30, 80)
    board = chess.Board()
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        random_move = random.choice(legal_moves)
        board.push(random_move)
    
    return board


class TestChessPlaying:
    stockfish_path = r"..\engine\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    def setup_method(self, method):
        print(f"Setting up {method}")
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def teardown_method(self, method):
        print(f"Tearing down {method}")
        self.engine.quit()
        del self.engine

    def test_one(self):
        move_scores = evaluate_all_moves_simple_engine(self.board, self.engine)
        print("Move Scores:", move_scores)
        assert len(move_scores) > 0
