import pytest
from utils.chess_utils import *

class TestChessPlaying:
    stockfish_path = r"C:\Users\Bartek\Desktop\Studies\chessbot\stockfish\stockfish-windows-x86-64-avx2.exe"
    
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
