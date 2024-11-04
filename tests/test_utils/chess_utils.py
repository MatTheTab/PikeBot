import chess
import unittest
import yaml
from typing import Tuple
from unittest.mock import patch
from utils.chess_utils import ChessBot, mean_aggr, max_aggr, Uniform_model

class Test_mean_aggr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load configuration from YAML file
        with open('tests/test_utils/pikeBot-config.yaml') as config_file:
            config = yaml.safe_load(config_file)

        # Retrieve paths and parameters from the config
        cls.stockfish_path = config['stockfish_path']

    def mock_get_board_score(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """
        Mock function to return different values based on the board state.
        """
        if 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR' in board.fen():
            return 1
        if 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR' in board.fen():
            return 1
        return 0
    
    @patch.object(ChessBot, 'get_board_score')
    def test_induce_own_move(self, mock_get_board_score):
        """
        Unit test for ChessBot's induce_own_move method, mocking get_board_score with side_effect.
        """
        mock_get_board_score.side_effect = self.mock_get_board_score

        try:
            chessBot = ChessBot(
                model=Uniform_model(None),
                aggregate=mean_aggr,
                stockfish_path=self.stockfish_path,
            )

            board = chess.Board()
            move, score = chessBot.get_best_move(board)

            self.assertAlmostEqual(score, 0.1)
            self.assertEqual(move.uci(), 'e2e4')
        # Clean up
        finally:
            chessBot.close()

class Test_max_aggr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load configuration from YAML file
        with open('tests/test_utils/pikeBot-config.yaml') as config_file:
            config = yaml.safe_load(config_file)

        # Retrieve paths and parameters from the config
        cls.stockfish_path = config['stockfish_path']

    def mock_get_board_score(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """
        Mock function to return different values based on the board state.
        """
        if 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR' in board.fen():
            return 1
        if 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR' in board.fen():
            return 1
        if 'rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR' in board.fen():
            return 2
        
        return 0
    
    @patch.object(ChessBot, 'get_board_score')
    def test_induce_own_move(self, mock_get_board_score):
        """
        Unit test for ChessBot's induce_own_move method, mocking get_board_score with side_effect.
        """
        mock_get_board_score.side_effect = self.mock_get_board_score

        try:
            chessBot = ChessBot(
                model=Uniform_model(None),
                aggregate=max_aggr,
                stockfish_path=self.stockfish_path,
            )

            board = chess.Board()
            move, score = chessBot.get_best_move(board)


            self.assertAlmostEqual(score, 2)
            self.assertEqual(move.uci(), 'd2d4')
            # Clean up
        finally:
            chessBot.close()

if __name__ == "__main__":
    unittest.main()


