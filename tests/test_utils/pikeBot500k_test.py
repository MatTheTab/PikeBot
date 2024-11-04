import chess
import yaml
import unittest
import torch
import numpy as np
from utils.chess_utils import mean_aggr
from utils.pikeBot500k import PikeBot500k, PikeBotModelWrapper500k


class TestPikeBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load configuration from YAML file
        with open('tests/test_utils/configs/pikeBot500k-config.yaml') as config_file:
            config = yaml.safe_load(config_file)

        # Retrieve paths and parameters from the config
        cls.stockfish_path = config['stockfish_path']
        cls.model_path = config["model_path"]
        cls.preprocessing_parameters_path = config["preprocessing_parameters_path"]

        # Initialize the model and bot once for all tests
        cls.model = PikeBotModelWrapper500k(cls.model_path, cls.preprocessing_parameters_path)
        cls.chessBot = PikeBot500k(model=cls.model, 
                               aggregate=mean_aggr,
                               stockfish_path=cls.stockfish_path,
                               color='white',
                               opponents_elo=1500)

    @classmethod
    def tearDownClass(cls):
        cls.chessBot.close()

    def test_bot_initialization(self):
        self.assertIsNotNone(self.chessBot)
        self.assertEqual(self.chessBot.color, chess.WHITE)
        self.assertEqual(self.chessBot.opponents_elo, 1500)

    def test_predict(self):
        move_history = list()
        board = chess.Board()
        for move_str in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6']:
            move_history.append(board.copy())
            board.push(chess.Move.from_uci(move_str))
        self.chessBot.move_history = move_history
        evaluation_history = [
            self.chessBot.engine.analyse(
                    board,
                    chess.engine.Limit(depth=self.chessBot.engine_depth)
                )['score'].pov(color=self.chessBot.color).score(mate_score=900)
                for board in move_history
            ]
        self.chessBot.evaluation_history = evaluation_history
        encoded_state = self.chessBot.model.encode(
            move_history,
            evaluation_history,
            self.chessBot.get_additional_attributes(),
            )
        self.assertIsNotNone(encoded_state)
        self.assertEqual((1, 6, 76, 8, 8), encoded_state[0].shape)   
        prediction = self.chessBot.model.predict(encoded_state)
        self.assertEqual(type(encoded_state[0]), np.ndarray)  
        self.assertEqual(type(encoded_state[1]), torch.Tensor)  
        prediction = self.chessBot.model.predict(encoded_state)
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 1)


    def test_bot_make_move(self):
        board = chess.Board() 
        move = self.chessBot.get_best_move(board)
        self.assertIsNotNone(move)

if __name__ == '__main__':
    unittest.main()