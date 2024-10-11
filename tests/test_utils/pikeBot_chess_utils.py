import chess
import yaml
import unittest
from utils.chess_utils import mean_aggr
from utils.pikeBot_chess_utils import Pikebot, PikeBotModelWrapper


class TestPikeBot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load configuration from YAML file
        with open('tests/test_utils/pikeBot-config.yaml') as config_file:
            config = yaml.safe_load(config_file)

        # Retrieve paths and parameters from the config
        cls.stockfish_path = config['stockfish_path']
        cls.model_path = config["model_path"]
        cls.preprocessing_parameters_path = config["preprocessing_parameters_path"]

        # Initialize the model and bot once for all tests
        cls.model = PikeBotModelWrapper(cls.model_path, cls.preprocessing_parameters_path)
        cls.chessBot = Pikebot(model=cls.model, 
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

    def test_bot_make_move(self):
        board = chess.Board() 
        move = self.chessBot.get_best_move(board)
        self.assertIsNotNone(move)

if __name__ == '__main__':
    unittest.main()