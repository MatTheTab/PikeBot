from utils.chess_utils import *
from utils.pikeBot_chess_utils import *

class PikeBotModelWrapper500k(PikeBotModelWrapper):
    """
    A wrapper to implement the same methods of board encoding and evaluation as the simple models from the chess_utils
    utilizing the NN model.

    Class designed specifically for 500k small model, should be compatible with the regular PikeBot class.

    Attributes:
    - model (torch.nn.Module): NN model for the evaluation of the board state
    - attributes_to_standardize (dict): list of attribute names that need to be standardized along with the corresponding mean and standard deviation
    - attributes_to_scale (dict): list of attribute names that need to be scaled along with the corresponding min and max value
    - attributes_order (list): the order in which the attributes should be fed to the NN

    Attributes:
    - model (torch.nn.Module): NN model for the evaluation of the board state
    - attributes_to_standardize (dict): list of attribute names that need to be standardized along with the corresponding mean and standard deviation
    - attributes_to_scale (dict): list of attribute names that need to be scaled along with the corresponding min and max value
    - attributes_order (list): the order in which the attributes should be fed to the NN

    Methods:
    - __init__(self, model_path, preprocessing_parameters_path): Initializes the PikeBotModelWrapper object.
    - encode(self, board : chess.Board, attributes : dict) preprocess the board and the additional parameters to a form that can be fed to a NN
    - predict(self, board_state) based on the preprocessed values returns an NN model prediction about the state.

    """

    def encode(
            self,
            move_history: List[chess.Board],
            evaluation_history: List[int],
            attributes : dict,
            fen=False):
        '''
        Preprocess the board and the additional parameters to a form that can be fed to a NN.

        Parameters:
        - move_history: List[chess.Board] List of board states representing the sequence of previous moves.
        - evaluation_history: List of evaluations of stockfish engine for each of the moves in the move history.
        - attributes (dict) Additional information about the game and the opponent.

        Returns:
        - Preprocessed parameters in the form that can be fed to the NN model.

        '''
        boards = move_history[-6:]
        bitboards = [
            self.get_intupt_bitboard(
                board,
                board.fen if fen else ''
            )
            for board in boards
        ]

        #pad bitboards if history to short
        if len(bitboards) < 6:
            bitboards = [torch.zeros_like(bitboards[0])] * (6-len(bitboards)) + bitboards

        attributes_encoded = self.encode_attributes(attributes)
        bitboards = np.stack(bitboards, axis=1)

        return (bitboards, attributes_encoded)

    # def predict_batch(self, encoded_states: list):
    #     '''
    #     Predict the probabilities for a batch of positions to be achieved by a human.

    #     Parameters:
    #     - encoded_states: list of states encoded using an encode method constituting a batch

    #     Returns:
    #     - List of floats: predicted probabilities of human moves for each position in the batch.
    #     '''

    #     raise NotImplementedError
    
class PikeBot500k(Pikebot):
    def get_additional_attributes(self):
        my_score = self.evaluation_history[-2]
        opponent_score = self.evaluation_history[-1]
        return {
            'elo': self.opponents_elo,
            'color': self.is_white,
            'clock': np.random.randint(100, 900),
            'stockfish_score_depth_8': opponent_score,
            'stockfish_difference_depth_8': my_score + opponent_score,
            'bullet': 0,
            'rapid': 0,
            'classic': 1,
            'blitz': 0,
            'rated': 1
        }
    

