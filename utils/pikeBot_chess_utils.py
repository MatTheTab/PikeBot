from utils.model_utils import *
from utils.data_utils import *
from utils.chess_utils import * 
import chess
import json

class PikeBotModelWrapper:
    '''
    A wrapper to implement the same methods of board encoding and evaluation as the simple models from the chess_utils
    utilizing the NN model.

    Note That the model's methods involve additional hyperparameters such as elo or stockfish evaluation opposed to the simple models so they may not be
    backwards compatible with previous models at a current stage.

    Attributes:
    - model (torch.nn.Module): NN model for the evaluation of the board state
    - attributes_to_standardize (dict): list of attribute names that need to be standardized along with the corresponding mean and standard deviation
    - attributes_to_scale (dict): list of attribute names that need to be scaled along with the corresponding min and max value
    - attributes_order (list): the order in which the attributes should be fed to the NN

    Methods:
    - __init__(self, model_path, preprocessing_parameters_path): Initializes the PikeBotModelWrapper object.
    - encode(self, board : chess.Board, attributes : dict) preprocess the board and the additional parameters to a form that can be fed to a NN
    - predict(self, board_state) based on the preprocessed values returns an NN model prediction about the state

    '''
    def __init__(self, model_path: str, preprocessing_parameters_path : str):
        '''
        Initialize the PikeBotModelWrapper

        Parameters:
        - model_path (str): a path to .pth file containing a NN model
        - preprocessing_parameters_path (str): a path to a .json file containing the meta information about the preprocessing parameters of the attributes
        '''
        self.model = torch.load(model_path)
        self.model.eval()  

        with open(preprocessing_parameters_path, 'r') as f:
            parameters = json.load(f)

        self.attributes_to_standardize = parameters["attributes_to_standardize"]
        self.attributes_to_scale = parameters["attributes_to_scale"]
        self.attributes_order = parameters["attributes_order"]

    def encode(self, board : chess.Board, attributes : dict, fen=False):
        '''
        Preprocess the board and the additional parameters to a form that can be fed to a NN.

        Parameters:
        - board (chess.Board) The current state of the chess Board.
        - attributes (dict) Additional information about the game and the opponent.

        Returns:
        - Preprocessed parameters in the form that can be fed to the NN model.

        '''
        if fen:
            str_board = board.fen()
        else:
            str_board = ''
        bitboard = get_bitboards(str_board = str_board, board =  board, str_functions = [], board_functions = [board_to_bitboards_optimized,get_all_attacks])
        input_bitboard = torch.tensor(bitboard[np.newaxis, ...], dtype = torch.float32)


        for key, value in self.attributes_to_standardize.items():
            attributes[key] -= value[0]
            attributes[key] /= value[1]


        for key, value in self.attributes_to_scale.items():
            attributes[key] -= value[0]
            attributes[key] /= value[1]
            
        attributes_to_fit = np.array([attributes[key] for key in self.attributes_order])
        attributes_encoded = torch.tensor(attributes_to_fit[np.newaxis, ...], dtype = torch.float32)

        return (input_bitboard, attributes_encoded)

    def predict(self, board_state):
        '''
        Predict the probability that the given position will be achieved by a human.

        Parameters:
        - board_state: preprocessed current state of the chess board and additional parameters.
        
        Returns:
        - float: predicted probability of the human move
        '''
        bitboard, hanging_inputs = board_state
        return self.model(bitboard, hanging_inputs)
    def predict_batch(self, bitboards_list, hanging_inputs_list):
        '''
        Predict the probabilities for a batch of positions to be achieved by a human.

        Parameters:
        - bitboards_list: list of numpy arrays or tensors, each representing the board state 
        (e.g., (batch_size, channels, width, height)).
        - hanging_inputs_list: list of numpy arrays or tensors, each representing additional 
        features for the state.

        Returns:
        - List of floats: predicted probabilities of human moves for each position in the batch.
        '''
        # Batch the bitboards and hanging inputs using torch.stack
        bitboards = torch.cat(bitboards_list)  # Stack the bitboards into a single tensor
        hanging_inputs = torch.cat(hanging_inputs_list)  # Stack the hanging inputs into a single tensor

        # Perform batch prediction using the model
        return self.model(bitboards, hanging_inputs) 
class Pikebot(ChessBot):
    '''
    Extension of ChessBot class, a playing agent with a modified loop to account for the hyperparameters of the game and track the game state.
    Enables possible future extensions of the model.

    Attributes:
    - name (str): The name of the bot.
    - engine (chess.engine.SimpleEngine): The engine used for move analysis.
    - time_limit (float): The time limit for move analysis.
    - model: The model used for board evaluation.
    - aggregate (function): The function used for aggregating move predictions.
    - depth (int): The depth of search for the bot's moves.
    - engine_depth (int): The depth of search for the engine's analysis.
    - color (chess.Color): The color of the bot, either chess.WHITE or chess.BLACK.
    - move_history (list): The list of moves as a sequence of the positions throughout the game.
    - opponents_elo (int): elo ranking of the opponent
    - is_white (bool): true if the bot plays as white and the opponent plays as black, false otherwise
    '''
    def __init__(self, 
                 model, 
                 aggregate, 
                 stockfish_path:str, 
                 color:str="white", 
                 time_limit:float=0.001, 
                 engine_depth:int=8, 
                 name:str="PikeBot", 
                 opponents_elo:int=1500
                 ):
        '''
        Initializes the Pikebot object.

        Parameters:
        - model: The model used for board evaluation.
        - aggregate (function): The function used for aggregating move predictions.
        - stockfish_path (str): The path to the Stockfish executable.
        - color (str, optional): The color of the bot, either "white" or "black". Defaults to "white".
        - time_limit (float, optional): The time limit for move analysis. Defaults to 0.01.
        - engine_depth (int, optional): The depth of search for the engine's analysis. Defaults to 20.
        - name (str, optional): The name of the bot. Defaults to "ChessBot".
        - move_history (list): The list of moves as a sequence of the positions throughout the game.
        - opponents_elo (int): elo ranking of the opponent
        - is_white (bool): true if the bot plays as white and the opponent plays as black, false otherwise
        '''
        super().__init__(model, aggregate, stockfish_path, color, time_limit, engine_depth, name)
        self.move_history = list()
        self.evaluation_history = list()
        self.opponents_elo = opponents_elo
        self.is_white = 1 if color == 'white' else 0

    def save_to_history(self, board: chess.Board):
        '''
        Saves current board state and its evaluation to move the history

        Parameters:
        -board (chess.Board): State of the board to be saved in the history.
        '''

        board_copy1 = board.copy()
        self.move_history.append(board_copy1)
        opponent_score = self.get_board_score(board)
        self.evaluation_history.append(opponent_score)
        
    def get_best_move_batch(self, board):
        '''
        Gets the best move.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
      
        self.save_to_history(board)
        
        my_moves_scores = []
        prediction_vars = []
        my_moves = list(board.legal_moves)

        board_state_cache = {}
        batch_inputs = []

        for move in my_moves:
            board.push(move)
            my_score = self.get_board_score(board)
            my_moves_scores.append(my_score)

            board_state2 = self.model.encode(board, {
                'elo': self.opponents_elo,
                'color': self.is_white,
                'clock': np.random.randint(100, 900),
                'stockfish_score_depth_8': my_score,
                'stockfish_difference_depth_8': my_score - self.evaluation_history[-1],
                'bullet': 0,
                'rapid': 0,
                'classic': 1,
                'blitz': 0,
                'rated': 1
            })
            
            opponent_moves = list(board.legal_moves)
            board_state_cache[move] = board_state2

       
            for next_move in opponent_moves:
                board.push(next_move)
                opponent_score = self.get_board_score(board)
                
                board_state = self.model.encode(board, {
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
                })

                bitboards = np.stack((board_state2[0], board_state[0]), axis=1)
                batch_inputs.append((move, next_move, bitboards, board_state[1], opponent_score))
                board.pop()

            board.pop()

     
        all_bitboards = [torch.from_numpy(np.reshape(bitboards, (bitboards.shape[0], -1, bitboards.shape[3], bitboards.shape[4]))) for _, _, bitboards, _, _ in batch_inputs]
        all_meta_data = [meta_data for _, _, _, meta_data, _ in batch_inputs]


        all_predictions = self.model.predict_batch(all_bitboards, all_meta_data)

  
        for idx, (move, next_move, _, _, opponent_score) in enumerate(batch_inputs):
            choice_prob = all_predictions[idx]
            
            prediction_vars.append((move, next_move, choice_prob, opponent_score))

     
        best_move = self.aggregate(prediction_vars)

     
        board_copy2 = board.copy()
        board_copy2.push(best_move)
        self.save_to_history(board_copy2)

        return best_move
    
    def get_best_move(self, board):
        '''
        Gets the best move.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
        #save opponents move and its evaluation to the history
        self.save_to_history(board)

        prediction_vars = []
        my_moves_scores = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)
            my_score = self.get_board_score(board)
            my_moves_scores.append(my_score)
            board_state2 = self.model.encode(board, {
                    'elo' : self.opponents_elo,
                    'color' : self.is_white,
                    'clock':np.random.randint(100,900),
                    'stockfish_score_depth_8' : my_score,
                    'stockfish_difference_depth_8' : my_score - self.evaluation_history[-1],
                    'bullet':0,
                    'rapid':0,
                    'classic':1,
                    'blitz':0,
                    'rated':1
                    })
            
            opponent_moves = list(board.legal_moves)    
            for next_move in opponent_moves:
                board.push(next_move)
                score = self.get_board_score(board)

                board_state = self.model.encode(board, {
                    'elo' : self.opponents_elo,
                    'color' : self.is_white,
                    'clock':np.random.randint(100,900),
                    'stockfish_score_depth_8' : score,
                    'stockfish_difference_depth_8' : my_score + score,
                    'bullet':0,
                    'rapid':0,
                    'classic':1,
                    'blitz':0,
                    'rated':1
                    })

             
                bitboards = np.stack((board_state2[0], board_state[0]), axis=1)
             
                batch_dim, mul, channel_dim, width, height = bitboards.shape
                bitboards = bitboards.reshape(batch_dim, mul*channel_dim, width, height)
           
                i1=torch.from_numpy(bitboards)
                
                choice_prob = self.model.predict((i1,board_state[1]))

                prediction_vars.append(tuple([move, next_move, choice_prob, score]))
                board.pop()
            board.pop()
        best_move = self.aggregate(prediction_vars)

        #save own move and its evaluation to the history
        board_copy2 = board.copy()
        board_copy2.push(best_move)
        self.save_to_history(board_copy2)

        
        return best_move
