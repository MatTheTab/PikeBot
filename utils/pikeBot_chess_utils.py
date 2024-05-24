from utils.model_utils import *
from utils.data_utils import *
from utils.chess_utils import * 
import chess
import json

class PikeBotModelWrapper:
    def __init__(self, model_path, preprocessing_parameters_path):
        self.model = torch.load(model_path)
        self.model.eval()  

        with open(preprocessing_parameters_path, 'r') as f:
            parameters = json.load(f)

        self.attributes_to_standardize = parameters["attributes_to_standardize"]
        self.attributes_to_scale = parameters["attributes_to_scale"]
        self.attributes_order = parameters["attributes_order"]

    def encode(self, board : chess.Board, attributes : dict):
        str_board = board.fen()
        bitboard = get_bitboards(str_board = str_board, board =  board, str_functions = [str_to_board_all_figures_colors], board_functions = [get_all_attacks])
        input_bitboard = torch.tensor(bitboard[np.newaxis, ...], dtype = torch.float32)


        for key in self.attributes_to_standardize.keys():
            attributes[key] -= self.attributes_to_standardize[key][0]
            attributes[key] /= self.attributes_to_standardize[key][1]


        for key in self.attributes_to_scale.keys():
            attributes[key] -= self.attributes_to_scale[key][0]
            attributes[key] /= self.attributes_to_scale[key][1]
            
        attributes_to_fit = np.array([attributes[key] for key in self.attributes_order])
        attributes_encoded = torch.tensor(attributes_to_fit[np.newaxis, ...], dtype = torch.float32)
        return (input_bitboard, attributes_encoded)

    def predict(self, board_state):
        bitboard, hanging_inputs = board_state
        return self.model(bitboard, hanging_inputs)

class Pikebot(ChessBot):
    def __init__(self, model, aggregate, stockfish_path, color="white", time_limit=0.001, engine_depth=8, name="PikeBot", opponents_elo=1500):
        super().__init__(model, aggregate, stockfish_path, color, time_limit, engine_depth, name)
        self.move_history = list()
        self.evaluation_history = list()
        self.opponents_elo = opponents_elo
        self.is_white = 1 if color == 'white' else 0
        
    def get_best_move(self, board):
        '''
        Gets the best move.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
        #save opponents move and its evaluation to the history
        board_copy1 = board.copy()
        self.move_history.append(board_copy1)
        opponent_info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth, time=self.time_limit))
        self.evaluation_history.append(opponent_info['score'])

        prediction_vars = []
        my_moves_scores = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)
            my_info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth, time=self.time_limit))
            my_score = my_info['score'].pov(color=self.color).score(mate_score=900)
            my_moves_scores.append(my_score)

            opponent_moves = list(board.legal_moves)
            for next_move in opponent_moves:
                board.push(next_move)
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth, time=self.time_limit))
                score = info['score'].pov(color=self.color).score(mate_score=900)

                board_state = self.model.encode(board, {
                    'elo' : self.opponents_elo,
                    'color' : self.is_white,
                    'stockfish_score_depth_8' : score,
                    'stockfish_difference_depth_8' : score - my_score
                    })
                choice_prob = self.model.predict(board_state)

                prediction_vars.append(tuple([move, next_move, choice_prob, score]))
                board.pop()
            board.pop()
        best_move = self.aggregate(prediction_vars)

        #save own move to the history
        board_copy2 = board.copy()
        board_copy2.push(best_move)
        self.move_history.append(board_copy2)

        #save own move evaluation to the history 
        index = my_moves.index(best_move)
        self.evaluation_history.append(my_moves_scores[index])
        
        return best_move

    def close(self):
        self.engine.quit()
