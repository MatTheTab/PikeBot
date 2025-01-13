import chess
import numpy as np
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot

class PikeBotHeuristic8BB(Pikebot):

    def __init__(self, model, aggregate, stockfish_path: str, color: str = "white", time_limit: float = 0.001, engine_depth: int = 8, name: str = "PikeBot", opponents_elo: int = 1500,
                 max_depth: int = 3, n_best_moves: int = 5):
        super().__init__(model, aggregate, stockfish_path, color, time_limit, engine_depth, name, opponents_elo)
        self.max_depth = max_depth
        self.n_best_moves = n_best_moves
        self.depth_check = 0

    def induce_own_move(
            self,
            board: chess.Board,
            depth: int = 0,
            **kwargs
        ):
        self.depth_check = max(depth, self.depth_check)

        alpha = kwargs.get('alpha', -float('inf'))
        beta = kwargs.get('beta', float('inf'))

        if board.legal_moves.count() == 0:
            return None, self.get_board_score(board)
        my_moves_scores = []
        
        # CHECK IF NOT TOO LONG INDUCTION FOR engine_depth-depth
        best_move_scores = self.get_n_best_move_scores(board, self.n_best_moves, self.engine_depth-depth)
        best_score = -float('inf') 
        for move, score in best_move_scores.items():
            board.push(move)
            self.move_history.append(board.copy())
            self.evaluation_history.append(score)

            _, my_score = self.induce_opponents_move(board, depth+1, alpha=alpha, beta=beta)

            best_score = max(my_score, best_score)

            self.evaluation_history.pop()
            my_moves_scores.append((move, my_score))
            self.move_history.pop()
            board.pop()

        return max(my_moves_scores, key=lambda x: x[1])
    
    def induce_opponents_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        self.depth_check = max(depth, self.depth_check)
        
        alpha = kwargs.get('alpha', -float('inf'))
        beta = kwargs.get('beta', float('inf'))

        encoded_states = list()
        if board.legal_moves.count() == 0:
            return None, self.get_board_score(board)
        
        self.evaluation_history = list(map(lambda x: -1*x, self.evaluation_history))
        self.is_white = not self.is_white
        mock_score = self.evaluation_history[-1]

        legal_moves = list(board.legal_moves)
        for next_move in legal_moves:
            board.push(next_move)
            self.move_history.append(board.copy())
            self.evaluation_history.append(-1*mock_score)

            encoded_state = self.model.encode(
                self.move_history,
                self.evaluation_history,
                self.get_additional_attributes()
            )
                
            encoded_states.append(encoded_state)

            self.evaluation_history.pop()
            self.move_history.pop()
            board.pop()

        self.is_white = not self.is_white
        self.evaluation_history = list(map(lambda x: -1*x, self.evaluation_history))

        choice_probs = self.model.predict_batch(encoded_states)
        best_probs_index = np.argsort(choice_probs.reshape(-1))

        moves_scores = list()

        best_score = float('inf')
        for i in best_probs_index[-self.n_best_moves:]:
            next_move = legal_moves[i]
            board.push(next_move)
            self.move_history.append(board.copy())
            score = self.get_board_score(board)
            self.evaluation_history.append(score)

            prob = choice_probs[i]
            score = score * (1 - prob) if score > 0 else score * prob
            
            if depth <= self.max_depth:
                _, induced_score = self.induce_own_move(board, depth+1, alpha=alpha, beta=beta)
                moves_scores.append((next_move, induced_score))
            else:
                moves_scores.append((next_move, score))

            self.evaluation_history.pop()
            self.move_history.pop()
            board.pop()

        return min(moves_scores, key=lambda x: x[1])