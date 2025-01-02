import chess
import numpy as np
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot

class RandomHeuristic(Pikebot):

    def __init__(self, model, aggregate, stockfish_path: str, color: str = "white", time_limit: float = 0.001, engine_depth: int = 8, name: str = "PikeBot", opponents_elo: int = 1500):
        super().__init__(model, aggregate, stockfish_path, color, time_limit, engine_depth, name, opponents_elo)
        self.max_depth = 3
        self.n_best_moves = 5

    def induce_own_move(
            self,
            board: chess.Board,
            depth: int = 0,
            **kwargs
        ):
        
        alpha = kwargs.get('alpha', -float('inf'))
        beta = kwargs.get('beta', float('inf'))

        if board.legal_moves.count() == 0:
            return None, self.get_board_score(board)
        my_moves_scores = []
        
        best_move_scores = self.get_n_best_move_scores(board, self.n_best_moves, self.depth-depth)
        best_score = -float('inf') 
        for move, score in best_move_scores.items():
            board.push(move)
            self.move_history.append(board.copy())
            self.evaluation_history.append(score)

            _, my_score = self.induce_opponents_move(board, depth+1, alpha=alpha, beta=beta)

            # best_score = max(my_score, best_score)
            # alpha = max(my_score, best_score)
            # if beta <= alpha:
            #     self.evaluation_history.pop()
            #     self.move_history.pop()
            #     board.pop()
            #     return None, best_score

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
        
        alpha = kwargs.get('alpha', -float('inf'))
        beta = kwargs.get('beta', float('inf'))

        scores = list()

        if board.legal_moves.count() == 0:
            return None, self.get_board_score(board)
        
        mock_score = self.evaluation_history[-1]
        
        legal_moves = list(board.legal_moves)
        moves_scores = list()

        best_score = float('inf')
        for next_move in np.random.choice(legal_moves, replace=False, size=min(5, len(legal_moves))):
            board.push(next_move)
            self.move_history.append(board.copy())
            score = self.get_board_score(board)
            self.evaluation_history.append(score)
            
            if depth < self.max_depth:
                _, induced_score = self.induce_own_move(board, depth+1, alpha=alpha, beta=beta)
            else:
                induced_score = score

            moves_scores.append((next_move, induced_score))

            # best_score = min(best_score, induced_score)
            # beta = min(best_score, beta)
            # if beta <= alpha:
            #     self.evaluation_history.pop()
            #     self.move_history.pop()
            #     board.pop()
            #     return None, best_score

            self.evaluation_history.pop()
            self.move_history.pop()
            board.pop()

        return min(moves_scores, key=lambda x: x[1])