import chess
import numpy as np
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot

class PikeBotHeuristic8BC(Pikebot):

    def __init__(self, model, aggregate, stockfish_path: str, color: str = "white", time_limit: float = 0.001, engine_depth: int = 8, name: str = "PikeBot", opponents_elo: int = 1500,
                 max_depth: int = 3, n_best_moves: int = 5):
        super().__init__(model, aggregate, stockfish_path, color, time_limit, engine_depth, name, opponents_elo)
        self.max_depth = max_depth
        self.n_best_moves = n_best_moves
        self.stockfish_moves = 0
        self.non_stockfish_moves = 0

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
        
        # best_move_scores = self.get_n_best_move_scores(board, self.n_best_moves, self.engine_depth-depth)
        best_move_scores = self.get_n_best_move_scores_time(board, self.n_best_moves, self.time_limit)
        stockfish_move = max(best_move_scores, key=best_move_scores.get)

        better_move_found = False
        best_score = -float('inf') 
        for move, score in best_move_scores.items():
            board.push(move)
            self.move_history.append(board.copy())
            self.evaluation_history.append(score)

            # if the move rated by stockfish is highly deteriorating don't expand the tree
            if self.evaluation_history[-1] - self.evaluation_history[-2] < -100:
                my_score = -float('inf') 
            else:
                better_move_found = True
                _, my_score = self.induce_opponents_move(board, depth+1, alpha=alpha, beta=beta)

            best_score = max(my_score, best_score)

            self.evaluation_history.pop()
            my_moves_scores.append((move, my_score))
            self.move_history.pop()
            board.pop()

        if stockfish_move == max(my_moves_scores, key=lambda x: x[1])[0]:
            self.stockfish_moves += 1
        else:
            self.non_stockfish_moves += 1
            
        if not better_move_found:
            return stockfish_move, best_score

        return max(my_moves_scores, key=lambda x: x[1])
    
    def induce_opponents_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        
        alpha = kwargs.get('alpha', -float('inf'))
        beta = kwargs.get('beta', float('inf'))

        encoded_states = list()
        scores = list()
        moves = list()

        if board.legal_moves.count() == 0:
            return None, self.get_board_score(board)
        
        # op_best_move_scores = self.get_n_best_move_scores(board, self.n_best_moves, self.engine_depth-depth)
        op_best_move_scores = self.get_n_best_move_scores_time(board, self.n_best_moves, self.time_limit)

        # Skip highly deteriorating moves for opponent but leave at least the best possible move
        best_move_opponent = min(op_best_move_scores, key=op_best_move_scores.get)

        op_best_move_scores = {
            move: score
            for move, score in op_best_move_scores.items()
            if score - (-1*self.evaluation_history[-1]) > -100 or move == best_move_opponent
        }

        # Move should pass through the rest of the code to get prob adjusted score
        # if len(op_best_move_scores) == 1:
        #     move, op_move_score = list(op_best_move_scores.items())[0]
        #     return move, -1*op_move_score

        # Invert history for opponent induction
        self.evaluation_history = list(map(lambda x: -1*x, self.evaluation_history))
        self.is_white = not self.is_white
        for next_move, op_score in op_best_move_scores.items():
            board.push(next_move)
            self.move_history.append(board.copy())
            self.evaluation_history.append(op_score)

            encoded_state = self.model.encode(
                self.move_history,
                self.evaluation_history,
                self.get_additional_attributes()
            )
                
            encoded_states.append(encoded_state)
            # Store in opponent pov
            scores.append(-1*op_score)
            moves.append(next_move)

            self.evaluation_history.pop()
            self.move_history.pop()
            board.pop()

        # Invert back
        self.is_white = not self.is_white
        self.evaluation_history = list(map(lambda x: -1*x, self.evaluation_history))

        choice_probs = self.model.predict_batch(encoded_states)
        
        scores = np.array(scores)
        probabilities = choice_probs.reshape(-1)
        
        moves_scores = list()
        i = 0
        for next_move, op_score in op_best_move_scores.items():
            board.push(next_move)
            self.move_history.append(board.copy())
            # Revert score to our pov
            self.evaluation_history.append(-1*op_score)

            prob = probabilities[i]
            i += 1
            
            if depth <= self.max_depth:
                _, induced_score = self.induce_own_move(board, depth+1, alpha=alpha, beta=beta)
                moves_scores.append((next_move, induced_score))
            else:
                moves_scores.append((next_move, prob))

            self.evaluation_history.pop()
            self.move_history.pop()
            board.pop()

        return max(moves_scores, key=lambda x: x[1])