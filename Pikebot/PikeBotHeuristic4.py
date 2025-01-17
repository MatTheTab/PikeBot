import chess
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot


class PikeBotHeuristic4(Pikebot):
    def induce_own_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        
        my_moves_scores = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)
            score = self.get_board_score(board)
            self.move_history.append(board.copy())
            self.evaluation_history.append(score)

            _, my_score = self.induce_opponents_move(
                board,
                )
            
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
        
        opponent_moves = list(board.legal_moves)
        used_moves = list()
        encoded_states = list()
        scores = list()

        if not opponent_moves:
            return None, self.get_board_score(board)

        for next_move in opponent_moves:
                board.push(next_move)
                score = self.get_board_score(board)
                self.move_history.append(board.copy())
                self.evaluation_history.append(score)


                if self.evaluation_history[-1] - self.evaluation_history[-2] < 50 and len (encoded_states)>0:
                    self.evaluation_history.pop()
                    self.move_history.pop()
                    board.pop()
                    continue
                
                encoded_state = self.model.encode(
                    self.move_history,
                    self.evaluation_history,
                    self.get_additional_attributes())
                
                used_moves.append(next_move)
                encoded_states.append(encoded_state)
                scores.append(score)

                self.evaluation_history.pop()
                self.move_history.pop()
                board.pop()
        
        choice_probs = self.model.predict_batch(encoded_states)
        prediction_vars = list(zip(used_moves, choice_probs, scores))
        return self.aggregate(prediction_vars)
