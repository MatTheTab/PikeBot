import chess
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot

class PikeBotHeuristic2(Pikebot):
    def induce_own_move(
            self,
            board: chess.Board,
            depth: int=1,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
          
        my_moves_scores = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)

            if depth > 0:
                _, my_score = self.induce_opponents_move(
                    board,
                    depth=depth-1
                    )
            else:
                my_score = self.get_board_score(board)

            my_moves_scores.append((move, my_score))
            board.pop()

        return max(my_moves_scores, key=lambda x: x[1])

    def induce_opponents_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        
        my_moves_scores = []
        my_moves = list(board.legal_moves)
        if not my_moves:
            return None, self.get_board_score(board)
        for move in my_moves:
            board.push(move)

            if depth > 0:
                _, my_score = self.induce_own_move(
                    board,
                    depth=depth-1
                    )
            else:
                my_score = self.get_board_score(board)

            my_moves_scores.append((move, my_score))
            board.pop()

        return min(my_moves_scores, key=lambda x: x[1])
