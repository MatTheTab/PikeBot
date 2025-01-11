import chess
from typing import Tuple
from utils.pikeBot_chess_utils import Pikebot
import stocksnake

class PikeBotHeuristic5(Pikebot):  
    def __init__(self,
            model,
            aggregate,
            stockfish_path):
        super().__init__(
            model,
            aggregate,
            stockfish_path)
        self.b=None
    def induce_own_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        
        my_moves_scores = []
        my_moves = self.b.legal_moves()
        for move in my_moves:
         
            mv=self.stockfish_to_pychess(move)
            board.push(mv)
          
            self.b.push(move)
            if self.b.in_check():
                sc=list()
                for move in self.b.evasion_moves():
                    self.b.push(move)
                    sc.append(self.get_board_score(board))
                    self.b.pop()
                score=max(sc)
            else:
                score = self.get_board_score(board)
            self.move_history.append(board.copy())
            self.evaluation_history.append(score)

            _, my_score = self.induce_opponents_move(
                board,
                )
            
            self.evaluation_history.pop()
            my_moves_scores.append((mv, my_score))
            self.move_history.pop()
            board.pop()
            self.b.pop()
        

        return max(my_moves_scores, key=lambda x: x[1])

    def induce_opponents_move(
            self,
            board: chess.Board,
            depth: int=0,
            **kwargs,
            ) -> Tuple[chess.Move, float]:
        
        opponent_moves = self.b.legal_moves()
        used_moves = list()
        encoded_states = list()
        scores = list()

        if not opponent_moves:
            return None, self.get_board_score(board)

        for next_move in opponent_moves:
                
                mv=self.stockfish_to_pychess(next_move)
                board.push(mv)
                self.b.push(next_move)
                if self.b.in_check():
                    sc=list()
                    for move in self.b.evasion_moves():
                        self.b.push(move)
                        sc.append(self.get_board_score(board))
                        self.b.pop()
                    score=max(sc)
                else:
                    score = self.get_board_score(board)
                self.move_history.append(board.copy())
                self.evaluation_history.append(score)

                encoded_state = self.model.encode(
                    self.move_history,
                    self.evaluation_history,
                    self.get_additional_attributes())
                
                used_moves.append(mv)
                encoded_states.append(encoded_state)
                scores.append(score)

                self.evaluation_history.pop()
                self.move_history.pop()
                board.pop()
                self.b.pop()
        choice_probs = self.model.predict_batch(encoded_states)
        prediction_vars = list(zip(used_moves, choice_probs, scores))
        return self.aggregate(prediction_vars)
        
    def get_board_score(self, board) -> int:
            return self.b.stockfish_value()
   # def get_board_score(self, board) -> int:
            if self.color=='white':
                return self.b.stockfish_value()
            else: 

                return -self.b.stockfish_value()
       

    def is_en_passant(self, move):

        if not self.board.piece_type_at(move.from_square) == chess.PAWN:
            return False
    
        file_diff = abs(chess.square_file(move.from_square) - chess.square_file(move.to_square))
        if file_diff != 1:
            return False
            
        return move.to_square == self.board.ep_square

    def is_castling(self, move):
        """
        Check if move is castling based on coordinates and piece type.
        
        Args:
            board: chess.Board instance representing the current position
            move: chess.Move instance to check
        """

        if not self.board.piece_type_at(move.from_square) == chess.KING:
            return False
            
       
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
 
        return (from_rank == to_rank and abs(to_file - from_file) == 2)
    def stockfish_to_pychess(self, move_int):

        if move_int == 0: 
            return chess.Move.null()
        
        from_square = (move_int >> 6) & 0x3F
        to_square = move_int & 0x3F
        move_type = (move_int >> 14) & 0x3
        
        promotion = None
        if move_type == 1:
            promotion_code = (move_int >> 12) & 0x3
            promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promotion_code]
        
        return chess.Move(from_square, to_square, promotion=promotion)

    def pychess_to_stockfish(self, move):

        if move.null():
            return 0 
        
        move_int = (move.from_square << 6) | move.to_square
        
        if move.promotion:
            move_int |= (1 << 14) 
            promo_piece = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
                chess.QUEEN: 3
            }[move.promotion]
            move_int |= (promo_piece << 12)
        
        elif self.is_en_passant(move):
            move_int |= (2 << 14)  
        
        elif self.is_castling(move):
            move_int |= (3 << 14) 
        
        return move_int

    def get_best_move(self, board):
        '''
        Gets the best move.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
        self.b=stocksnake.Position(board.fen())
        #save opponents move and its evaluation to the history
        self.save_to_history(board)
        
        
        
        best_move, _ = self.induce_own_move(board)

     
        board_copy2 = board.copy()
        board_copy2.push(best_move)
        self.save_to_history(board_copy2)

        return best_move