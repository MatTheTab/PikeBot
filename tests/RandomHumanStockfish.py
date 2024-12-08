import os
import re
from utils.chess_utils import *

class RandomHumanStockfish(Player):
    '''
    OptimalStockFishAgent class, represents a player agent using Stockfish scores randomly picked
    according to human move quality distributions

    Attributes:
    - engine (Stockfish): The Stockfish engine used by the agent.

    Methods:
    - __init__(self, stockfish_path, elo, histograms_path, depth=8): Initializes the OptimalSimpleEngineAgent object.
    - get_best_move(self, board): Returns the best move calculated by the engine.
    - analyse_board(self, board): Runs stockfish evaluation fo all legal moves in a given position.
    - close(self): Closes the engine.
    '''
    def __init__(self, stockfish_path, elo, histograms_path, color:str="white", limits={"depth": 8}):
        '''
        Initialize the OptimalStockFishAgent object.

        Parameters:
        - stockfish_path (str): The path to the Stockfish executable.
        - elo (int): Elo distribution to be picked.
        - histograms_path (str): Path to human move quality histograms.
        - color (chess.Color): The color of the bot, either chess.WHITE or chess.BLACK.
        - limits (dict, optional): The limits to be used in chess.engine.Limit. Defaults to depth of 8.
       '''
        self.color = color
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine = stockfish
        self.elo = elo
        self.limits = limits

        filenames = os.listdir(os.path.join(histograms_path, 'bins'))
        
        pattern = re.compile(r'(\d+)-?(\d+)?\+?\.npy')    
        elo_ranges = [list(map(int, match)) for match in pattern.findall(' '.join(filenames[:-1]))]
        elo_ranges.append([int(filenames[-1].split("+")[0])+1, 3000])
        index = next((i for i, (x, y) in enumerate(elo_ranges) if x <= elo <= y), -1)
        
        self.bins = np.load(os.path.join(histograms_path, 'bins', filenames[index]))
        self.counts = np.load(os.path.join(histograms_path, 'counts', filenames[index]))

    def analyse_board(self, board: chess.Board) -> dict:
        """
        Runs stockfish evaluation fo all legal moves in a given position.

        Parameters
        -board (chess.Board): current state of the chess board.

        Returns
        -move_scores (dict): dictionary of all legal moves with corresponding scores.
        """
        # Set MultiPV to the number of legal moves
        legal_moves_count = len(list(board.legal_moves))
        info = self.engine.analyse(board, chess.engine.Limit(**self.limits), multipv=legal_moves_count)
        
        move_scores = {}
        for entry in info:
            move = entry["pv"][0]  # Principal Variation (PV) move
            score = entry["score"].pov(color=self.color).score(mate_score=900)  # Score from White's perspective
            move_scores[move] = score

        return move_scores
    
    def get_closest_move(self, moves, target_score):
        """
        Returns the move that is closest to target score.
        """
        return min(moves, key=lambda move: abs(moves[move] - target_score))

    def get_normalized_scores(self, moves):
        """
        Returns a dictionary of moves with stockfish scores converted to normalized move scoring.
        """
        scores = np.fromiter(moves.values(), dtype=np.int16)

        if len(scores) == 1:
            return {moves.keys()[0]: 1.0}

        min_score = np.min(scores)
        max_score = np.max(scores)
        
        new_scores = {}
        for move, score in moves.items():
            new_scores[move] = (score - min_score) / (max_score - min_score)
        
        return new_scores

    def get_best_move(self, board):
        '''
        Get the best move calculated by the engine.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the engine.
        '''
        move_scores = self.analyse_board(board)
        scores = np.fromiter(move_scores.values(), dtype=np.int16)
        
        # Return a random move if all scores equal or one move available
        if len(scores) == 1 or np.max(scores) - np.min(scores) == 0:
            return np.random.choice(list(move_scores.keys()))

        norm_scores = self.get_normalized_scores(move_scores)
        return self.get_closest_move(norm_scores, np.random.choice(self.bins[:-1], p=self.counts/sum(self.counts)))
    
    def close(self):
        '''
        Closes the engine - run this at the end of use.
        '''
        self.engine.quit()


class RandomHumanStockfishDebug(Player):
    '''
    DEBUG MODE

    OptimalStockFishAgent class, represents a player agent using Stockfish scores randomly picked
    according to human move quality distributions

    Attributes:
    - engine (Stockfish): The Stockfish engine used by the agent.

    Methods:
    - __init__(self, stockfish_path, elo, histograms_path, depth=8): Initializes the OptimalSimpleEngineAgent object.
    - get_best_move(self, board): Returns the best move calculated by the engine.
    - analyse_board(self, board): Runs stockfish evaluation fo all legal moves in a given position.
    - close(self): Closes the engine.
    '''
    def __init__(self, stockfish_path, elo, histograms_path, color:str="white", limits={"depth": 8}):
        '''
        Initialize the OptimalStockFishAgent object.

        Parameters:
        - stockfish_path (str): The path to the Stockfish executable.
        - elo (int): Elo distribution to be picked.
        - histograms_path (str): Path to human move quality histograms.
        - color (chess.Color): The color of the bot, either chess.WHITE or chess.BLACK.
        - limits (dict, optional): The limits to be used in chess.engine.Limit. Defaults to depth of 8.
       '''
        self.color = color
        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine = stockfish
        self.elo = elo
        self.limits = limits

        filenames = os.listdir(os.path.join(histograms_path, 'bins'))
        
        pattern = re.compile(r'(\d+)-?(\d+)?\+?\.npy')    
        elo_ranges = [list(map(int, match)) for match in pattern.findall(' '.join(filenames[:-1]))]
        elo_ranges.append([int(filenames[-1].split("+")[0])+1, 3000])
        index = next((i for i, (x, y) in enumerate(elo_ranges) if x <= elo <= y), -1)

        print("Selected elo range:", elo_ranges[index])
        
        self.bins = np.load(os.path.join(histograms_path, 'bins', filenames[index]))
        self.counts = np.load(os.path.join(histograms_path, 'counts', filenames[index]))

    def analyse_board(self, board: chess.Board) -> dict:
        """
        Runs stockfish evaluation fo all legal moves in a given position.

        Parameters
        -board (chess.Board): current state of the chess board.

        Returns
        -move_scores (dict): dictionary of all legal moves with corresponding scores.
        """
        # Set MultiPV to the number of legal moves
        legal_moves_count = len(list(board.legal_moves))
        info = self.engine.analyse(board, chess.engine.Limit(**self.limits), multipv=legal_moves_count)
        
        move_scores = {}
        for entry in info:
            move = entry["pv"][0]  # Principal Variation (PV) move
            score = entry["score"].pov(color=self.color).score(mate_score=900)  # Score from White's perspective
            move_scores[move] = score

        return move_scores
    
    def get_closest_move(self, moves, target_score):
        """
        Returns the move that is closest to target score.
        """
        min_move = min(moves, key=lambda move: abs(moves[move] - target_score))
        return min_move 

    def get_normalized_scores(self, moves):
        """
        Returns a dictionary of moves with stockfish scores converted to normalized move scoring.
        """
        scores = np.fromiter(moves.values(), dtype=np.int16)

        if len(scores) == 1:
            return {moves.keys()[0]: 1.0}

        min_score = np.min(scores)
        max_score = np.max(scores)
        
        new_scores = {}
        for move, score in moves.items():
            new_scores[move] = (score - min_score) / (max_score - min_score)
        
        return new_scores

    def get_best_move(self, board):
        '''
        Get the best move calculated by the engine.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the engine.
        '''
        move_scores = self.analyse_board(board)
        scores = np.fromiter(move_scores.values(), dtype=np.int16)
        
        # Return a random move if all scores equal or one move available
        if len(scores) == 1 or np.max(scores) - np.min(scores) == 0:
            return np.random.choice(list(move_scores.keys()))

        norm_scores = self.get_normalized_scores(move_scores)
        random_quality = np.random.choice(self.bins[:-1], p=self.counts/sum(self.counts))
        closest_move = self.get_closest_move(norm_scores, random_quality)
        print(f"Closest move: {closest_move}, Quality: {norm_scores[closest_move]:.3f}, Random picked quality: {random_quality:.3f}")
        return closest_move
    
    def close(self):
        '''
        Closes the engine - run this at the end of use.
        '''
        self.engine.quit()