from stockfish import Stockfish
import chess
import chess.engine
import random

def get_random_chessboard(min_moves=20, max_moves=100):
    """
    Generate a chessboard at a position created after making between min_moves and max_moves random legal moves.

    Parameters:
    - min_moves (int, optional): minimum legal moves to be made.
    - max_moves (int, optional): maximum legal moves to be made.

    Returns:
    - chess.Board: a chessboard at a random position
    """
    assert min_moves < max_moves, "min_moves should be lower than max_moves"

    num_moves = random.randint(30, 80)
    board = chess.Board()
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        random_move = random.choice(legal_moves)
        board.push(random_move)
    
    return board

def evaluate_all_moves_simple_engine(board, engine, time_limit=0.1):
    """
    Evaluate all legal moves on the given board using a simple chess engine. We iterate over every legal move using board.legal_moves
    and use the engine's evaluate function to obtain scores of each move.

    Parameters:
    - board (chess.Board): The chess board to evaluate.
    - engine (chess.engine.SimpleEngine): The simple chess engine to use for evaluation.
    - time_limit (float, optional): The time limit for each move evaluation in seconds. Defaults to 0.1.

    Returns:
    - dict: A dictionary where keys are legal moves and values are their corresponding scores.
    """
    print("hello")
    all_moves = list(board.legal_moves)
    move_scores = {}
    for move in all_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        move_scores[move] = info['score'].relative.score()
        board.pop()
    return move_scores

def evaluate_all_moves_stock_fish(board, engine):
    """
    Evaluate all legal moves on the given board using Stockfish library. We iterate over every legal move using board.legal_moves
    and use the engine's get_evaluation function to obtain the score.

    Parameters:
    - board (chess.Board): The chess board to evaluate.
    - engine (Stockfish): The Stockfish chess engine to use for evaluation.

    Returns:
    - dict: A dictionary where keys are legal moves and values are their corresponding scores.
    """
    all_moves = list(board.legal_moves)
    move_scores = {}
    for move in all_moves:
        board.push(move)
        info = engine.get_evaluation()
        move_scores[move] = info['value']
        board.pop()
    #engine.quit()
    return move_scores

def generate_moves_tree(board, depth):
    """
    Generate a tree of legal moves up to a specified depth for the given board. Works recursively.

    Parameters:
    - board (chess.Board): The chess board to generate moves tree for.
    - depth (int): The depth of the moves tree to generate.

    Returns:
    - list: A list representing the moves tree. Each element of the list is a tuple containing a move and its child moves tree.
            Due to the way the function works, for depth=3 you will have 4 levels of nested lists with last one being empty.
    """
    if depth == 0:
        return []
    legal_moves = list(board.legal_moves)
    moves_tree = []
    for move in legal_moves:
        board.push(move)
        child_moves = generate_moves_tree(board, depth - 1)
        board.pop()
        moves_tree.append((move, child_moves))
    return moves_tree

def show_moves_tree(tree, board, depth=0):
    """
    Display the moves tree recursively.

    Parameters:
    - tree (list): The moves tree to display.
    - board (chess.Board): The chess board corresponding to the root of the moves tree.
    - depth (int, optional): The depth of the current node in the moves tree. Defaults to 0. By default only used for printing the depth.
    """
    for move, subtree in tree:
        print(f"Depth: {depth}")
        board.push(move)
        print(board)
        show_moves_tree(subtree, board.copy(), depth + 1)
        board.pop()

def evaluate_tree_engine(board, depth, engine, time_limit=0.1):
    """
    Generate a tree of legal moves up to a specified depth for the given board
    while evaluating each position using an engine, in this case chess' Simple Engine.

    Parameters:
    - board (chess.Board): The chess board to generate moves tree for.
    - depth (int): The depth of the moves tree to generate.
    - engine (chess.engine.SimpleEngine): engine to evaluate the tree positions
    - time limit: default = 0.1, maximum time a model can spend evaluating the position

    Returns:
    - list: A list representing the moves tree. Each element of the list is a tuple containing a move, score and child_moves.
            The last element is an empty list due to how the function works.
    """
    if depth == 0:
        return []
    legal_moves = list(board.legal_moves)
    moves_tree = []
    for move in legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info['score'].relative.score()
        child_moves = evaluate_tree_engine(board, depth - 1, engine)
        board.pop()
        moves_tree.append((move, score, child_moves))
    return moves_tree

def evaluate_tree_stockfish(board, depth, engine):
    """
    Generate a tree of legal moves up to a specified depth for the given board
    while evaluating each position using an engine, in this case Stockfish engine using Stockfish library.
    Parameters:
    - board (chess.Board): The chess board to generate moves tree for.
    - depth (int): The depth of the moves tree to generate.
    - engine (Stockfish): engine to evaluate the tree positions

    Returns:
    - list: A list representing the moves tree. Each element of the list is a tuple containing a move, score and child_moves.
            The last element is an empty list due to how the function works.
    """
    if depth == 0:
        return []
    legal_moves = list(board.legal_moves)
    moves_tree = []
    for move in legal_moves:
        board.push(move)
        info = engine.get_evaluation()
        score = info['value']
        child_moves = evaluate_tree_stockfish(board, depth - 1, engine)
        board.pop()
        moves_tree.append((move, score, child_moves))
    return moves_tree

def show_evaluated_moves(tree, board, depth=0):
    """
    Display the moves and scores of the tree recursively.

    Parameters:
    - tree (list): The moves and scores tree to display.
    - board (chess.Board): The chess board corresponding to the root of the moves tree.
    - depth (int, optional): The depth of the current node in the moves tree. Defaults to 0. By default only used for printing the depth.
    """
    for move, score, subtree in tree:
        print(f"Depth: {depth}, score: {score}")
        board.push(move)
        print(board)
        show_evaluated_moves(subtree, board.copy(), depth + 1)
        board.pop()

class Player:
    '''
    Player class, implements basic functionality necessary for every Bot to have.
    '''
    def play_move(self, move, board):
        '''
        Play chosen move.

        Parameters:
        - move (chess.Move): The move to be played.
        - board (chess.Board): The current state of the chess board.
        '''
        board.push(move)

    def is_game_over(self, board):
        '''
        Check if the game is over.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - bool: True if the game is over, False otherwise.
        '''
        return board.is_game_over()

    def display_board(self, board):
        '''
        Display the current state of the chess board.

        Parameters:
        - board (chess.Board): The current state of the chess board.
        '''
        print(board)

class OptimalSimpleEngineAgent(Player):
    '''
    OptimalSimpleEngineAgent class, represents a player agent using a simple chess engine for optimal moves.

    Attributes:
    - engine (chess.engine.SimpleEngine): The simple chess engine used by the agent.
    - time_limit (float): The time limit for move calculation in seconds.

    Methods:
    - __init__(self, stockfish_path, time_limit=0.1): Initializes the OptimalSimpleEngineAgent object.
    - get_best_move(self, board): Returns the best move calculated by the engine.
    - close(self): Closes the engine.
    '''
    def __init__(self, stockfish_path, limits={"time": 0.1}):
        '''
        Initialize the OptimalSimpleEngineAgent object.

        Parameters:
        - stockfish_path (str): The path to the Stockfish executable.
        - limits (dict, optional): The limits to be used in chess.engine.Limit. Defaults to 0.1 second time limit.
        '''
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.settings = limits

    def get_best_move(self, board):
        '''
        Get the best move calculated by the engine.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the engine.
        '''
        result = self.engine.play(board, chess.engine.Limit(**self.limits))
        return result.move

    def close(self):
        '''
        Closes the engine - run this at the end of use.
        '''
        self.engine.quit()

class OptimalStockFishAgent(Player):
    '''
    OptimalStockFishAgent class, represents a player agent using Stockfish for optimal moves.

    Attributes:
    - engine (Stockfish): The Stockfish engine used by the agent.

    Methods:
    - __init__(self, stockfish_path, depth=8, settings={"Threads": 2, "Hash": 2048}): Initializes the OptimalStockFishAgent object.
    - get_best_move(self, board): Returns the best move calculated by the engine.
    '''
    def __init__(self, stockfish_path, depth=8, settings = {"Threads": 2, "Hash": 2048}):
        '''
        Initialize the OptimalStockFishAgent object.

        Parameters:
        - stockfish_path (str): The path to the Stockfish executable.
        - depth (int, optional): The search depth for Stockfish engine. Defaults to 8.
        - settings (dict, optional): Additional parameters for Stockfish engine. Defaults to {"Threads": 2, "Hash": 2048}.
       '''
        stockfish = Stockfish(path=stockfish_path, depth=depth, parameters=settings)
        self.engine = stockfish

    def get_best_move(self, board):
        '''
        Get the best move calculated by the engine.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the engine.
        '''
        # Update opponents move in the engine's chessboard
        try:
            previous_move = board.peek()
            self.engine.make_moves_from_current_position([previous_move])
        except IndexError:
            # If first move continue
            return chess.Move.from_uci(self.engine.get_best_move())

        return chess.Move.from_uci(self.engine.get_best_move())
    
    def play_move(self, move, board):
        # Move in the engine's chessboard
        self.engine.make_moves_from_current_position([move])

        board.push(move)


class Human(Player):
    '''
    Human class, represents a human player.

    Methods:
    - get_best_move(self, board): Gets the move input from the human player.
    '''
    def get_best_move(self, board):
        '''
        Gets the move input from the human player.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The move input by the human player.
        '''
        while True:
            try:
                move = input("\nEnter your move (e.g., e2e4): \n")
                return chess.Move.from_uci(move)
            except ValueError:
                print("Invalid move. Please enter a move in the format 'e2e4'.")


def play_chess(white_player, black_player, mute=False):
    '''
    Play a game of chess between two players.

    Parameters:
    - white_player (Player): The player controlling the white pieces.
    - black_player (Player): The player controlling the black pieces.
    - mute (bool, optional): If True, suppresses the printing of game progress. Defaults to False.

    Returns:
    - int: 1 if white wins, -1 if black wins, 0 if it's a tie.
    '''
    if not mute:
        print("Game Started!")
    board = chess.Board()
    while not board.is_game_over():
        print()
        if not mute:
            white_player.display_board(board)

        white_move = white_player.get_best_move(board)
        if not mute:
            print("\nWhite's Move:", white_move, "\n")
        white_player.play_move(white_move, board)
        if not mute:
            black_player.display_board(board)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                if not mute:
                    print("\nWhite Wins!\n")
                return 1
            elif result == "0-1":
                if not mute:
                    print("\nBlack Wins!\n")
                return -1
            else:
                if not mute:
                    print("\nIt's a tie!\n")
                return 0
            
        black_move = black_player.get_best_move(board)
        if not mute:
            print("\nBlack's Move:", black_move, "\n")
        black_player.play_move(black_move, board)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                if not mute:
                    print("\nWhite Wins!\n")
                return 1
            elif result == "0-1":
                if not mute:
                    print("\nBlack Wins!\n")
                return -1
            else:
                if not mute:
                    print("\nIt's a tie!\n")
                return 0
            break

def play_chess_debug(white_player, black_player):
    '''
    Play a game of chess between two players, additional output for debuging purposes.

    Parameters:
    - white_player (Player): The player controlling the white pieces.
    - black_player (Player): The player controlling the black pieces.

    Returns:
    - int: 1 if white wins, -1 if black wins, 0 if it's a tie.
    '''
    print("Game Started!")
    board = chess.Board()
    while not board.is_game_over():
        print()
        white_player.display_board(board)

        white_move = white_player.get_best_move_verbose(board)
        print("\nWhite's Move:", white_move, "\n")
        white_player.play_move(white_move, board)

        black_player.display_board(board)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                print("\nWhite Wins!\n")
            elif result == "0-1":
                print("\nBlack Wins!\n")
            else:
                print("\nIt's a tie!\n")
            break

        black_move = black_player.get_best_move_verbose(board)
        print("\nBlack's Move:", black_move, "\n")
        black_player.play_move(black_move, board)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                print("\nWhite Wins!\n")
            elif result == "0-1":
                print("\nBlack Wins!\n")
            else:
                print("\nIt's a tie!\n")
            break

class Uniform_model:
    '''
    Uniform_model class, represents a uniform model for board evaluation.

    Attributes:
    - model_path (str): The path to the model file.

    Methods:
    - __init__(self, model_path): Initializes the Uniform_model object.
    - encode(self, board): Encodes the board state in a representation compatible with the model (what will be fed into predict).
    - predict(self, board_state): Predicts the chance of a human making this move - in this case always 1.0
    '''
    def __init__(self, model_path):
        '''
        Initializes the Uniform_model object.

        Parameters:
        - model_path (str): The path to the model file.
        '''
        print("Model Initialized!")

    def encode(self, board):
        '''
        Encodes the board state.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Board: The encoded board state.
        '''
        return board

    def predict(self, board_state):
        '''
        Predicts the value of the board state.

        Parameters:
        - board_state (chess.Board): The encoded board state.

        Returns:
        - float: The predicted value of the board state.
        '''
        return 1.0
    
def mean_aggr(preds_scores):
    '''
    Calculate the mean value of predictions based on choice probabilities and scores. Multiply for each of opponents move the probability they will do this move times its score (score respective to how good it is for us)
    then calculate mean of these outcomes for each of our moves and pick the best move, i.e. choose the move with the highest average utility of score*prob.

    Parameters:
    - preds_scores (list): A list of tuples containing predictions and scores for each of our and opponents moves (our_move, opponents move, probability, score)

    Returns:
    - chess.Move: The move with the highest average utility
    '''
    move_stats = {}
    for move, next_move, choice_prob, score in preds_scores:
        move_stats[move] = move_stats.get(move, [0, 0])
        move_stats[move][0] += 1
        if score is None or choice_prob is None:
            value = 0
        else:
            value = choice_prob * score
        move_stats[move][1] += value
    move_means = {}
    for move, stats in move_stats.items():
        total_occurrences, total_value = stats
        move_means[move] = total_value / total_occurrences if total_occurrences > 0 else float('-inf')
    best_move = max(move_means, key=move_means.get)
    return best_move

def mean_aggr_debug(preds_scores):
    '''
    Calculate the mean value of predictions based on choice probabilities and scores. Multiply for each of opponents move the probability they will do this move times its score (score respective to how good it is for us)
    then calculate mean of these outcomes for each of our moves and pick the best move, i.e. choose the move with the highest average utility of score*prob. Additional info printed for debugging purposes.

    Parameters:
    - preds_scores (list): A list of tuples containing predictions and scores for each of our and opponents moves (our_move, opponents move, probability, score)

    Returns:
    - chess.Move: The move with the highest average utility
    '''
    print(preds_scores)
    move_stats = {}
    for move, next_move, choice_prob, score in preds_scores:
        move_stats[move] = move_stats.get(move, [0, 0])
        move_stats[move][0] += 1
        value = choice_prob * score
        move_stats[move][1] += value
    print(move_stats)
    move_means = {}
    for move, stats in move_stats.items():
        total_occurrences, total_value = stats
        move_means[move] = total_value / total_occurrences if total_occurrences > 0 else float('-inf')
    print(move_means)
    best_move = max(move_means, key=move_means.get)
    return best_move

def max_aggr(preds_scores):
    '''
    Calculate the best-case move prediction as move which leads to the maximum value of probability of move multiplied by chance the opponent will make this move.

    Parameters:
    - preds_scores (list): A list of tuples containing predictions and scores for each of our and opponents moves (our_move, opponents move, probability, score)

    Returns:
    - chess.Move: The move with the highest average utility
    '''
    best_move = None
    best_score = float("-inf")
    for move, next_move, choice_prob, score in preds_scores:
        if choice_prob*score>best_score:
            best_score = choice_prob*score
            best_move = move
    if best_move is None:
        best_move = preds_scores[0][0]
    return best_move

def max_aggr_debug(preds_scores):
    '''
    Calculate the best-case move prediction as move which leads to the maximum value of probability of move multiplied by chance the opponent will make this move. Includes additional print statments for debugging.

    Parameters:
    - preds_scores (list): A list of tuples containing predictions and scores for each of our and opponents moves (our_move, opponents move, probability, score)

    Returns:
    - chess.Move: The move with the highest average utility
    '''
    best_move = None
    best_score = float("-inf")
    for move, next_move, choice_prob, score in preds_scores:
        if choice_prob*score>best_score:
            print(f"New best move: {move}")
            print(f"Opponents Move: {next_move}")
            print(f"New best score: {choice_prob*score}")
            print(f"Prev Score: {best_score}")
            print(f"Evaluated Move Val: {score}")
            print(f"Move Prob: {choice_prob}")
            best_score = choice_prob*score
            best_move = move
    if best_move is None:
        print(f"Failsafe triggered: {preds_scores[0][0]}")
        best_move = preds_scores[0][0]
    return best_move


class ChessBot(Player):
    '''
    ChessBot class, represents a chess-playing agent combining model-based and engine-based evaluation.

    Attributes:
    - name (str): The name of the bot.
    - engine (chess.engine.SimpleEngine): The engine used for move analysis.
    - time_limit (float): The time limit for move analysis.
    - model: The model used for board evaluation.
    - aggregate (function): The function used for aggregating move predictions.
    - depth (int): The depth of search for the bot's moves.
    - engine_depth (int): The depth of search for the engine's analysis.
    - color (chess.Color): The color of the bot, either chess.WHITE or chess.BLACK.

    Methods:
    - __init__(self, model, aggregate, stockfish_path, color="white", time_limit=0.01, engine_depth=20, name="ChessBot"): Initializes the ChessBot object.
    - __str__(self): Returns a string representation of the ChessBot object.
    - get_best_move_verbose(self, board): Gets the best move considering verbose output.
    - get_best_move(self, board): Gets the best move.
    - close(self): Closes the engine.
    '''
    def __init__(self, model, aggregate, stockfish_path, color="white", time_limit=0.01, engine_depth=20, name="ChessBot"):
        '''
        Initializes the ChessBot object.

        Parameters:
        - model: The model used for board evaluation.
        - aggregate (function): The function used for aggregating move predictions.
        - stockfish_path (str): The path to the Stockfish executable.
        - color (str, optional): The color of the bot, either "white" or "black". Defaults to "white".
        - time_limit (float, optional): The time limit for move analysis. Defaults to 0.01.
        - engine_depth (int, optional): The depth of search for the engine's analysis. Defaults to 20.
        - name (str, optional): The name of the bot. Defaults to "ChessBot".
        '''
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.time_limit = time_limit
        self.model = model
        self.aggregate = aggregate
        self.depth = 1
        self.engine_depth = engine_depth
        if color=="white":
            self.color = chess.WHITE
        else:
            self.color = chess.BLACK

    def __str__(self):
        '''
        Returns a string representation of the ChessBot object.
        '''
        print(f"-----------{self.name}-----------")
        print(f"Engine: {self.engine}")
        print(f"Time Limit: {self.time_limit}")
        print(f"Model: {self.model}")
        print(f"Aggregating Function: {self.aggregate}")
        print()

    def get_best_move_verbose(self, board):
        '''
        Gets the best move considering verbose output.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
        prediction_vars = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)
            opponent_moves = list(board.legal_moves)
            for next_move in opponent_moves:
                board.push(next_move)
                print()
                print("Board: ")
                print(board)
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth, time=self.time_limit))
                score = info['score'].pov(color=self.color).score(mate_score=900)
                board_state = self.model.encode(board)
                choice_prob = self.model.predict(board_state)
                print("Probability = ", choice_prob)
                print("Score = ", score)
                print("Info Score = ", info)
                prediction_vars.append(tuple([move, next_move, choice_prob, score]))
                board.pop()
            board.pop()
        print(prediction_vars)
        best_move = self.aggregate(prediction_vars)
        return best_move

    def get_best_move(self, board):
        '''
        Gets the best move.

        Parameters:
        - board (chess.Board): The current state of the chess board.

        Returns:
        - chess.Move: The best move calculated by the bot.
        '''
        prediction_vars = []
        my_moves = list(board.legal_moves)
        for move in my_moves:
            board.push(move)
            opponent_moves = list(board.legal_moves)
            for next_move in opponent_moves:
                board.push(next_move)
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.engine_depth, time=self.time_limit))
                score = info['score'].pov(color=self.color).score(mate_score=900)
                board_state = self.model.encode(board)
                choice_prob = self.model.predict(board_state)
                prediction_vars.append(tuple([move, next_move, choice_prob, score]))
                board.pop()
            board.pop()
        best_move = self.aggregate(prediction_vars)
        return best_move

    def close(self):
        self.engine.quit()


