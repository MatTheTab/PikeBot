import chess
import time

from tqdm import tqdm

import chess.engine
import utils.chess_utils as chess_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import chess.pgn
from typing import List, Tuple, Dict
import atexit
from utils.utils import get_paths
config=get_paths()
stockfish_path=config['stockfish_path']
assert os.path.exists(stockfish_path)

engine: chess.engine.SimpleEngine 

def set_engine_history(engine: chess_utils.ChessBot, move_history: List[chess.Board]):
    engine.move_history = move_history
    engine.evaluation_history = [
        engine.get_board_score(history_board)
            for history_board in move_history
        ]


def compare_engines(engine1:chess_utils.Player, engine2:chess_utils.Player, boards_dataset:List[tuple[chess.Board, List[chess.Board]]]):
    engine1.color = chess.WHITE
    engine2.color = chess.BLACK
    results = []
    games = []

    for board, move_history in tqdm(boards_dataset):
        # set move history and evaluation history
        for engine in [engine1, engine2]:
            if isinstance(engine, chess_utils.ChessBot):
                set_engine_history(engine, move_history)
        result = chess_utils.play_chess(engine1, engine2, mute=True, board=board.copy())
        results.append(result[0])
        games.append(result[1])
        if isinstance(engine1, chess_utils.ChessBot):
            engine1.reset()
        if isinstance(engine2, chess_utils.ChessBot):
            engine2.reset()
    return results, games

def process_games(boards: list, name1:str, name2:str, save_dir: str, white_elo: int, black_elo: int) -> List[List[int]]:
    """
    For each of the boards get evaluation of each move and save the
    game to the pgn file.
    """
    evaluations = list()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    for i, board in enumerate(boards):
        new_board = chess.Board()
        evaluation = [get_score(new_board, engine)]

        game = chess.pgn.Game()
        node = game

        for move in board.move_stack:
            new_board.push(move)
            node = node.add_variation(move)
            evaluation.append(get_score(new_board, engine))
        evaluations.append(evaluation)

        with open(f'results/{save_dir}/games/{name1}_{name2}_{i}.pgn', 'w') as pgn_file:
            #save game
            game.headers["Event"] = f"test game {i}"
            game.headers["White"] = name1
            game.headers["Black"] = name2
            game.headers["Result"] = new_board.result()
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["WhiteElo"] = str(white_elo)
            game.headers["BlackElo"] = str(black_elo)
            pgn_file.write(str(game) + "\n\n")

    engine.quit()

    return evaluations

def process_results(results:list, name1:str='', name2:str='', dir_path:str=''):
        fig, ax = plt.subplots()
        counts = [results.count(i) for i in [1, 0, -1]]
        ax.bar(['white', 'draw', 'black'], 
                counts,
                color=['white', 'gray', 'black'])
        ax.set_facecolor('lightblue') 
        ax.set_title(f'{name1} vs {name2}')
        if dir_path:
                plt.savefig(f'results/{dir_path}/plots/{name1}_{name2}_result.png', dpi=300, bbox_inches='tight')
                with open(f'results/{dir_path}/statistics/{name1}_{name2}_winrate.csv', 'w') as output_file:
                        output_file.write('white;draw;black;\n')
                        for count in counts:
                                output_file.write(f'{count};')

        plt.show()

def process_evaluation(evaluation:list, name1:str='', name2:str='', dir_path:str=''):
    for eval in evaluation:
        plt.plot(eval, c='gray')
        if dir_path:
            with open(f'results/{dir_path}/statistics/{name1}_{name2}_evaluations.csv', 'a') as file:
                values_str = ';'.join(map(str, eval))
                file.write(values_str)
                file.write('\n')
    plt.title(f'white perspective evaluation {name1} vs {name2}')
    if dir_path:
        plt.savefig(f'results/{dir_path}/plots/{name1}_{name2}_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def process_length(boards: List[chess.Board], name1:str, name2:str, dir_path: str):
    game_lengths = [len(board.move_stack) for board in boards]
    lengths_count = np.unique(game_lengths, return_counts=True)
    plt.bar(lengths_count[0], lengths_count[1])
    plt.title(f'lengths of the games {name1} vs {name2}')
    if dir_path:
        plt.savefig(f'results/{dir_path}/plots/{name1}_{name2}_game_lengths.png', dpi=300, bbox_inches='tight')
        with open(f'results/{dir_path}/statistics/{name1}_{name2}_game_lengths.csv', 'w') as file:
            values_str = ';'.join(map(str, lengths_count[1]))
            labels_str = ';'.join(map(str, lengths_count[0]))
            file.write(labels_str + '\n')
            file.write(values_str + '\n')
    plt.show()


def get_score(board: chess.Board, engine):
    result = engine.analyse(board,  chess.engine.Limit(depth=8))
    value = result['score'].pov(color=chess.WHITE).score(mate_score=900)
    return value


def get_probabilities(
        engine: chess_utils.ChessBot,
        elo: int,
        board: chess.Board,
        model: chess_utils.MoveEvaluationModel
    ) -> List[float]:
    engine.opponents_elo = elo
    encoded_states = list()
    legal_moves = list(board.legal_moves)
    for next_move in legal_moves:
        board.push(next_move)
        score = engine.get_board_score(board)
        engine.move_history.append(board.copy())
        engine.evaluation_history.append(score)

        encoded_state = model.encode(engine.move_history, engine.evaluation_history, engine.get_additional_attributes())
        encoded_states.append(encoded_state)

        engine.evaluation_history.pop()
        engine.move_history.pop()
        board.pop()
    if len(legal_moves) == 0:
        return []
    choice_probs = engine.model.predict_batch(encoded_states).reshape(-1)
    return choice_probs

def get_move_evaluation_pairs(
        engine: chess_utils.ChessBot,
        elo: int,
        board: chess.Board,
        model: chess_utils.MoveEvaluationModel
) -> Dict[str, float]:
    choice_probs = get_probabilities(engine, elo, board, model)
    result = dict()
    moves = dict()
    for move, prob in zip(board.legal_moves, choice_probs):
        board.push(move)
        push_fen = board.fen()
        board.pop()
        result[push_fen] = prob
        moves[push_fen] = move
    return result, moves

def load_pgn(path: str) -> List[chess.pgn.Game]:
    games = list()
    with open(path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def write_pgn(games: List[chess.pgn.Game]):
    for game in games:
        game.headers["Event"] = "Example Event"
        game.headers["White"] = "Player 1"
        game.headers["Black"] = "Player 2"
        game.headers["Result"] = board.result()

        pgn_file.write(str(game) + "\n\n")


def get_boards_dataset(path: str) -> List[Tuple[chess.Board, List[chess.Board]]]:
    boards_dataset = list()
    
    games = load_pgn(path)
    for game in games:
        board = chess.Board()
        move_history = list()
        for move in game.mainline_moves():
            move_history.append(board.copy())
            board.push(move)
        boards_dataset.append((board, move_history))

    return boards_dataset


def evaluate_engines(
        engine1: chess_utils.Player,
        engine2: chess_utils.Player,
        boards_datasets: Dict[str, List[Tuple[chess.Board, List[chess.Board]]]],
        elo1: int, elo2: int
    ):
    for path, boards_dataset in boards_datasets.items():
        result1, boards1 = compare_engines(engine1, engine2, boards_dataset)
        result2, boards2 = compare_engines(engine2, engine1, boards_dataset)

        name1, name2 = type(engine1).__name__, type(engine2).__name__
        
        #define saving directory
        path_to_save = path.replace('games/', '')
        path_to_save = path_to_save.replace('.pgn', '')
        timestamp = time.asctime().replace(' ', '_')
        timestamp = timestamp.replace(':', '-')

        dir_path = f'{name1}_{name2}_{path_to_save}_{timestamp}'
        os.makedirs(f'results/{dir_path}', exist_ok=True) 
        os.makedirs(f'results/{dir_path}/games', exist_ok=True) 
        os.makedirs(f'results/{dir_path}/statistics', exist_ok=True) 
        os.makedirs(f'results/{dir_path}/plots', exist_ok=True) 

        #get move evaluation and save games
        evaluation1 = process_games(boards1, name1, name2, dir_path, elo1, elo2)
        evaluation2 = process_games(boards2, name2, name1, dir_path, elo2, elo1)

        #plot results and save plots
        process_results(result1, name1, name2, dir_path)
        process_results(result2, name2, name1, dir_path)

        process_evaluation(evaluation1, name1, name2, dir_path)
        process_evaluation(evaluation2, name2, name1, dir_path)

        process_length(boards1, name1, name2, dir_path) 
        process_length(boards2, name2, name1, dir_path) 