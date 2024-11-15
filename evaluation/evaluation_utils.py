import chess
import time
import utils.chess_utils as chess_utils
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List

stockfish_path = "D:/Program Files/Stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
assert os.path.exists(stockfish_path)

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

    for board, move_history in boards_dataset:
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

def process_games(boards: list, name1:str, name2:str, save_dir: str) -> List[List[int]]:
    """
    For each of the boards get evaluation of each move and save the
    game to the pgn file.
    """
    evaluations = list()
    
    for i, board in enumerate(boards):
        new_board = chess.Board()
        evaluation = [get_score(new_board)]

        game = chess.pgn.Game()
        node = game

        for move in board.move_stack:
            new_board.push(move)
            node = node.add_variation(move)
            evaluation.append(get_score(new_board))
        evaluations.append(evaluation)

        with open(f'results/{save_dir}/games/{name1}_{name2}_{i}.pgn', 'w') as pgn_file:
            #save game
            game.headers["Event"] = f"test game {i}"
            game.headers["White"] = name1
            game.headers["Black"] = name2
            game.headers["Result"] = new_board.result()
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            pgn_file.write(str(game) + "\n\n")

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

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

def get_score(board: chess.Board):
    result = engine.analyse(board,  chess.engine.Limit(depth=8))
    value = result['score'].pov(color=chess.WHITE).score(mate_score=900)
    return value