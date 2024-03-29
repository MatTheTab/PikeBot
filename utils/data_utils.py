import chess.pgn
import zstandard
import io
import numpy as np
import os
from collections import deque
import pandas as pd
import chess.engine
import random


def str_to_bitboard_all_pieces(str_notation):
    '''
    Converts a chess board string notation into a binary bitboard representing all pieces.

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.

    Returns:
    - bit_board: A 2D numpy array representing the board state, where 1 indicates the presence of a piece and 0 indicates an empty square.
    '''
    row_num=0
    col_num=0
    bit_board = np.zeros((8, 8))
    for val in str_notation:
        if val.isdigit():
            row_num += int(val)
        elif val == "/":
            col_num += 1
            row_num = 0
        elif val!=" ":
            bit_board[col_num][row_num] = 1
            row_num+=1
        else:
            break
    return bit_board

def str_to_board_white(str_notation):
    '''
    Converts a chess board string notation into a binary bitboard representing white board pieces.

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.

    Returns:
    - bit_board: A 2D numpy array representing the board state, where 1 indicates the presence of a piece and 0 indicates an empty square.
    '''
    row_num=0
    col_num=0
    bit_board = np.zeros((8, 8))
    for val in str_notation:
        if val.isdigit():
            row_num += int(val)
        elif val == "/":
            col_num += 1
            row_num = 0
        elif val.isupper():
            bit_board[col_num][row_num] = 1
            row_num += 1
        elif val == " ":
            break
        else:
            row_num += 1
    return bit_board

def str_to_board_black(str_notation):
    '''
    Converts a chess board string notation into a binary bitboard representing black board pieces.

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.

    Returns:
    - bit_board: A 2D numpy array representing the board state, where 1 indicates the presence of a piece and 0 indicates an empty square.
    '''
    row_num=0
    col_num=0
    bit_board = np.zeros((8, 8))
    for val in str_notation:
        if val.isdigit():
            row_num += int(val)
        elif val == "/":
            col_num += 1
            row_num = 0
        elif val.islower():
            bit_board[col_num][row_num] = 1
            row_num += 1
        elif val == " ":
            break
        else:
            row_num += 1
    return bit_board

def str_to_board_figure(str_notation, figure = "p"):
    '''
    Converts a chess board string notation into a binary bitboard represetning presence of a given piece, regardless of piece color.

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.
    - figure: character representing the piece, regardless of color, lowercase - possible values: [p, k, n, r, q, k]

    Returns:
    - bit_board: A 2D numpy array representing the board state, where 1 indicates the presence of a piece and 0 indicates an empty square.
    '''
    row_num=0
    col_num=0
    bit_board = np.zeros((8, 8))
    for val in str_notation:
        if val.isdigit():
            row_num += int(val)
        elif val == "/":
            col_num += 1
            row_num = 0
        elif val.lower() == figure:
            bit_board[col_num][row_num] = 1
            row_num += 1
        elif val == " ":
            break
        else:
            row_num += 1
    return bit_board

def str_to_board_figure_color(str_notation, figure = "p"):
    '''
    Converts a chess board string notation into a binary bitboard represetning presence of a given piece, taking into account the piece color.

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.
    - figure: character representing the piece, , lowercase - black, upper case - white
        Possible values:
        - White -> [P, K, N, R, Q, K]
        - Black -> [p, k, n, r, q, k]

    Returns:
    - bit_board: A 2D numpy array representing the board state, where 1 indicates the presence of a piece and 0 indicates an empty square.
    '''
    row_num=0
    col_num=0
    bit_board = np.zeros((8, 8))
    for val in str_notation:
        if val.isdigit():
            row_num += int(val)
        elif val == "/":
            col_num += 1
            row_num = 0
        elif val == figure:
            bit_board[col_num][row_num] = 1
            row_num += 1
        elif val == " ":
            break
        else:
            row_num += 1
    return bit_board

def str_to_board_all_figures(str_notation, figs = ["p", "r", "n", "b", "q", "k"]):
    '''
    Converts a chess board string notation into a numpy array of bitboards, representing all the the figures indicated in the figs list, regardless of color.
    For more info see str_to_board_figure()

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.
    - figs: list representing all figures for which a bitboard should be created, one figure -> one bitboard

    Returns:
    - bit_board: A 3D numpy array where each array represents a 2D numpy array
    '''
    complete_bit_board = []
    for fig in figs:
        board = str_to_board_figure(str_notation, figure = fig)
        complete_bit_board.append(board)
    return np.array(complete_bit_board)

def str_to_board_all_figures_colors(str_notation, figs = ["p", "r", "n", "b", "q", "k", "P", "R", "N", "B", "Q", "K"]):
    '''
    Converts a chess board string notation into a numpy array of bitboards, representing all the the figures indicated in the figs list, takes color into account.
    For more info see str_to_board_figure_color()

    Parameters:
    - str_notation: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.
    - figs: list representing all figures for which a bitboard should be created, one figure -> one bitboard

    Returns:
    - bit_board: A 3D numpy array where each array represents a 2D numpy array
    '''
    complete_bit_board = []
    for fig in figs:
        board = str_to_board_figure_color(str_notation, figure = fig)
        complete_bit_board.append(board)
    return np.array(complete_bit_board)

def get_all_attacks(board):
    '''
    Generates attack bitboards for all squares on the chess board.

    Parameters:
    - board: A chess.Board object representing the current board state.

    Returns:
    - bit_boards: A 3D numpy array representing attack bitboards for all squares on the board.
    Each 2D slice corresponds to the attack bitboard for a specific square.
    '''
    bit_boards = []
    for square in chess.SQUARES:
        row_num = 0
        col_num = 0
        bit_board = np.zeros((8, 8))
        attack_board = board.attacks(square)
        for val in str(attack_board).replace(" ",""):
            if row_num == 8:
                row_num = 0
                col_num += 1
            if val == "1":
                bit_board[col_num][row_num] = 1
                row_num+=1
            elif val == ".":
                row_num+=1
        bit_boards.append(bit_board)
    return np.array(bit_boards)

def get_bitboards(str_board, board, str_functions, board_functions):
    '''
    Generates a collection of bitboards from various board representations and functions.

    Parameters:
    - str_board: A string representing the chess board in FEN (Forsyth-Edwards Notation) format.
    - board: A chess.Board object representing the current board state.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.

    Returns:
    - complete_board: A 3D numpy array containing all generated bitboards.
    Each 2D slice corresponds to a single bitboard representing a specific aspect of the board.
    '''
    complete_board = []
    boards = []
    for func in str_functions:
        temp_board = func(str_board)
        boards.append(temp_board)

    for func in board_functions:
        temp_board = func(board)
        boards.append(temp_board)

    for bit_board in boards:
        if len(bit_board.shape)>2:
            for sub_board in bit_board:
                complete_board.append(sub_board)
        else:
            complete_board.append(bit_board)
    return np.array(complete_board)


def generate_fake_game(engine, depth, time_limit, df_filename, file_path, game_number, game, str_functions, board_functions, directory = "D:\\PikeBot\\Processed_Data"):
    '''
    Generates a fake chess game and saves the game data to a CSV file. 
    The games are generated by picking random moves.
    Depricated, left in here just in case.

    Parameters:
    - engine: The chess engine used for analysis.
    - depth: Depth parameter for engine analysis.
    - time_limit: Time limit for engine analysis.
    - df_filename: Name of the CSV file to save the game data.
    - file_path: Identifier for the game file path.
    - game_number: Identifier for the game number.
    - game: The chess game object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - directory: Directory to save the processed data. Default is "D:\\PikeBot\\Processed_Data".

    Returns:
    - None
    '''
    data = {"human": [],"player": [], "elo": [], "color": [], "event": [], "stockfish_score": [], "past_move_1": [], "past_move_2": [], "past_move_3": [], "past_move_4": [], "past_move_5": [], "past_move_6": [], "past_move_7": [], "past_move_8": [], "past_move_9": [],  "past_move_10": [], "past_move_11": [], "past_move_12": [], "current_move": []}
    j=0
    file_path_queue = deque(maxlen=12)
    filename = os.path.join(directory, f"fake_game_{file_path}_game_{game_number}_move_{j}.npy")
    empty_filename = os.path.join(directory, f"fake_game_{file_path}_game_{game_number}_move_empty.npy")
    df_filepath = os.path.join(directory, df_filename)
    
    for _ in range(12):
        file_path_queue.append(f"fake_game_{file_path}_game_{game_number}_move_empty.npy")     
    
    white_player = "Bot"
    black_player = "Bot"
    white_elo = random.randint(400, 2500)
    black_elo = random.randint(400, 2500)
    
    board = chess.Board()
    info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
    score = info['score'].pov(color=chess.WHITE).score(mate_score=900)
    data["stockfish_score"].append(score)
    str_board = board.fen()
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    empty_board = np.zeros_like(new_board)
    np.save(empty_filename, empty_board)
    np.save(filename, new_board)

    data["current_move"].append(f"fake_game_{file_path}_game_{game_number}_move_{j}.npy")
    data["event"].append(game.headers["Event"])
    data["player"].append(white_player)
    data["color"].append("White")
    data["elo"].append(white_elo)
    data["human"].append(False)

    x=1
    for temp_path in file_path_queue:
        data[f"past_move_{x}"].append(temp_path)
        x+=1
    file_path_queue.append(filename)
    
    while not board.is_game_over() and j<40:
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        j+=1
        board.push(move)
        str_board = board.fen()
        x=1
        for temp_path in file_path_queue:
            data[f"past_move_{x}"].append(temp_path)
            x+=1
        new_board = get_bitboards(str_board, board, str_functions, board_functions)
        filename = os.path.join(directory, f"fake_game_{file_path}_game_{game_number}_move_{j}.npy")
        np.save(filename, new_board)  
        data["current_move"].append(f"fake_game_{file_path}_game_{game_number}_move_{j}.npy")
        file_path_queue.append(f"fake_game_{file_path}_game_{game_number}_move_{j}.npy")

        data["human"].append(False)
        data["event"].append(game.headers["Event"])
        if board.turn == chess.WHITE:
            data["player"].append(white_player)
            data["color"].append("White")
            data["elo"].append(white_elo)
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
            score = info['score'].pov(color=chess.WHITE).score(mate_score=900)
            data["stockfish_score"].append(score)
        else:
            data["player"].append(black_player)
            data["color"].append("Black")
            data["elo"].append(black_elo)
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
            score = info['score'].pov(color=chess.BLACK).score(mate_score=900)
            data["stockfish_score"].append(score)
    df = pd.DataFrame(data)
    df.to_csv(df_filepath, index=False)

def set_scores(board, engine, depths, time_limits, color, mate_score, data):
        '''
        Set stockfish scores for chosen depths and time limits.
        Requires data dictionary to expect key in format stockfish_score_depth_{depth}
        for every depth in provided list.
        '''
        for i, depth in enumerate(depths):
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limits[i]))
            score = info['score'].pov(color=color).score(mate_score=mate_score)
            data[f"stockfish_score_depth_{depth}"].append(score)

def save_game_data(engine, depths, time_limits, df_filename, file_path, game_number, game, str_functions, board_functions, directory = "D:\\PikeBot\\New_Processed_Data"):
    '''
    Read and transform data from pgn game format to dataframe with appropriate information.
    '''
    data = {"human": [],"player": [], "elo": [], "color": [], "event": [], "clock": [], "stockfish_score_depth_1": [], "stockfish_score_depth_2": [], "stockfish_score_depth_3": [], "stockfish_score_depth_4": [], "stockfish_score_depth_5": [], "stockfish_score_depth_8": [], "stockfish_score_depth_10": [], "stockfish_score_depth_12": [], "stockfish_score_depth_15": [], "stockfish_score_depth_16": [], "stockfish_score_depth_18": [], "stockfish_score_depth_20": [], "past_move_1": [], "past_move_2": [], "past_move_3": [], "past_move_4": [], "past_move_5": [], "past_move_6": [], "past_move_7": [], "past_move_8": [], "past_move_9": [],  "past_move_10": [], "past_move_11": [], "past_move_12": [], "current_move": []}
    j=0
    file_path_queue = deque(maxlen=12)
    filename = os.path.join(directory, f"{file_path}_game_{game_number}_move_{j}.npy")
    empty_filename = os.path.join(directory, f"{file_path}_game_{game_number}_move_empty.npy")
    df_filepath = os.path.join(directory, df_filename)
    
    for _ in range(12):
        file_path_queue.append(f"{file_path}_game_{game_number}_move_empty.npy")     
    
    white_player = game.headers["White"]
    black_player = game.headers["Black"]
    white_elo = game.headers["WhiteElo"]
    black_elo = game.headers["BlackElo"]
    
    board = game.board()
    set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data)
    str_board = board.fen()
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    empty_board = np.zeros_like(new_board)
    np.save(empty_filename, empty_board)
    np.save(filename, new_board)

    data["current_move"].append(f"{file_path}_game_{game_number}_move_{j}.npy")
    data["event"].append(game.headers["Event"])
    for node in game.mainline():
        clock = node.clock()
        break
    data["clock"].append(clock)
    data["player"].append(white_player)
    data["color"].append("Starting Move")
    data["elo"].append(white_elo)
    data["human"].append(True)

    x=1
    for temp_path in file_path_queue:
        data[f"past_move_{x}"].append(temp_path)
        x+=1
    file_path_queue.append(filename)
    
    for node in game.mainline():
        j+=1
        player_color = board.turn
        #random move
        legal_moves = list(board.legal_moves)
        bot_move = random.choice(legal_moves)
        bot_clock = node.clock()
        board.push(bot_move)
        bot_str_board = board.fen()
        x=1
        for temp_path in file_path_queue:
            data[f"past_move_{x}"].append(temp_path)
            x+=1
        bot_new_board = get_bitboards(bot_str_board, board, str_functions, board_functions)
        bot_filename = os.path.join(directory, f"bot_{file_path}_game_{game_number}_move_{j}.npy")
        np.save(bot_filename, bot_new_board)  
        data["current_move"].append(f"bot_{file_path}_game_{game_number}_move_{j}.npy")

        data["human"].append(False)
        data["event"].append(game.headers["Event"])
        data["clock"].append(bot_clock)
        if player_color == chess.WHITE:
            data["player"].append("bot")
            data["color"].append("White")
            data["elo"].append(white_elo)
            set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data)
        else:
            data["player"].append("bot")
            data["color"].append("Black")
            data["elo"].append(black_elo)
            set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data)
        board.pop()

        #human move
        move = node.move
        clock = node.clock()
        board.push(move)
        str_board = board.fen()
        x=1
        for temp_path in file_path_queue:
            data[f"past_move_{x}"].append(temp_path)
            x+=1
        new_board = get_bitboards(str_board, board, str_functions, board_functions)
        filename = os.path.join(directory, f"{file_path}_game_{game_number}_move_{j}.npy")
        np.save(filename, new_board)  
        data["current_move"].append(f"{file_path}_game_{game_number}_move_{j}.npy")
        file_path_queue.append(f"{file_path}_game_{game_number}_move_{j}.npy") #Do not add this to random moves!

        data["human"].append(True)
        data["event"].append(game.headers["Event"])
        data["clock"].append(clock)
        if player_color == chess.WHITE:
            data["player"].append(white_player)
            data["color"].append("White")
            data["elo"].append(white_elo)
            set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data)
        else:
            data["player"].append(black_player)
            data["color"].append("Black")
            data["elo"].append(black_elo)
            set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data)

    df = pd.DataFrame(data)
    df.to_csv(df_filepath, index=False)



def save_data(txt_file_dir, txt_file_name, directory_path, file_name, verbose = True, str_functions = [str_to_board_all_figures_colors], board_functions = [get_all_attacks], stockfish_path = "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe", depths = [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20], time_limits = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]): #Double check time limits later
    '''
    Saves data extracted from PGN games into CSV files.
    
    Parameters:
    - txt_file_dir (str): The directory path where the text files containing filenames will be stored.
    - txt_file_name (str): The name of the text file containing filenames.
    - directory_path (str): The directory path where PGN files are stored.
    - file_name (str): The name of the PGN file.
    - verbose (bool, optional): If True, prints the number of processed games in the file. Defaults to True.
    - str_functions (list, optional): List of functions to convert board state to string representation. Defaults to [str_to_board_all_figures_colors].
    - board_functions (list, optional): List of functions to apply on chess board. Defaults to [get_all_attacks].
    - stockfish_path (str, optional): The path to the Stockfish chess engine executable. Defaults to "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe".
    - depths (list, optional): List of depths for Stockfish engine analysis. Defaults to [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20].
    - time_limits (list, optional): List of time limits for Stockfish engine analysis. Defaults to [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]. Should be the same length as depths list.
    '''
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    file_path = os.path.join(directory_path, file_name)
    txt_file_path = os.path.join(txt_file_dir, txt_file_name)

    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            pgn_text = decompressed_file.read().decode("utf-8")

    pgn_io = io.StringIO(pgn_text)
    i = 0
    while True:
        pgn_game = chess.pgn.read_game(pgn_io)
        if pgn_game is None:
            break

        df_filename = f"{file_name}_game_{i}_df.csv"
        mode = 'a' if os.path.exists(txt_file_path) else 'w'
        with open(txt_file_path, mode) as file:
          file.write(df_filename + '\n')
        save_game_data(engine, depths, time_limits, df_filename, file_name, i, pgn_game, str_functions, board_functions)
        i+=1
    
    if verbose:
        print(f"Num processed games in a file = {i}")

def read_data(text_file_path, num_dataframes=None, skip_positions=0):
    '''
    Reads data from multiple CSV files as pandas DataFrames.

    Parameters:
    - text_file_path: Path to the text file containing a list of CSV filenames.
    - num_dataframes: Number of CSV files to read. If None, reads all files. Default is None.
    - skip_positions: Number of CSV filenames to skip from the beginning of the list. Default is 0.

    Returns:
    - dataframes: A list of pandas DataFrames containing the data read from the CSV files.
    '''
    dataframes = []
    with open(text_file_path, 'r') as file:
        csv_filenames = file.read().splitlines()
    csv_filenames = csv_filenames[skip_positions:]
    if num_dataframes is not None:
        csv_filenames = csv_filenames[:num_dataframes]
    for csv_filename in csv_filenames:
        csv_file_path = os.path.join(os.path.dirname(text_file_path), csv_filename)
        df = pd.read_csv(csv_file_path)
        dataframes.append(df)
    
    return dataframes



def read_all(text_file_path, num_dataframes=None, skip_positions=0):
    '''
    Reads data from multiple CSV files along with associated numpy array files (npy) and returns as pandas DataFrames.

    Parameters:
    - text_file_path: Path to the text file containing a list of CSV filenames.
    - num_dataframes: Number of CSV files to read. If None, reads all files. Default is None.
    - skip_positions: Number of CSV filenames to skip from the beginning of the list. Default is 0.

    Returns:
    - dataframes: A list of pandas DataFrames containing the data read from the CSV files along with associated numpy arrays.
    '''
    dataframes = []
    with open(text_file_path, 'r') as file:
        csv_filenames = file.read().splitlines()
    csv_filenames = csv_filenames[skip_positions:]
    if num_dataframes is not None:
        csv_filenames = csv_filenames[:num_dataframes]
    for csv_filename in csv_filenames:
        csv_file_path = os.path.join(os.path.dirname(text_file_path), csv_filename)
        df = pd.read_csv(csv_file_path)
        for column_name in df.columns:
            if 'move' in column_name:
                for index, row in df.iterrows():
                    npy_file_path = os.path.join(os.path.dirname(csv_file_path), row[column_name])
                    numpy_array = np.load(npy_file_path)
                    df.at[index, column_name] = numpy_array
        
        dataframes.append(df)
    return dataframes

