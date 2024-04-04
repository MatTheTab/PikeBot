import chess.pgn
import zstandard
import io
import numpy as np
import os
from collections import deque
import pandas as pd
import chess.engine
import random
import gzip


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

def compress_file(filename):
    '''
    Compresses a file using gzip compression and removes the original file.

    Parameters:
    - filename (str): Path to the file to be compressed.
    '''
    with open(filename, 'rb') as f_in:
        with gzip.open(filename+".gz", 'wb') as f_out:
            f_out.writelines(f_in)
    os.remove(filename)

def get_initial_filepaths(directory, game_number, df_filename, file_path, columns_data, j):
    '''
    Generates initial file paths and a file path queue.

    Parameters:
    - directory (str): Directory path.
    - game_number (int): Number of the game.
    - df_filename (str): Filename for the DataFrame.
    - file_path (str): File path for the game data.
    - columns_data (dict): Dictionary defining the columns of the DataFrame.
    - j (int): Move number.

    Returns:
    - file_path_queue (deque): Queue containing file paths.
    - filename (str): Path to the filename.
    - empty_filename (str): Path to the empty filename.
    - df_filepath (str): Path to the DataFrame file.
    '''
    file_path_queue = deque(maxlen=columns_data["num_past_moves"])
    filename = os.path.join(directory, f"{file_path}_game_{game_number}_move_{j}.npy")
    empty_filename = os.path.join(directory, f"{file_path}_game_{game_number}_move_empty.npy")
    df_filepath = os.path.join(directory, df_filename)
    return file_path_queue, filename, empty_filename, df_filepath

def get_initial_game_state(data, file_path_queue, game, engine, depths, time_limits, game_number, file_path, empty_filename, filename, str_functions, board_functions, columns_data, j):
    '''
    Initializes game state and saves initial board configurations.

    Parameters:
    - data (dict): Dictionary containing data.
    - file_path_queue (deque): Queue containing file paths.
    - game: Chess game object.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - game_number (int): Number of the game.
    - file_path (str): File path for the game data.
    - empty_filename (str): Path to the empty filename.
    - filename (str): Path to the filename.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - columns_data (dict): Dictionary defining the columns of the DataFrame.
    - j (int): Move number.

    Returns:
    - data (dict): Updated dictionary containing data.
    - board: Chess board object.
    - white_player (str): Name of the white player.
    - white_elo (str): Elo rating of the white player.
    - black_player (str): Name of the black player.
    - black_elo (str): Elo rating of the black player.
    - file_path_queue (deque): Updated queue containing file paths.
    '''
    for _ in range(columns_data["num_past_moves"]):
        file_path_queue.append(f"{file_path}_game_{game_number}_move_empty.npy.gz")     
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
    compress_file(empty_filename)
    compress_file(filename)
    data["current_move"].append(f"{file_path}_game_{game_number}_move_{j}.npy.gz")
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
    file_path_queue.append(filename+".gz")
    return data, board, white_player, white_elo, black_player, black_elo, file_path_queue

def get_save_random_move(board, node, game_number, directory, str_functions, board_functions, file_path, j, file_path_queue, data, player_color, engine, depths, time_limits, game, white_elo, black_elo):
    '''
    Performs and saves a random move made by the bot.

    Parameters:
    - board: Chess board object.
    - node: Chess node object.
    - game_number (int): Number of the game.
    - directory (str): Directory path.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - file_path (str): File path for the game data.
    - j (int): Move number.
    - file_path_queue (deque): Queue containing file paths.
    - data (dict): Dictionary containing data.
    - player_color: Color of the player.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - game: Chess game object.
    - white_elo (str): Elo rating of the white player.
    - black_elo (str): Elo rating of the black player.

    Returns:
    - data (dict): Updated dictionary containing data.
    - file_path_queue (deque): Updated queue containing file paths.
    - board: Updated chess board object.
    '''
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
    compress_file(bot_filename)
    data["current_move"].append(f"bot_{file_path}_game_{game_number}_move_{j}.npy.gz")

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
    return data, file_path_queue, board 

def get_save_human_move(board, node, game_number, directory, str_functions, board_functions, file_path, j, file_path_queue, data, player_color, engine, depths, time_limits, game, white_player, white_elo, black_player, black_elo):
    '''
    Performs and saves a human move.

    Parameters:
    - board: Chess board object.
    - node: Chess node object.
    - game_number (int): Number of the game.
    - directory (str): Directory path.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - file_path (str): File path for the game data.
    - j (int): Move number.
    - file_path_queue (deque): Queue containing file paths.
    - data (dict): Dictionary containing data.
    - player_color: Color of the player.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - game: Chess game object.
    - white_player (str): Name of the white player.
    - white_elo (str): Elo rating of the white player.
    - black_player (str): Name of the black player.
    - black_elo (str): Elo rating of the black player.

    Returns:
    - data (dict): Updated dictionary containing data.
    - file_path_queue (deque): Updated queue containing file paths.
    - filename (str): Path to the saved file.
    - board: Updated chess board object.
    '''
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
    compress_file(filename)
    data["current_move"].append(f"{file_path}_game_{game_number}_move_{j}.npy.gz")
    file_path_queue.append(f"{file_path}_game_{game_number}_move_{j}.npy.gz")

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
    return data, file_path_queue, filename, board

def save_game_data(engine, depths, time_limits, df_filename, file_path, game_number, game, str_functions, board_functions,
                    directory = "D:\\PikeBot\\New_Processed_Data", columns_data = {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True},
                    shuffle = True, seed = 42):
    '''
    Read and transform data from PGN game format to a DataFrame with appropriate information, and save it to a compressed CSV file.

    Parameters:
    - engine (str): Stockfish engine used for evaluating moves.
    - depths (list): Depths of the search for the engine.
    - time_limits (list): Time limits for the search.
    - df_filename (str): Filename for the DataFrame.
    - file_path (str): File path for the game data.
    - game_number (int): Number of the game.
    - game: The chess game object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - directory (str, optional): Directory path where the data will be saved. Defaults to "D:\\PikeBot\\New_Processed_Data".
    - columns_data (dict, optional): Dictionary defining the columns of the DataFrame. Defaults to {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True}.
    - shuffle (bool, optional): If the datframe should be shuffled before being returned, default True
    - seed (int, optional): seed for shuffling, default 42
    '''

    data = {}
    data = initialize_data(data, depths, columns_data)
    j=0
    file_path_queue, filename, empty_filename, df_filepath = get_initial_filepaths(directory, game_number, df_filename, file_path, columns_data, j)
    data, board, white_player, white_elo, black_player, black_elo, file_path_queue = get_initial_game_state(data, file_path_queue, game, engine,
                                                                                                             depths, time_limits, game_number,
                                                                                                               file_path, empty_filename, filename,
                                                                                                                 str_functions, board_functions, columns_data, j)
    for node in game.mainline():
        j+=1
        player_color = board.turn
        data, file_path_queue, board = get_save_random_move(board, node, game_number, directory, str_functions, board_functions, file_path, j, file_path_queue, data, player_color, engine, depths, time_limits, game, white_elo, black_elo)
        data, file_path_queue, filename, board = get_save_human_move(board, node, game_number, directory, str_functions, board_functions, file_path, j, file_path_queue, data, player_color, engine, depths, time_limits, game, white_player, white_elo, black_player, black_elo)
    df = pd.DataFrame(data)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(df_filepath, index=False, compression='gzip')



def save_data(txt_file_dir, txt_file_name, directory_path, file_name, verbose = True, str_functions = [str_to_board_all_figures_colors], board_functions = [get_all_attacks], 
              stockfish_path = "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe", depths = [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20], 
              time_limits = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], max_num_games = np.inf,
              columns_data = {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True},
              shuffle = True, seed = 42): #Double check time limits later
    '''
    Saves data extracted from PGN games into CSV files. Uses .gz compression to minimize file size.
    
    Parameters:
    - txt_file_dir (str): The directory path where the text files containing filenames will be stored.
    - txt_file_name (str): The name of the text file containing filenames.
    - directory_path (str): The directory path where PGN files are stored.
    - file_name (str): The name of the PGN file.
    - verbose (bool, optional): If True, prints the number of processed games in the file. Defaults to True.
    - str_functions (list, optional): List of functions to convert board state to bitboard representation. Defaults to [str_to_board_all_figures_colors].
    - board_functions (list, optional): List of functions to apply on chess board to convert to bitboard notation. Defaults to [get_all_attacks].
    - stockfish_path (str, optional): The path to the Stockfish chess engine executable. Defaults to "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe".
    - depths (list, optional): List of depths for Stockfish engine analysis. Defaults to [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20].
    - time_limits (list, optional): List of time limits for Stockfish engine analysis. Defaults to [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]. Should be the same length as depths list.
    - max_num_games (int, optional): Number of games to read from the PGN file, by default np.infinite, i.e. all games will be read from the file.
    - shuffle (bool, optional): If the datframe should be shuffled before being returned, default True
    - seed (int, optional): seed for shuffling, default 42
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
        if pgn_game is None or i >= max_num_games:
            break

        df_filename = f"{file_name}_game_{i}_df.csv.gz"
        mode = 'a' if os.path.exists(txt_file_path) else 'w'
        with open(txt_file_path, mode) as file:
          file.write(df_filename + '\n')
        save_game_data(engine, depths, time_limits, df_filename, file_name, i, pgn_game, str_functions, board_functions, directory = txt_file_dir, columns_data = columns_data, shuffle = shuffle, seed = seed)
        i+=1
    
    if verbose:
        print(f"Num processed games in a file = {i}")

def read_data(text_file_path, num_dataframes=None, skip_positions=0, shuffle = True, seed = 42):
    '''
    Reads data from multiple CSV files as pandas DataFrames. Supports .csv files compressed as .gz files.

    Parameters:
    - text_file_path: Path to the text file containing a list of CSV filenames.
    - num_dataframes: Number of CSV files to read. If None, reads all files. Default is None.
    - skip_positions: Number of CSV filenames to skip from the beginning of the list. Default is 0.
    - shuffle (bool, optional): if list reperesenting the games should be shuffled
    - seed (int, optional): seed for shuffling, default 42

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
        if csv_filename.endswith('.gz'):
            df = pd.read_csv(csv_file_path, compression='gzip')
        else:
            df = pd.read_csv(csv_file_path)
        dataframes.append(df)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(dataframes)
    return dataframes



def read_all(text_file_path, num_dataframes=None, skip_positions=0, shuffle = True, seed = 42):
    '''
    Reads data from multiple CSV files along with associated numpy array files (npy) and returns as pandas DataFrames.
    Supports .csv files compressed as .gz files and .npy files compressed as .gz files.

    Parameters:
    - text_file_path: Path to the text file containing a list of CSV filenames.
    - num_dataframes: Number of CSV files to read. If None, reads all files. Default is None.
    - skip_positions: Number of CSV filenames to skip from the beginning of the list. Default is 0.
    - shuffle (bool, optional): if list reperesenting the games should be shuffled
    - seed (int, optional): seed for shuffling, default 42

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
        if csv_filename.endswith('.gz'):
            df = pd.read_csv(csv_file_path, compression='gzip')
        else:
            df = pd.read_csv(csv_file_path)
        for column_name in df.columns:
            if 'move' in column_name:
                for index, row in df.iterrows():
                    npy_file_path = os.path.join(os.path.dirname(csv_file_path), row[column_name])
                    if npy_file_path.endswith('.gz'):
                        with gzip.open(npy_file_path, 'rb') as f:
                            numpy_array = np.load(f)
                    else:
                        numpy_array = np.load(npy_file_path)
                    df.at[index, column_name] = numpy_array
        dataframes.append(df)
    if shuffle:
        random.seed(seed)
        random.shuffle(dataframes)
    return dataframes

def initialize_data(data, depths, columns_data):
    '''
    Initializes a dictionary with data placeholders.
    Dictionary of data will be filled with empty lists for all columns.

    Parameters:
    - data (dict): Dictionary to initialize.
    - depths (list): List of depth values.
    - columns_data (dict): Dictionary defining the columns of the DataFrame.

    Returns:
    - data (dict): Updated dictionary containing data.
    '''
    for arg in columns_data.keys():
        if arg != "num_past_moves" and columns_data[arg] == True:
            if arg != "depths":
                data[arg] = []
            else:
                for depth in depths:
                    data[f"stockfish_score_depth_{depth}"] = []
        elif arg == "num_past_moves":
            for i in range(columns_data[arg]):
                data[f"past_move_{i+1}"] = []
    return data

def initalize_starting_moves(data, columns_data, game, board, str_functions, board_functions):
    '''
    Initializes data and file paths for starting moves.

    Parameters:
    - data (dict): Dictionary containing data.
    - columns_data (dict): Dictionary defining the columns of the DataFrame.
    - game: Chess game object.
    - board: Chess board object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.

    Returns:
    - data (dict): Updated dictionary containing data.
    - file_path_queue (deque): Queue containing file paths.
    - white_player (str): Name of the white player.
    - white_elo (str): Elo rating of the white player.
    - black_player (str): Name of the black player.
    - black_elo (str): Elo rating of the black player.
    '''
    str_board = board.fen()
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    empty_board = np.zeros_like(new_board)

    file_path_queue = deque(maxlen=columns_data["num_past_moves"])
    for _ in range(columns_data["num_past_moves"]):
        file_path_queue.append(empty_board)     
    
    white_player = game.headers["White"]
    black_player = game.headers["Black"]
    white_elo = game.headers["WhiteElo"]
    black_elo = game.headers["BlackElo"]

    data["current_move"].append(new_board)
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
    file_path_queue.append(new_board)
    return data, file_path_queue, white_player, white_elo, black_player, black_elo

def get_random_move(game, player_color, data, engine, depths, time_limits, white_elo, black_elo, file_path_queue, board, node, str_functions, board_functions):
    '''
    Performs a random move made by the bot.

    Parameters:
    - game: Chess game object.
    - player_color: Color of the player.
    - data (dict): Dictionary containing data.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - white_elo (str): Elo rating of the white player.
    - black_elo (str): Elo rating of the black player.
    - file_path_queue (deque): Queue containing file paths.
    - board: Chess board object.
    - node: Chess node object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.

    Returns:
    - data (dict): Updated dictionary containing data.
    - file_path_queue (deque): Updated queue containing file paths.
    - board: Updated chess board object.
    '''
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
    data["current_move"].append(bot_new_board)

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
    return data, file_path_queue, board

def get_human_move(game, player_color, data, engine, depths, time_limits, white_player, white_elo, black_player, black_elo, file_path_queue, board, node, str_functions, board_functions):
    '''
    Performs a human move.

    Parameters:
    - game: Chess game object.
    - player_color: Color of the player.
    - data (dict): Dictionary containing data.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - white_player (str): Name of the white player.
    - white_elo (str): Elo rating of the white player.
    - black_player (str): Name of the black player.
    - black_elo (str): Elo rating of the black player.
    - file_path_queue (deque): Queue containing file paths.
    - board: Chess board object.
    - node: Chess node object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.

    Returns:
    - data (dict): Updated dictionary containing data.
    - file_path_queue (deque): Updated queue containing file paths.
    - board: Updated chess board object.
    '''
    move = node.move
    clock = node.clock()
    board.push(move)
    str_board = board.fen()
    x=1
    for temp_path in file_path_queue:
        data[f"past_move_{x}"].append(temp_path)
        x+=1
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    data["current_move"].append(new_board)
    file_path_queue.append(new_board)
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
    return data, file_path_queue, board

def greedy_save_game(engine, depths, time_limits, game, str_functions, board_functions, columns_data, shuffle = True, seed = 42):
    '''
    Saves the data of a chess game using a greedy strategy.

    Parameters:
    - engine: Chess engine object.
    - depths (list): List of depth values.
    - time_limits (list): List of time limits for the engine search.
    - game: Chess game object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - columns_data (dict): Dictionary defining the columns of the DataFrame.
    - shuffle (bool, optional): If the datframe should be shuffled before being returned, default True
    - seed (int, optional): seed for shuffling, default 42

    Returns:
    - df (DataFrame): DataFrame containing the saved data.
    '''
    data = {}
    data = initialize_data(data, depths, columns_data)
    j=0
    board = game.board()
    set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data)
    data, file_path_queue, white_player, white_elo, black_player, black_elo = initalize_starting_moves(data, columns_data, game, board, str_functions, board_functions)
    
    for node in game.mainline():
        j+=1
        player_color = board.turn
        data, file_path_queue, board = get_random_move(game, player_color, data, engine, depths, time_limits, white_elo, black_elo, file_path_queue, board, node, str_functions, board_functions)
        data, file_path_queue, board = get_human_move(game, player_color, data, engine, depths, time_limits, white_player, white_elo, black_player, black_elo, file_path_queue, board, node, str_functions, board_functions)
    df = pd.DataFrame(data)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def greedy_read(directory_path, file_name, verbose = True, str_functions = [str_to_board_all_figures_colors], board_functions = [get_all_attacks], 
                stockfish_path = "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe", depths = [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20], 
                time_limits = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], starting_game = 0, num_games = np.inf,
                columns_data = {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True},
                shuffle_df = True, shuffle_list = True, list_seed = 42):
    '''
    Reads and processes chess game data from a compressed file using a greedy strategy. Instead of saving the dataframe, everything is stored in the memory directly from PGN file, allows for dynamic data reading without preprocessing.

    Parameters:
    - directory_path (str): Path to the directory containing the compressed file.
    - file_name (str): Name of the compressed file.
    - verbose (bool, optional): If True, prints information during processing. Default is True.
    - str_functions (list, optional): List of functions to convert chess elements to strings. Default is [str_to_board_all_figures_colors].
    - board_functions (list, optional): List of functions to interact with the chess board. Default is [get_all_attacks].
    - stockfish_path (str, optional): Path to the Stockfish engine executable. Default is "D:\PikeBot\stockfish\stockfish-windows-x86-64-avx2.exe".
    - depths (list, optional): List of depth values for engine search. Default is [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20].
    - time_limits (list, optional): List of time limits for the engine search. Default is [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01].
    - starting_game (int, optional): Index of the first game to process. Default is 0.
    - num_games (int, optional): Number of games to process. Default is np.inf.
    - columns_data (dict, optional): Dictionary defining the columns of the DataFrame. Default is {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True}.
    - shuffle_df (bool, optional): If each datframe representing a game should be shuffled, default True
    - shuffle_list (bool, optional): If list representing num_games should be shuffled, default True
    - list_seed (int, optional): Seed for list shuffle

    Returns:
    - data (list): List of DataFrames containing processed data from each game.
    '''
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    file_path = os.path.join(directory_path, file_name)

    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            pgn_text = decompressed_file.read().decode("utf-8")

    pgn_io = io.StringIO(pgn_text)
    i = 0
    data = []
    while True:
        pgn_game = chess.pgn.read_game(pgn_io)
        if i<starting_game:
            i+=1
            continue
        if pgn_game is None or i >= num_games:
            break
        df = greedy_save_game(engine, depths, time_limits, pgn_game, str_functions, board_functions, columns_data, shuffle = shuffle_df)
        data.append(df)
        i+=1
    if shuffle_list:
        random.seed(list_seed)
        random.shuffle(data)
    
    if verbose:
        print(f"Num processed games in a file = {i}")
    return data