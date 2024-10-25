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

def set_scores(board, engine, depths, time_limits, color, mate_score, data, is_human):
        '''
        Set stockfish scores for chosen depths and time limits.
        Requires data dictionary to expect key in format stockfish_score_depth_{depth}
        for every depth in provided list.
        '''
        for i, depth in enumerate(depths):
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limits[i]))
            score = info['score'].pov(color=color).score(mate_score=mate_score)
            data[f"stockfish_score_depth_{depth}"].append(score)
            if len(data[f"stockfish_score_depth_{depth}"]) > 3:
                if is_human:
                    data[f"stockfish_difference_depth_{depth}"].append(data[f"stockfish_score_depth_{depth}"][-1]+data[f"stockfish_score_depth_{depth}"][-3])
                else:
                    data[f"stockfish_difference_depth_{depth}"].append(data[f"stockfish_score_depth_{depth}"][-1]+data[f"stockfish_score_depth_{depth}"][-2])
            else:
                data[f"stockfish_difference_depth_{depth}"].append(data[f"stockfish_score_depth_{depth}"][-1])

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

def get_initial_boards(columns_data):
    '''
    Generates initial file paths and a file path queue.

    Parameters:
    - columns_data (dict): Dictionary defining the columns of the DataFrame.

    Returns:
    - board_queue (deque): Queue containing board games.
    '''
    board_queue = deque(maxlen=columns_data["num_past_moves"])
    return board_queue

def get_initial_game_state(data, board_queue, game, engine, depths, time_limits, str_functions, board_functions, columns_data):
    '''
    Initializes game state and saves initial board configurations.

    Parameters:
    - data (dict): Dictionary containing data.
    - board_queue (deque): Queue containing board games.
    - game: Chess game object.
    - engine (str): Engine used for playing the game.
    - depths (int): Depth of the search for the engine.
    - time_limits (dict): Time limits for the game.
    - depths (list) - list of depth of the engine
    - time_limits (list) - list of times for engine
    - str_functions (list) - functions to use to create bitboards based on str representation
    - board_functions (list) - functions to use to create bitboards based on board representations
    - columns_data (list) - columns used for the creation of dataframe

    Returns:
    - data (dict): Updated dictionary containing data.
    - board: Chess board object.
    - white_player (str): Name of the white player.
    - white_elo (str): Elo rating of the white player.
    - black_player (str): Name of the black player.
    - black_elo (str): Elo rating of the black player.
    - board_queue (deque): Queue containing board games.
    '''
    white_player = game.headers["White"]
    black_player = game.headers["Black"]
    white_elo = game.headers["WhiteElo"]
    black_elo = game.headers["BlackElo"]
    board = game.board()
    set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human=True)
    str_board = board.fen()
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    empty_board = np.zeros_like(new_board)
    for _ in range(columns_data["num_past_moves"]):
        board_queue.append(empty_board)
    data["current_move"].append(new_board)
    data["current_move_str"].append(str_board)
    data["event"].append(game.headers["Event"])
    clock = 0.0 #Adding 0.0 in case the game is bugged and there is no clock
    for node in game.mainline():
        clock = node.clock()
        break
    data["clock"].append(clock)
    data["player"].append(white_player)
    data["color"].append("Starting Move")
    data["elo"].append(white_elo)
    data["human"].append(True)
    x=1
    for temp_board in board_queue:
        data[f"past_move_{x}"].append(temp_board)
        x+=1
    board_queue.append(new_board)
    return data, board, white_player, white_elo, black_player, black_elo, board_queue

def get_save_random_move(board, node, str_functions, board_functions, board_queue, data, player_color, engine, depths, time_limits, game, white_elo, black_elo):
    '''
    Performs and saves a random move made by the bot.

    Parameters:
    - board: Chess board object.
    - node: Chess node object.
    - game_number (int): Number of the game.
    - directory (str): Directory path.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - board_queue (deque): Queue containing board games.
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
    for temp_board in board_queue:
        data[f"past_move_{x}"].append(temp_board)
        x+=1
    bot_new_board = get_bitboards(bot_str_board, board, str_functions, board_functions)
    data["current_move"].append(bot_new_board)
    data["current_move_str"].append(bot_str_board)

    data["human"].append(False)
    data["event"].append(game.headers["Event"])
    data["clock"].append(bot_clock)
    if player_color == chess.WHITE:
        data["player"].append("bot")
        data["color"].append("White")
        data["elo"].append(white_elo)
        set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human = False)
    else:
        data["player"].append("bot")
        data["color"].append("Black")
        data["elo"].append(black_elo)
        set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data, is_human = False)
    board.pop()
    return data, board_queue, board 

def get_save_human_move(board, node, str_functions, board_functions, board_qeue, data, player_color, engine, depths, time_limits, game, white_player, white_elo, black_player, black_elo):
    '''
    Performs and saves a human move.

    Parameters:
    - board: Chess board object.
    - node: Chess node object.
    - str_functions: A list of functions that convert string notation to bitboards.
    - board_functions: A list of functions that generate bitboards from chess.Board objects.
    - file_path (str): File path for the game data.
    - j (int): Move number.
    - board_queue (deque): Queue containing board games.
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
    for temp_board in board_qeue:
        data[f"past_move_{x}"].append(temp_board)
        x+=1
    new_board = get_bitboards(str_board, board, str_functions, board_functions)
    data["current_move"].append(new_board)
    data["current_move_str"].append(str_board)
    board_qeue.append(new_board)

    data["human"].append(True)
    data["event"].append(game.headers["Event"])
    data["clock"].append(clock)
    if player_color == chess.WHITE:
        data["player"].append(white_player)
        data["color"].append("White")
        data["elo"].append(white_elo)
        set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human = True)
    else:
        data["player"].append(black_player)
        data["color"].append("Black")
        data["elo"].append(black_elo)
        set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data, is_human = True)
    return data, board_qeue, board

def save_game_data(columns, engine, depths, time_limits, game_number, game, str_functions, board_functions,
                    directory = "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe", columns_data = {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True, "current_move_str": True},
                    shuffle = True, seed = 42, batch_size = 1000, all_games_df = None, max_games = np.inf):
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
    - directory (str, optional): Directory path where the data will be saved. Defaults to "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe".
    - columns_data (dict, optional): Dictionary defining the columns of the DataFrame. Defaults to {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True}.
    - shuffle (bool, optional): If the datframe should be shuffled before being returned, default True
    - seed (int, optional): seed for shuffling, default 42
    '''

    data = {}
    data = initialize_data(data, depths, columns_data)
    j=0
    board_queue = get_initial_boards(columns_data)
    data, board, white_player, white_elo, black_player, black_elo, board_queue = get_initial_game_state(data, board_queue, game, engine,
                                                                                                        depths, time_limits, str_functions, board_functions, columns_data)
    for node in game.mainline():
        j+=1
        player_color = board.turn
        data, board_queue, board = get_save_random_move(board, node, str_functions, board_functions, board_queue, data, player_color, engine, depths, time_limits, game, white_elo, black_elo)
        data, board_queue, board = get_save_human_move(board, node, str_functions, board_functions, board_queue, data, player_color, engine, depths, time_limits, game, white_player, white_elo, black_player, black_elo)
    df = pd.DataFrame(data)
    if all_games_df.empty:
        all_games_df = df.copy()
    else:
        all_games_df = pd.concat([all_games_df, df], axis=0)
    all_games_df.reset_index(drop=True, inplace=True)
    if game_number+1 == max_games or (game_number>0 and (game_number+1)%batch_size == 0):
        if shuffle:
            all_games_df = all_games_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        game_filename = f"{directory}/game_batch_{game_number//batch_size}.npy"
        np.save(game_filename, all_games_df.to_numpy())
        compress_file(game_filename)
        all_games_df = None
        all_games_df = pd.DataFrame(columns=columns)
    return all_games_df

def save_column_names(file_path, columns):
    with open(file_path, 'w') as file:
        file.write('\n'.join(columns))

def save_data(txt_file_dir, directory_path, file_name, verbose = True, str_functions = [str_to_board_all_figures_colors], board_functions = [get_all_attacks], 
              stockfish_path = "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe", depths = [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20], 
              time_limits = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], max_num_games = np.inf,
              columns_data = {"human": True, "player": True, "elo": True, "color": True, "event": True, "clock": True, "depths": True, "num_past_moves": 12, "current_move": True, "current_move_str": True},
              shuffle = True, seed = 42, batch_size = 1000):
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
    - stockfish_path (str, optional): The path to the Stockfish chess engine executable. Defaults to "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe".
    - depths (list, optional): List of depths for Stockfish engine analysis. Defaults to [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20].
    - time_limits (list, optional): List of time limits for Stockfish engine analysis. Defaults to [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]. Should be the same length as depths list.
    - max_num_games (int, optional): Number of games to read from the PGN file, by default np.infinite, i.e. all games will be read from the file.
    - shuffle (bool, optional): If the datframe should be shuffled before being returned, default True
    - seed (int, optional): seed for shuffling, default 42
    '''
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    file_path = os.path.join(directory_path, file_name)
    columns = ['human', 'player', 'elo', 'color', 'event', 'clock',
       'stockfish_score_depth_1', 'stockfish_difference_depth_1',
       'stockfish_score_depth_2', 'stockfish_difference_depth_2',
       'stockfish_score_depth_3', 'stockfish_difference_depth_3',
       'stockfish_score_depth_4', 'stockfish_difference_depth_4',
       'stockfish_score_depth_5',  'stockfish_difference_depth_5',
       'stockfish_score_depth_8', 'stockfish_difference_depth_8',
       'stockfish_score_depth_10', 'stockfish_difference_depth_10',
       'stockfish_score_depth_12', 'stockfish_difference_depth_12',
       'stockfish_score_depth_15', 'stockfish_difference_depth_15',
       'stockfish_score_depth_16', 'stockfish_difference_depth_16',
       'stockfish_score_depth_18', 'stockfish_difference_depth_18',
       'stockfish_score_depth_20', 'stockfish_difference_depth_20',
       'past_move_1', 'past_move_2', 'past_move_3', 'past_move_4', 'past_move_5',
       'past_move_6', 'past_move_7', 'past_move_8', 'past_move_9',
       'past_move_10', 'past_move_11', 'past_move_12', 'current_move',
       'current_move_str']
    all_games_df = pd.DataFrame(columns=columns)
    save_column_names(txt_file_dir + "/column_names.txt", columns=columns)
    done = False
    i = 0
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            while not done:
                chunk = decompressed_file.read(1024**3) #Read one GB at a time
                if not chunk:
                    print("No more Chunks")
                    break
                pgn_text = chunk.decode("utf-8")
                pgn_io = io.StringIO(pgn_text)
                while True:
                    pgn_game = chess.pgn.read_game(pgn_io)
                    if i >= max_num_games:
                        done = True
                        break
                    elif pgn_game is None:
                        print("Chunk Done!")
                        break
                    try: #Checking if some game was only partially saved
                        temp_var = pgn_game.headers["WhiteElo"]
                        temp_var = pgn_game.headers["BlackElo"]
                    except KeyError:
                        continue
                    all_games_df = save_game_data(columns, engine, depths, time_limits, i, pgn_game, str_functions, board_functions, directory = txt_file_dir, 
                                                columns_data = columns_data, shuffle = shuffle, seed = seed, batch_size = batch_size, max_games=max_num_games, 
                                                all_games_df=all_games_df)
                    i+=1
    if verbose:
        print(f"Num processed games in a file = {i}")

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
                    data[f"stockfish_difference_depth_{depth}"] = []
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
        set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human = False)
    else:
        data["player"].append("bot")
        data["color"].append("Black")
        data["elo"].append(black_elo)
        set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data, is_human = False)
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
        set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human = True)
    else:
        data["player"].append(black_player)
        data["color"].append("Black")
        data["elo"].append(black_elo)
        set_scores(board, engine, depths, time_limits, color=chess.BLACK, mate_score=900, data=data, is_human = True)
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
    set_scores(board, engine, depths, time_limits, color=chess.WHITE, mate_score=900, data=data, is_human = True)
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
                stockfish_path = "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe", depths = [1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 18, 20], 
                time_limits = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], starting_game = 0, num_games = np.inf,
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
    - stockfish_path (str, optional): Path to the Stockfish engine executable. Default is "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe".
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

def read(data_file, column_names_file):
    '''
    Function to read all the data from an npy file.
    Requires the file to be saved using save_data() function.
    Parameters:
     - data_file (str) - path to file to read the saved data, requires an .npy file compressed to .gz
     - column_names_file - path to file with columns in a form of .txt file
    '''
    with gzip.open(data_file, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    with open(column_names_file, 'r') as file:
        column_names = file.read().splitlines()
    df = pd.DataFrame(data, columns=column_names)
    return df