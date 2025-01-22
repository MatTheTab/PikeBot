import chess.pgn
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chess import Board
import chess
import chess.pgn
import chess.engine

def load_pgn_games(pgn_path):
    """
    Loads games from a PGN file and yields each game.

    Parameters:
        pgn_path (str): Path to the PGN file.

    Yields:
        chess.pgn.Game: Each game in the PGN file.
    """
    with open(pgn_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game

def calculate_stats(game_lengths):
    mean = statistics.mean(game_lengths)
    min_value = min(game_lengths)
    max_value = max(game_lengths)
    std_dev = statistics.stdev(game_lengths)
    variance = statistics.variance(game_lengths)
    median = statistics.median(game_lengths)
    q1 = np.percentile(game_lengths, 25)  # First quartile
    q3 = np.percentile(game_lengths, 75)  # Third quartile
    range_value = max_value - min_value

    # Print results
    print(f"Mean: {mean}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Median: {median}")
    print(f"First Quartile (Q1): {q1}")
    print(f"Third Quartile (Q3): {q3}")
    print(f"Range: {range_value}")

def read_supervisor_data(game_path_supervisor, verbose, pikebot_game_lengths, stockfish_game_lengths, date_filter_func):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Supervisor Data: ")
    for game in load_pgn_games(game_path_supervisor):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        # Check if one of the players is Stockfish
        if "lichess ai level 8" not in white.lower() and "lichess ai level 8" not in black and "pikebot" not in white.lower() and "pikebot" not in black.lower():
            continue

        if game.board().fen() != standard_starting_fen:
            if verbose:
                print(f"Skipping game due to non-standard starting position: {white} vs {black}, Date: {date}")
            continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        if "lichess ai level 8" in white.lower() or "lichess ai level 8" in black:
            stockfish_game_lengths.append(num_moves)

        elif "pikebot" in white.lower() or "pikebot" in black.lower():
            pikebot_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return pikebot_game_lengths, stockfish_game_lengths

def read_stockfish_data(game_path_stockfish, stockfish_game_lengths, verbose):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Stockfish Data: ")
    for game in load_pgn_games(game_path_stockfish):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if game.board().fen() != standard_starting_fen:
            if verbose:
                print(f"Skipping game due to non-standard starting position: {white} vs {black}, Date: {date}")
            continue


        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        stockfish_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return stockfish_game_lengths

def read_pikebot_data(game_path_pikebot, pikebot_game_lengths, verbose, date_filter_func):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Pikebot Data: ")
    for game in load_pgn_games(game_path_pikebot):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if game.board().fen() != standard_starting_fen:
            if verbose:
                print(f"Skipping game due to non-standard starting position: {white} vs {black}, Date: {date}")
            continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"
        
        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        pikebot_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return pikebot_game_lengths

def get_games_from_dirs(game_path_supervisor, game_path_pikebot, game_path_stockfish, read_supervisor_data, read_stockfish_data, read_pikebot_data, date_func, verbose=1):
    stockfish_game_lengths = []
    pikebot_game_lengths = []
    pikebot_game_lengths, stockfish_game_lengths = read_supervisor_data(game_path_supervisor, verbose, pikebot_game_lengths, stockfish_game_lengths, date_func)
    stockfish_game_lengths = read_stockfish_data(game_path_stockfish, stockfish_game_lengths, verbose)
    pikebot_game_lengths = read_pikebot_data(game_path_pikebot, pikebot_game_lengths, verbose, date_func)
    return pikebot_game_lengths, stockfish_game_lengths

def plot_distributions_of_game_lengths(list1, list2, title1 = "Game Lengths of Pikebot (Heuristic 3)", title2 = "Game Lengths of Stockfish", title3 = "Comparison of Game Lengths of Pikebot vs Stockfish (Heuristic 3)"):

    # Customize these titles and axis labels as needed
    xlabel1 = "Game Lengths"
    ylabel1 = "Relative Frequency"

    xlabel2 = "Game Lengths"
    ylabel2 = "Relative Frequency"

    xlabel3 = "Game Lengths"
    ylabel3 = "Relative Frequency"
    binwidth = 10

    # Plot 1: Distribution of List 1
    plt.figure(figsize=(10, 8))
    sns.histplot(list1, color="blue", alpha=0.7, stat="probability", binwidth=binwidth)
    plt.title(title1, fontsize=18)
    plt.xlabel(xlabel1, fontsize=16)
    plt.ylabel(ylabel1, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    # Plot 2: Distribution of List 2
    plt.figure(figsize=(10, 8))
    sns.histplot(list2, color="orange", alpha=0.7, stat="probability", binwidth=binwidth)
    plt.title(title2, fontsize=18)
    plt.xlabel(xlabel2, fontsize=16)
    plt.ylabel(ylabel2, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    # Plot 3: Comparison of Distributions
    plt.figure(figsize=(10, 8))
    sns.histplot(list1, color="blue", alpha=0.5, label="Pikebot", stat="probability", binwidth=binwidth)
    sns.histplot(list2, color="orange", alpha=0.5, label="Stockfish", stat="probability", binwidth=binwidth)
    plt.title(title3, fontsize=18)
    plt.xlabel(xlabel3, fontsize=16)
    plt.ylabel(ylabel3, fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_cumulative_analysis(list1, list2, label1="stockfish", label2="Pikebot", title="Cumulative Analysis of Game Lengths"):
    # Sort the lists to compute the cumulative percentage
    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)

    # Compute the cumulative percentages
    cum_percentage1 = np.arange(1, len(sorted_list1) + 1) / len(sorted_list1) * 100
    cum_percentage2 = np.arange(1, len(sorted_list2) + 1) / len(sorted_list2) * 100

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(sorted_list1, cum_percentage1, label=label1, color="blue")
    plt.plot(sorted_list2, cum_percentage2, label=label2, color="orange")

    # Add labels, title, and legend
    plt.xlabel("Length", fontsize=14)
    plt.ylabel("Cumulative Percentage of Finished Games", fontsize=14)
    plt.title(title, fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.show()

def time_filtration_no_11_12(date):
    if "2024" in str(date) or str(date) == "2025.01.11" or str(date) == "2025.01.12":
        return False
    return True

def time_filtration_no_11(date):
    if "2024" in str(date) or str(date) == "2025.01.11":
        return False
    return True

def time_filtration_no_12(date):
    if "2024" in str(date) or str(date) == "2025.01.12":
        return False
    return True

def time_filtration_only_11(date):
    if str(date) == "2025.01.11":
        return True
    return False

def time_filtration_only_12(date):
    if str(date) == "2025.01.12":
        return True
    return False

def time_filtration_only_11_12(date):
    if str(date) == "2025.01.11" or str(date) == "2025.01.12":
        return True
    return False

def time_fintration_no_2024(date):
    if "2024" in str(date):
        return False
    return True

def read_supervisor_data_alt(game_path_supervisor, verbose, pikebot_game_lengths, stockfish_game_lengths, date_filter_func):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Supervisor Data: ")
    for game in load_pgn_games(game_path_supervisor):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        # Check if one of the players is Stockfish
        if "lichess ai level 8" not in white.lower() and "lichess ai level 8" not in black and "pikebot" not in white.lower() and "pikebot" not in black.lower():
            continue

        if game.board().fen() == standard_starting_fen:
            if verbose:
                print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
            continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        if "lichess ai level 8" in white.lower() or "lichess ai level 8" in black:
            stockfish_game_lengths.append(num_moves)

        elif "pikebot" in white.lower() or "pikebot" in black.lower():
            pikebot_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return pikebot_game_lengths, stockfish_game_lengths

def read_stockfish_data_alt(game_path_stockfish, stockfish_game_lengths, verbose):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Stockfish Data: ")
    for game in load_pgn_games(game_path_stockfish):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if game.board().fen() == standard_starting_fen:
            if verbose:
                print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
            continue


        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        stockfish_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return stockfish_game_lengths

def read_pikebot_data_alt(game_path_pikebot, pikebot_game_lengths, verbose, date_filter_func):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Pikebot Data: ")
    for game in load_pgn_games(game_path_pikebot):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if game.board().fen() == standard_starting_fen:
            if verbose:
                print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
            continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"
        
        last_move = game.end().board()
        if not last_move.is_checkmate():
            continue

        # Count the number of moves
        num_moves = len(list(game.mainline_moves()))
        pikebot_game_lengths.append(num_moves)

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}, Moves: {num_moves}")
    return pikebot_game_lengths

def read_supervisor_winrate(game_path_supervisor, verbose, pikebot_game_winrates, stockfish_game_winrates, date_filter_func, custom_game_flag):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Supervisor Data: ")
    for game in load_pgn_games(game_path_supervisor):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        # Check if one of the players is Stockfish
        if "lichess ai level 8" not in white.lower() and "lichess ai level 8" not in black and "pikebot" not in white.lower() and "pikebot" not in black.lower():
            continue

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        #last_move = game.end().board()
        #if not last_move.is_checkmate():
        #    continue


        if "lichess ai level 8" in white.lower() or "lichess ai level 8" in black:
            if "lichess ai level 8" in winner.lower():
                stockfish_game_winrates[0] += 1
            elif winner.lower() == "draw":
                stockfish_game_winrates[1] += 1
            else:
                stockfish_game_winrates[-1] += 1

        elif "pikebot" in white.lower() or "pikebot" in black.lower():
            if "pikebot" in winner.lower():
                pikebot_game_winrates[0] += 1
            elif winner.lower() == "draw":
                pikebot_game_winrates[1] += 1
            else:
                pikebot_game_winrates[-1] += 1

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
    return pikebot_game_winrates, stockfish_game_winrates

def read_stockfish_winrate(game_path_stockfish, stockfish_game_winrates, verbose, custom_game_flag):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Stockfish Data: ")
    for game in load_pgn_games(game_path_stockfish):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue


        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        #last_move = game.end().board()
        #if not last_move.is_checkmate():
        #    continue

        if "piekebot" in winner.lower():
            stockfish_game_winrates[0] += 1
        elif winner.lower() == "draw":
            stockfish_game_winrates[1] += 1
        else:
            stockfish_game_winrates[-1] += 1


        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
    return stockfish_game_winrates

def read_pikebot_winrate(game_path_pikebot, pikebot_winrate, verbose, date_filter_func, custom_game_flag):
    standard_starting_fen = Board().fen()
    if verbose:
        print("\n Pikebot Data: ")
    for game in load_pgn_games(game_path_pikebot):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"
        
        #last_move = game.end().board()
        #if not last_move.is_checkmate():
        #    continue

        if "pikebot" in winner.lower():
            pikebot_winrate[0] += 1
        elif winner.lower() == "draw":
            pikebot_winrate[1] += 1
        else:
            pikebot_winrate[2] += 1

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
    return pikebot_winrate

def get_winrates_from_dirs(game_path_supervisor, game_path_pikebot, game_path_stockfish, read_supervisor_winrate, read_stockfish_winrate, read_pikebot_winrate, date_func, verbose=1, custom_game_flag=False):
    stockfish_winrate = [0, 0, 0] # win, draw, loss
    pikebot_winrate = [0, 0, 0] # win, draw, loss
    pikebot_winrate, stockfish_winrate = read_supervisor_winrate(game_path_supervisor, verbose, pikebot_winrate, stockfish_winrate, date_func, custom_game_flag)
    stockfish_winrate = read_stockfish_winrate(game_path_stockfish, stockfish_winrate, verbose, custom_game_flag)
    pikebot_winrate = read_pikebot_winrate(game_path_pikebot, pikebot_winrate, verbose, date_func, custom_game_flag)
    return pikebot_winrate, stockfish_winrate

def visualize_chess_engine_winrates(pikebot_winrate, stockfish_winrate, title="Winrate Comparison", 
                                    xlabel="Results", ylabel="Percentage (%)", title_fontsize=16, 
                                    axis_fontsize=12):
    """
    Visualize relative win rates for two chess engines using a bar chart.

    Parameters:
    - pikebot_winrate: List of integers [wins, draws, losses] for Pikebot.
    - stockfish_winrate: List of integers [wins, draws, losses] for Stockfish.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title_fontsize: Font size for the title.
    - axis_fontsize: Font size for axis labels and ticks.

    """
    categories = ["Wins", "Draws", "Losses"]
    x = np.arange(len(categories))  # Create x positions for the bars
    width = 0.35  # Width of the bars

    # Calculate relative percentages
    pikebot_total = sum(pikebot_winrate)
    stockfish_total = sum(stockfish_winrate)

    pikebot_percentage = [(count / pikebot_total) * 100 for count in pikebot_winrate]
    stockfish_percentage = [(count / stockfish_total) * 100 for count in stockfish_winrate]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    bars1 = ax.bar(x - width/2, pikebot_percentage, width, label="Pikebot", color="skyblue")
    bars2 = ax.bar(x + width/2, stockfish_percentage, width, label="Stockfish", color="orange")

    # Add labels, title, and legend
    ax.set_title(title, fontsize=title_fontsize, weight="bold")
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=axis_fontsize)
    ax.legend(fontsize=axis_fontsize)

    # Annotate bars with their values
    for bar_group, percentages in zip([bars1, bars2], [pikebot_percentage, stockfish_percentage]):
        for bar, percentage in zip(bar_group, percentages):
            height = bar.get_height()
            ax.annotate(f'{percentage:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=axis_fontsize - 2)

    # Show the plot
    plt.tight_layout()
    plt.show()

def read_supervisor_scores(stockfish_path, game_path_supervisor, verbose, pikebot_scores, pikebot_opponent_scores, stockfish_scores, stockfish_opponent_scores, date_filter_func, custom_game_flag):
    standard_starting_fen = Board().fen()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    if verbose:
        print("\n Supervisor Data: ")
    for game in load_pgn_games(game_path_supervisor):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        # Check if one of the players is Stockfish
        if "lichess ai level 8" not in white.lower() and "lichess ai level 8" not in black and "pikebot" not in white.lower() and "pikebot" not in black.lower():
            continue

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        white_evals = []
        black_evals = []

        board = game.board()
        for move_number, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)  # Play the move on the board

            # Evaluate the current position
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            score = info["score"].relative

            # Convert the evaluation to centipawns or mate score
            if score.is_mate():
                evaluation = f"Mate in {score.mate()}"
            else:
                evaluation = score.score()

            # Add the evaluation to the appropriate list
            if move_number % 2 == 0:  # White's move
                white_evals.append(evaluation)
            else:  # Black's move
                black_evals.append(evaluation)

            # Print the game summary
            if verbose:
                print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
        if "lichess ai level 8" in white.lower() or "lichess ai level 8" in black.lower():
            if "lichess ai level 8" in white.lower():
                stockfish_scores.append(white_evals.copy())
                stockfish_opponent_scores.append(black_evals.copy())
            else:
                stockfish_scores.append(black_evals.copy())
                stockfish_opponent_scores.append(white_evals.copy())
        
        elif "pikebot" in white.lower() or "pikebot" in black.lower():
            if "pikebot" in white.lower():
                pikebot_scores.append(white_evals.copy())
                pikebot_opponent_scores.append(black_evals.copy())
            else:
                pikebot_scores.append(black_evals.copy())
                pikebot_opponent_scores.append(white_evals.copy())
    return pikebot_scores, pikebot_opponent_scores, stockfish_scores, stockfish_opponent_scores

def read_stockfish_scores(stockfish_path, game_path_stockfish, stockfish_scores, stockfish_opponent_scores, verbose, custom_game_flag):
    standard_starting_fen = Board().fen()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    if verbose:
        print("\n Stockfish Data: ")
    for game in load_pgn_games(game_path_stockfish):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue


        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"

        #last_move = game.end().board()
        #if not last_move.is_checkmate():
        #    continue

        white_evals = []
        black_evals = []

        board = game.board()
        for move_number, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)  # Play the move on the board

            # Evaluate the current position
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            score = info["score"].relative

            # Convert the evaluation to centipawns or mate score
            if score.is_mate():
                evaluation = f"Mate in {score.mate()}"
            else:
                evaluation = score.score()

            # Add the evaluation to the appropriate list
            if move_number % 2 == 0:  # White's move
                white_evals.append(evaluation)
            else:  # Black's move
                black_evals.append(evaluation)

        if "piekebot" in white.lower():
            stockfish_scores.append(white_evals.copy())
            stockfish_opponent_scores.append(black_evals.copy())
        else:
            stockfish_scores.append(black_evals.copy())
            stockfish_opponent_scores.append(white_evals.copy())

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
    return stockfish_scores, stockfish_opponent_scores

def read_pikebot_scores(stockfish_path, game_path_pikebot, pikebot_scores, pikebot_opponent_scores, verbose, date_filter_func, custom_game_flag):
    standard_starting_fen = Board().fen()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    if verbose:
        print("\n Pikebot Data: ")
    for game in load_pgn_games(game_path_pikebot):
        # Extract game details
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        date = game.headers.get("Date", "Unknown")
        result = game.headers.get("Result", "*")

        if custom_game_flag:
            if game.board().fen() == standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue
        else:
            if game.board().fen() != standard_starting_fen:
                if verbose:
                    print(f"Skipping game due to standard starting position: {white} vs {black}, Date: {date}")
                continue

        if not date_filter_func(date):
            continue

        # Determine the winner
        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = "Draw"
        
        #last_move = game.end().board()
        #if not last_move.is_checkmate():
        #    continue

        white_evals = []
        black_evals = []

        board = game.board()
        for move_number, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)  # Play the move on the board

            # Evaluate the current position
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            score = info["score"].relative

            # Convert the evaluation to centipawns or mate score
            if score.is_mate():
                evaluation = f"Mate in {score.mate()}"
            else:
                evaluation = score.score()

            # Add the evaluation to the appropriate list
            if move_number % 2 == 0:  # White's move
                white_evals.append(evaluation)
            else:  # Black's move
                black_evals.append(evaluation)
        
        if "pikebot" in white.lower():
            pikebot_scores.append(white_evals.copy())
            pikebot_opponent_scores.append(black_evals.copy())
        else:
            pikebot_scores.append(black_evals.copy())
            pikebot_opponent_scores.append(white_evals.copy())

        # Print the game summary
        if verbose:
            print(f"Date: {date}, {white} vs {black}, Winner: {winner}")
    return pikebot_scores, pikebot_opponent_scores

def get_scores_from_dirs(stockfish_path, game_path_supervisor, game_path_pikebot, game_path_stockfish, read_supervisor_scores, read_stockfish_scores, read_pikebot_scores, date_func, verbose=1, custom_game_flag=False):
    stockfish_scores = []
    stockfish_opponent_scores = []
    pikebot_scores = []
    pikebot_opponent_scores = []
    pikebot_scores, pikebot_opponent_scores, stockfish_scores, stockfish_opponent_scores = read_supervisor_scores(stockfish_path, game_path_supervisor, verbose, pikebot_scores, pikebot_opponent_scores, stockfish_scores, stockfish_opponent_scores, date_func, custom_game_flag)
    stockfish_scores, stockfish_opponent_scores = read_stockfish_scores(stockfish_path, game_path_stockfish, stockfish_scores, stockfish_opponent_scores, verbose, custom_game_flag)
    pikebot_scores, pikebot_opponent_scores = read_pikebot_scores(stockfish_path, game_path_pikebot, pikebot_scores, pikebot_opponent_scores, verbose, date_func, custom_game_flag)
    return pikebot_scores, pikebot_opponent_scores, stockfish_scores, stockfish_opponent_scores

def violin_plot(list1, list2, title="Violin Plot", xlabel="Values", ylabel="Distributions", title_fontsize=18, axis_fontsize=14):
    # Helper function to clean the list by filtering out non-integer values
    def clean_list(lst):
        return [x for x in lst if isinstance(x, int)]

    # Clean the input lists
    clean_list1 = []
    clean_list2 = []
    suplement_1 = [clean_list(sublist) for sublist in list1]
    suplement_2 = [clean_list(sublist) for sublist in list2]
    
    for sublst in suplement_1:
        for element in sublst:
            clean_list1.append(element)

    for sublst in suplement_2:
        for element in sublst:
            clean_list2.append(element)
    
    # Combine the cleaned lists for the plot
    combined_data = clean_list1 + clean_list2
    labels = ['Pikebot (Heuristic 3)'] * len(clean_list1) + ['Stockfish'] * len(clean_list2)

    # Set the style for seaborn to make it more aesthetic
    sns.set(style="whitegrid", palette="muted")

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=labels, y=combined_data, inner="quart", linewidth=1.25, width=0.8)

    # Add customizations to the plot's title and axis labels
    plt.title(title, fontsize=title_fontsize, weight='bold')
    plt.xlabel(xlabel, fontsize=axis_fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=axis_fontsize, weight='bold')

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.show()

def first_positive_index(list_of_lists):
    result = []
    
    # Iterate over each sublist in the list of lists
    for sublist in list_of_lists:
        found_positive = False
        
        # Check each element in the sublist, ignoring non-integer values
        for idx, item in enumerate(sublist):
            if isinstance(item, int) and item > 0:
                result.append(idx)  # Append the index of the first positive integer
                found_positive = True
                break
        
        # If no positive value was found, append "Never"
        if not found_positive:
            result.append("Never")
    
    return result