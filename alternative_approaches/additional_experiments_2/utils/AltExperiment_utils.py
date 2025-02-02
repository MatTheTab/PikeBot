import io
import pandas as pd
import chess.pgn
import zstandard
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
import os
import ast
from utils.data_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.model_utils import count_parameters

def add_elo_and_color_to_bitboard(bit_board, elo, color):
    elo_normalized = int(elo) / 3000
    elo_board = np.full((8, 8), elo_normalized)
    color_board = np.full((8, 8), int(color))
    original_size = np.full((8, 8), 1)
    bit_board_with_elo = np.concatenate([bit_board, elo_board[np.newaxis, :, :]], axis=0)
    bitboard_with_elo_color = np.concatenate([bit_board_with_elo, color_board[np.newaxis, :, :]], axis=0)
    final_bitboard = np.concatenate([bitboard_with_elo_color, original_size[np.newaxis, :, :]], axis=0) 
    return final_bitboard

def read_game_alternative(game, all_games_df, acceptable_moves_to_finish):
    board = chess.Board()
    str_functions = [str_to_board_all_figures_colors]
    board_functions = [get_all_attacks]

    white_player = game.headers.get("White", "Unknown")
    black_player = game.headers.get("Black", "Unknown")
    white_elo = game.headers.get("WhiteElo", "Unknown")
    black_elo = game.headers.get("BlackElo", "Unknown")
    event_name = game.headers.get("Event", "Unknown")
    result = game.headers.get("Result", "*")
    
    if result == "1-0":
        winner = 1
    elif result == "0-1":
        winner = -1
    else:
        winner = 0
    
    board = chess.Board()
    all_moves = list(game.mainline_moves())
    total_moves = len(all_moves)
    
    move_count = 1
    for move in all_moves:
        # Determine whose turn it is
        current_player = white_player if board.turn else black_player
        current_elo = white_elo if board.turn else black_elo
        current_color_code = 1 if board.turn else -1

        board.push(move)  # Apply the move
        moves_left = total_moves - move_count  # Calculate remaining moves
        if moves_left <= acceptable_moves_to_finish:
            # Get the current position in bitboard format
            str_board = board.fen()
            bit_board = get_bitboards(str_board, board, str_functions, board_functions)
            bit_board = add_elo_and_color_to_bitboard(bit_board.copy(), current_elo, current_color_code)
            
            # Add values to the dictionary
            all_games_df['current_position'].append(bit_board)
            all_games_df['moves_left'].append(moves_left)
            all_games_df['winner'].append(winner)

        move_count += 1

    return all_games_df

def save_games_df(game_number, all_games_df, moves_batch, output_dir, verbose):
    os.makedirs(output_dir, exist_ok=True)
    shuffled_df = all_games_df.sample(frac=1).reset_index(drop=True)
    total_moves = len(shuffled_df)
    num_batches = (total_moves + moves_batch - 1) // moves_batch
    
    if verbose:
        print(f"Total moves: {total_moves}, Batches: {num_batches}, Moves per batch: {moves_batch}")
    
    for i in range(num_batches):
        start_idx = i * moves_batch
        end_idx = min(start_idx + moves_batch, total_moves)
        batch_df = shuffled_df.iloc[start_idx:end_idx]

        batch_filename = os.path.join(output_dir, f"Data_{game_number}_batch_{i + 1}")
        np.savez_compressed(batch_filename, data=batch_df.values)

def read_compressed_to_dataframe(data_file_path, column_file_path):
    with np.load(data_file_path, allow_pickle=True) as data:
        values = data['data']
    with open(column_file_path, 'r') as f:
        columns = f.read().splitlines()
    return pd.DataFrame(values, columns=columns)

def check_valid(game):
    white_elo = game.headers.get("WhiteElo", "Unknown")
    black_elo = game.headers.get("BlackElo", "Unknown")
    result = game.headers.get("Result", "*")

    if (not str(white_elo).isnumeric()) or (not str(black_elo).isnumeric()) or ("-" not in str(result)):
        return False
    return True

def process_all_games_alternative(data_file_path, total_number_games, txt_file_dir, games_in_batch, moves_batch, output_dir, acceptable_moves_to_finish, only_classical, verbose):
    columns = ['current_position', 'moves_left', 'winner']
    all_games_df = {col: [] for col in columns}
    os.makedirs(txt_file_dir, exist_ok=True)
    save_column_names(txt_file_dir + "/column_names.txt", columns=columns)
    done = False
    i = 0
    should_save = True
    with open(data_file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            while not done:
                chunk = decompressed_file.read(5 * (1024 ** 3)) #Read 5 GB at a time
                if not chunk:
                    print("No more Chunks")
                    break
                pgn_text = chunk.decode("utf-8")
                pgn_io = io.StringIO(pgn_text)
                while True:
                    pgn_game = chess.pgn.read_game(pgn_io)
                    if should_save and i > 0 and  (i%games_in_batch) == 0:
                        should_save = False
                        all_games_df = pd.DataFrame(all_games_df)
                        save_games_df(i, all_games_df, moves_batch, output_dir, verbose=verbose)
                        del all_games_df
                        all_games_df = {col: [] for col in columns}
                    if i >= total_number_games:
                        done = True
                        break
                    elif pgn_game is None:
                        print("Chunk Done!")
                        break
                    event = pgn_game.headers.get("Event", "Unknown")
                    if only_classical and "classical" not in event and "Classical" not in event:
                        continue
                    try: #Checking if some game was only partially saved
                        temp_var = pgn_game.headers["WhiteElo"]
                        temp_var = pgn_game.headers["BlackElo"]
                    except KeyError:
                        continue
                    if not check_valid(pgn_game):
                        continue
                    all_games_df = read_game_alternative(pgn_game, all_games_df, acceptable_moves_to_finish)
                    i += 1
                    should_save = True
    for key in all_games_df.keys():
        any_key = key
        break

    if len(all_games_df[any_key]) > 0:
        all_games_df = pd.DataFrame(all_games_df)
        save_games_df(i, all_games_df)

    if verbose:
        print(f"Num processed games in a file = {i}")

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        # Padding is set to (kernel_size // 2) to maintain spatial dimensions.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + identity  # Skip connection
        return out

class AltChessModel(nn.Module):
    def __init__(self):
        super(AltChessModel, self).__init__()

        # The input shape is specified as (79, 8, 8). We'll handle the input channels appropriately.
        self.initial_conv = nn.Conv2d(79, 256, kernel_size=1, stride=1)  # Project input to 256 channels.
        self.initial_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=False)

        # Three ResNet blocks with padding to maintain spatial dimensions.
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256)
        )

        # First head - predicts number of moves left in the game
        self.head1_conv = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        self.head1_bn = nn.BatchNorm2d(2)
        self.head1_fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 8),  # Flatten output to match dimensions.
            nn.ReLU(inplace=False),
            nn.Linear(8, 1),
            nn.ReLU(inplace=False)
        )

        # Second head - predicts victory or loss for the players (1 - White victory, 0 - Draw, -1 - Black Victory)
        self.head2_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.head2_bn = nn.BatchNorm2d(1)
        self.head2_fc = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),  # Flatten output to match dimensions.
            nn.ReLU(inplace=False),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Project input to 256 channels while maintaining spatial dimensions.
        x = self.relu(self.initial_bn(self.initial_conv(x)))

        # Pass through ResNet blocks.
        x = self.resnet_blocks(x)

        # First head output
        head1 = self.head1_conv(x)
        head1 = self.head1_bn(head1)
        head1 = self.head1_fc(head1)

        # Second head output
        head2 = self.head2_conv(x)
        head2 = self.head2_bn(head2)
        head2 = self.head2_fc(head2)

        return head1, head2
    
def compute_accuracy_alt(model_output, target):
    model_output_np = model_output.detach().numpy()
    target_np = target.detach().numpy()
    predicted = np.where(model_output_np < -0.5, -1, 
                 np.where(model_output_np > 0.5, 1, 0))
    correct_predictions = (predicted == target_np)
    accuracy = np.mean(correct_predictions) * 100  # as a percentage
    return accuracy

def find_npz_files(directory):
    npz_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return npz_files


def process_all_move_ranges(acceptable_moves_ranges, data_file_path, total_number_games, txt_file_dir, games_in_batch, moves_batch, results_dir, only_classical, verbose=1):
    os.makedirs(results_dir, exist_ok=True)
    for acceptable_moves_to_finish in acceptable_moves_ranges:
        output_dir = results_dir + f"/Range_{acceptable_moves_to_finish}"
        process_all_games_alternative(data_file_path, total_number_games, txt_file_dir, games_in_batch, moves_batch, output_dir, acceptable_moves_to_finish, only_classical, verbose)
        if verbose:
            print(f"Processed Games with move range: {acceptable_moves_to_finish}")

def train_alt_chess_model_single_pass(alt_model, alt_optimizer, train_data_source, val_data_source, test_data_source, alt_criterion_moves, alt_criterion_victory, alt_num_epochs, train_log_dir, acceptable_moves_to_finish, column_file_location, verbose):
    all_metrics = {
        "train" : [],
        "val" : [],
        "test" : []
    }

    for epoch in range(alt_num_epochs):
        alt_model, alt_optimizer, train_metrics = train_alt_epoch(alt_model, alt_optimizer, train_data_source, alt_criterion_moves, alt_criterion_victory, epoch, column_file_location, verbose)
        all_metrics["train"].append(train_metrics)
        val_metrics = eval_alt_epoch(epoch, "Validation", alt_model, val_data_source, alt_criterion_moves, alt_criterion_victory, column_file_location, verbose)
        all_metrics["val"].append(val_metrics)
    test_metrics = eval_alt_epoch(0, "Test", alt_model, test_data_source, alt_criterion_moves, alt_criterion_victory, column_file_location, verbose)
    all_metrics["test"].append(test_metrics)
    log_progress(all_metrics, train_log_dir, acceptable_moves_to_finish)
    return alt_model

def log_progress(all_metrics, train_log_dir, acceptable_moves_to_finish):
    log_file = os.path.join(train_log_dir, "training_log.txt")
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a' if file_exists else 'w') as f:
        for epoch in range(len(all_metrics["train"])):
            train_metrics = all_metrics["train"][epoch]
            f.write(f"Log for {acceptable_moves_to_finish} allowed moves")
            f.write(f"Train Epoch {epoch + 1}:\n")
            for metric_name, value in train_metrics.items():
                f.write(f"{metric_name}: {value}\n")
            val_metrics = all_metrics["val"][epoch]
            f.write(f"Val Epoch {epoch + 1}:\n")
            for metric_name, value in val_metrics.items():
                f.write(f"{metric_name}: {value}\n")
        test_metrics = all_metrics["test"][-1]
        f.write(f"Test Epoch {epoch + 1}:\n")
        for metric_name, value in test_metrics.items():
            f.write(f"{metric_name}: {value}\n")

def train_alt_epoch(alt_model, alt_optimizer, train_data_source, alt_criterion_moves, alt_criterion_victory, epoch, column_file_location, verbose):
    alt_model.train()
    all_train_files = find_npz_files(train_data_source)
    total_accuracy = 0
    total_loss_moves = 0
    total_loss_victory = 0
    total_num_examples = 0
    total_total_loss = 0
    total_mae_moves = 0

    for train_file in all_train_files:
        alt_optimizer.zero_grad()
        train_batch = read_compressed_to_dataframe(data_file_path = train_file, column_file_path = column_file_location)
        train_batch.sample(frac=1).reset_index(drop=True)
        model_input_vals = train_batch["current_position"].values
        model_input = torch.tensor(np.stack(model_input_vals, axis=0), dtype=torch.float32)
        y_moves = torch.tensor(np.array(train_batch["moves_left"].values,dtype = np.float32))
        y_victory = torch.tensor(np.array(train_batch["winner"].values,dtype = np.float32))
        output_moves_left, outputs_victory_prob = alt_model(model_input)
        
        output_moves_left = output_moves_left.squeeze(-1)
        outputs_victory_prob = outputs_victory_prob.squeeze(-1)

        loss_moves = alt_criterion_moves(output_moves_left, y_moves)
        loss_victory = alt_criterion_victory(outputs_victory_prob, y_victory)
        total_loss = (0.1*loss_moves) + (0.9*loss_victory)
        loss_victory.backward()
        alt_optimizer.step()

        mae_moves = torch.mean(torch.abs(output_moves_left - y_moves))
        acc = compute_accuracy_alt(outputs_victory_prob, y_victory)
        total_num_examples += y_victory.shape[0]
        total_loss_victory += loss_victory.item()
        total_loss_moves += loss_moves.item()
        total_total_loss += total_loss.item()
        total_accuracy += acc
        total_mae_moves += mae_moves.item()
    
    metrics = {
        "moves loss": total_loss_moves / total_num_examples,
        "victory loss": total_loss_victory / total_num_examples,
        "total loss": total_total_loss / total_num_examples,
        "accuracy": total_accuracy / total_num_examples,
        "mae moves": total_mae_moves / total_num_examples
    }
    
    if verbose:
        print(f"Train Epoch {epoch+1}, Loss (Moves): {total_loss_moves/total_num_examples:.4f}, "
              f"Loss (Victory): {total_loss_victory/total_num_examples:.4f}, "
              f"Total Loss: {total_total_loss/total_num_examples:.4f}, "
              f"Accuracy (Victory): {total_accuracy/total_num_examples:.4f}, "
              f"MAE (Moves): {total_mae_moves/total_num_examples:.4f}")
        
    return alt_model, alt_optimizer, metrics

def eval_alt_epoch(epoch, eval_type, alt_model, test_data_source, alt_criterion_moves, alt_criterion_victory, column_file_location, verbose):
    alt_model.eval()
    all_test_files = find_npz_files(test_data_source)
    
    total_accuracy = 0
    total_loss_moves = 0
    total_loss_victory = 0
    total_num_examples = 0
    total_total_loss = 0
    total_mae_moves = 0
    
    with torch.no_grad():  # Disable gradient computations
        for test_file in all_test_files:
            test_batch = read_compressed_to_dataframe(data_file_path=test_file, column_file_path=column_file_location)
            model_input_vals = test_batch["current_position"].values
            model_input = torch.tensor(np.stack(model_input_vals, axis=0), dtype=torch.float32)
            y_moves = torch.tensor(np.array(test_batch["moves_left"].values,dtype = np.float32))
            y_victory = torch.tensor(np.array(test_batch["winner"].values,dtype = np.float32))
            
            output_moves_left, outputs_victory_prob = alt_model(model_input)
            output_moves_left = output_moves_left.squeeze(-1)
            outputs_victory_prob = outputs_victory_prob.squeeze(-1)

            loss_moves = alt_criterion_moves(output_moves_left, y_moves)
            loss_victory = alt_criterion_victory(outputs_victory_prob, y_victory)
            total_loss = (0.2 * loss_moves) + (0.8 * loss_victory)
            
            mae_moves = torch.mean(torch.abs(output_moves_left - y_moves))
            acc = compute_accuracy_alt(outputs_victory_prob, y_victory)
            total_num_examples += y_victory.shape[0]
            total_loss_victory += loss_victory.item()
            total_loss_moves += loss_moves.item()
            total_total_loss += total_loss.item()
            total_accuracy += acc
            total_mae_moves += mae_moves.item()
    
    # Aggregate metrics
    metrics = {
        "moves loss": total_loss_moves / total_num_examples,
        "victory loss": total_loss_victory / total_num_examples,
        "total loss": total_total_loss / total_num_examples,
        "accuracy": total_accuracy / total_num_examples,
        "mae moves": total_mae_moves / total_num_examples
    }
    
    if verbose:
        print(f"Epoch {epoch+1}, {eval_type}, Loss (Moves): {total_loss_moves/total_num_examples:.4f}, "
              f"Loss (Victory): {total_loss_victory/total_num_examples:.4f}, "
              f"Total Loss: {total_total_loss/total_num_examples:.4f}, "
              f"Accuracy (Victory): {total_accuracy/total_num_examples:.4f}, "
              f"MAE (Moves): {total_mae_moves/total_num_examples:.4f}")
    
    return metrics


def train_alt_chess_model(acceptable_moves_ranges, model_dir, model_lr, train_dir, val_dir, test_dir, model_class, train_log_dir, num_epochs, column_file_location, verbose = 1):
    model = model_class()
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "AltModel.pth")
    torch.save(model, model_path)
    del model
    if verbose:
        print("Saved initial model successfuly")

    for acceptable_moves_to_finish in acceptable_moves_ranges:
        if verbose:
            print(f"Processing Moves: {acceptable_moves_to_finish}")
        train_data_source = os.path.join(train_dir, f"Range_{acceptable_moves_to_finish}/")
        val_data_source = os.path.join(val_dir, f"Range_{acceptable_moves_to_finish}/")
        test_data_source = os.path.join(test_dir, f"Range_{acceptable_moves_to_finish}/")
        model = torch.load(model_path, weights_only=False)
        model.train()
        criterion_moves = nn.MSELoss()
        criterion_victory = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = model_lr)
        model = train_alt_chess_model_single_pass(model, optimizer, train_data_source, val_data_source, test_data_source, criterion_moves, criterion_victory, num_epochs, train_log_dir, acceptable_moves_to_finish, column_file_location, verbose)
        torch.save(model, model_path)
        del model
    
    print("Finished !")