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


class MirroredLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(MirroredLayer, self).__init__()
        assert input_size % 2 == 0, "Input size must be even for mirroring."
        self.input_size = input_size
        self.output_size = output_size
        half_input_size = input_size // 2
        self.white_weights = nn.Parameter(torch.empty(output_size, half_input_size, dtype=torch.float32))
        nn.init.xavier_uniform_(self.white_weights)
        self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))

    def forward(self, x):
        mid_point = x.size(1) // 2
        x_white, x_black = x[:, :mid_point], x[:, mid_point:]
        black_weights = torch.flip(self.white_weights, [0, 1])
        out_white = F.linear(x_white, self.white_weights, self.bias)
        out_black = F.linear(x_black, black_weights, self.bias)      
        return out_white, out_black

    @property
    def weights(self):
        return torch.cat([self.white_weights, torch.flip(self.white_weights, [0, 1])], dim=1)
    
    def quantize_weights(self):
        with torch.no_grad():
            self.white_weights = nn.Parameter((self.white_weights.data.clamp(-32768, 32767)).to(torch.int16), requires_grad=False)
            self.bias = nn.Parameter((self.bias.data.clamp(-32768, 32767)).to(torch.int16), requires_grad=False)
        

class NNUEModel(nn.Module):
    def __init__(self, input_size, output_activation="sigmoid"):
        super(NNUEModel, self).__init__()
        
        assert input_size % 2 == 0, "Input size must be even."
        self.mirrored_layer = MirroredLayer(input_size, 256)
        self.fc_shared_1 = nn.Linear(512, 32)
        self.fc_shared_2 = nn.Linear(32, 32)
        self.fc_output = nn.Linear(32, 1)
        output_activation = output_activation.lower()
        if output_activation == "leakyrelu":
            self.output_activation = torch.nn.LeakyReLU()
        elif output_activation == "sigmoid":
            self.output_activation = torch.nn.Sigmoid()
        elif output_activation == "linear":
            self.output_activation = None
        else:
            raise ValueError("Invalid output activation type. Choose from 'leakyrelu', 'sigmoid', or 'linear'.")

    def forward(self, x):
        out_white, out_black = self.mirrored_layer(x)
        out = torch.cat((out_white, out_black), dim=1)
        out = F.leaky_relu(self.fc_shared_1(out))
        out = F.leaky_relu(self.fc_shared_2(out))
        if self.output_activation is not None:
            out = self.output_activation(self.fc_output(out))
        return out
    
    def quantize_weights(self):
        """Disable gradients and quantize all relevant layer weights for inference."""
        for param in self.parameters():
            param.requires_grad = False
        
        self.mirrored_layer.quantize_weights()
        with torch.no_grad():
            self.fc_shared_1.weight = nn.Parameter((self.fc_shared_1.weight.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
            self.fc_shared_1.bias = nn.Parameter((self.fc_shared_1.bias.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
            self.fc_shared_2.weight = nn.Parameter((self.fc_shared_2.weight.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
            self.fc_shared_2.bias = nn.Parameter((self.fc_shared_2.bias.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
            self.fc_output.weight = nn.Parameter((self.fc_output.weight.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
            self.fc_output.bias = nn.Parameter((self.fc_output.bias.data.clamp(-128, 127)).to(torch.int8), requires_grad=False)
    
    def check_mirrored_layer(self):
        """Checks if weights are mirrored"""
        mirrored_weights = list(self.mirrored_layer.weights.data.numpy())
        for i in range(len(mirrored_weights)):
            weight_matrix_1 = mirrored_weights[i]
            weight_matrix_2 = mirrored_weights[len(mirrored_weights)-1-i]
            for ii in range(len(weight_matrix_1)):
                val_1 = weight_matrix_1[ii]
                val_2 = weight_matrix_2[len(weight_matrix_2) - 1 - ii]
                assert val_1 == val_2
        print("Checked successfully, weights match!")

    def predict(self, x):
        """Predicts the output, fast"""
        start_x = x.to(torch.int16)
        out_white, out_black = self.mirrored_layer(start_x)
        out = torch.cat((out_white, out_black), dim=1)
        out = out.to(torch.int8)
        out = self.fc_shared_1(out)
        out = out.to(torch.float32)
        out = F.leaky_relu(out)
        out = out.to(torch.int8)
        out = self.fc_shared_2(out)
        out = out.to(torch.float32)
        out = F.leaky_relu(out)
        out = out.to(torch.int8)
        out = self.fc_output(out)
        out = out.to(torch.float32)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

def board_to_nnue(board):
    nnue_rep = []
    pieces = ["P", "N", "B", "R", "Q", "p", "n", "b", "r", "q"]
    for king_square in chess.SQUARES:
        king_piece = str(board.piece_at(king_square))
        is_king_here = True if (king_piece != "None" and king_piece == "K") else False
        for piece_type in pieces:
            for square in chess.SQUARES:
                piece = str(board.piece_at(square))
                is_piece_match = (piece != "None") and (piece == piece_type)
                if (not is_king_here) or (not is_piece_match):
                    nnue_rep.append(0)
                else:
                    nnue_rep.append(1)
    for king_square in chess.SQUARES:
        king_piece = str(board.piece_at(king_square))
        is_king_here = True if (king_piece != "None" and king_piece == "k") else False
        for piece_type in pieces:
            for square in chess.SQUARES:
                piece = str(board.piece_at(square))
                is_piece_match = (piece != "None") and (piece == piece_type)
                if (not is_king_here) or (not is_piece_match):
                    nnue_rep.append(0)
                else:
                    nnue_rep.append(1)
    return nnue_rep

def nnue_to_board(nnue_rep):
    board = chess.Board(None)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    mid_point = len(nnue_rep) // 2
    white_nnue = nnue_rep[:mid_point]
    black_nnue = nnue_rep[mid_point:]

    index = 0
    for king_square in chess.SQUARES:
        for curr_color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                for square in chess.SQUARES:
                    if white_nnue[index] == 1:
                        board.set_piece_at(king_square, chess.Piece(chess.KING, chess.WHITE))
                    index += 1

    index = 0
    for king_square in chess.SQUARES:
        for curr_color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                for square in chess.SQUARES:
                    is_piece_here = white_nnue[index]
                    index += 1
                    if is_piece_here:
                        board.set_piece_at(square, chess.Piece(piece_type, curr_color))

    index = 0
    for king_square in chess.SQUARES:
        for curr_color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                for square in chess.SQUARES:
                    if black_nnue[index] == 1:
                        board.set_piece_at(king_square, chess.Piece(chess.KING, chess.BLACK))
                    index += 1

    index = 0
    for king_square in chess.SQUARES:
        for curr_color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                for square in chess.SQUARES:
                    is_piece_here = black_nnue[index]
                    index += 1
                    if is_piece_here:
                        board.set_piece_at(square, chess.Piece(piece_type, curr_color))
    return board

def initialize_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, data, white_player, black_player):
    if time_limit is not None:
        initial_info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
    else:
        initial_info = engine.analyse(board, chess.engine.Limit(depth=depth))
    initial_score = initial_info['score'].pov(color=chess.WHITE).score(mate_score=mate_score)
    initial_nnue = board_to_nnue(board)
    move_ratio = total_moves / (total_moves + 1)
    stockfish_ratio = initial_score * move_ratio
    relative_stockfish_ratio = stockfish_ratio / total_moves
    data.append({
        "White": white_player,
        "Black": black_player,
        "Move Number": 0,
        "Total Moves": total_moves,
        "Moves Left": total_moves,
        "Move Ratio": move_ratio,
        "Stockfish Score": initial_score,
        "Stockfish Ratio": stockfish_ratio,
        "Relative Stockfish Ratio": relative_stockfish_ratio,
        "NNUE representation": initial_nnue
    })
    return data

def update_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, move_number, data, white_player, black_player):
    if time_limit is not None:
        info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
    else:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score'].pov(color=chess.WHITE).score(mate_score=mate_score)
        nnue_rep = board_to_nnue(board)
        moves_left = total_moves - move_number
        move_ratio = total_moves / (moves_left + 1)
        stockfish_ratio = score * move_ratio
        relative_stockfish_ratio = stockfish_ratio / total_moves
        data.append({
            "White": white_player,
            "Black": black_player,
            "Move Number": move_number,
            "Total Moves": total_moves,
            "Moves Left": moves_left,
            "Move Ratio": move_ratio,
            "Stockfish Score": score,
            "Stockfish Ratio": stockfish_ratio,
            "Relative Stockfish Ratio": relative_stockfish_ratio,
            "NNUE representation": nnue_rep
        })
        return data

def shuffle_save_NNUE_batches(output_dir, df, batch_size, chunk_count, verbose, shuffle):
    os.makedirs(output_dir, exist_ok=True)
    if shuffle:
        shuffled_df = df.sample(frac=1, random_state=chunk_count).reset_index(drop=True)
    else:
        shuffled_df = df
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        new_df = shuffled_df[start_index:end_index].reset_index(drop=True)
        file_name = f"{output_dir}/chess_games_batch_{chunk_count}_batch_{i}.csv.gz"
        new_df.to_csv(file_name, index=False, compression="gzip")
        if verbose:
            print(f"Saved {len(new_df)} moves to {file_name}")

def save_interval_NNUE(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_location, col, batch_size):
    df = pd.DataFrame(data)
    df = preprocess_batch_NNUE(df, col, preprocessing_means, metadata_location, chunk_count)
    shuffle_save_NNUE_batches(output_dir, df, batch_size, chunk_count, verbose, shuffle)

def save_interval_NNUE_metadata(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_summary, col, batch_size):
    df = pd.DataFrame(data)
    df = preprocess_batch_NNUE_metadata(df, col, preprocessing_means, metadata_summary, chunk_count)
    shuffle_save_NNUE_batches(output_dir, df, batch_size, chunk_count, verbose, shuffle)

def preprocess_batch_NNUE_metadata(data, col, preprocessing_means, metadata_summary, chunk_count):
    if preprocessing_means is None:
        return data

    if "divide" in preprocessing_means.lower():
        divider = float(metadata_summary["average_divider"])
        data[col] /= divider

    elif "normalize" in preprocessing_means.lower():
        min_val = float(metadata_summary["average_min"])
        max_val = float(metadata_summary["average_max"])
        if min_val != max_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0

    elif "standardize" in preprocessing_means.lower():
        mean = float(metadata_summary["average_mean"])
        std_dev = float(metadata_summary["average_std_dev"])
        if std_dev != 0:
            data[col] = (data[col] - mean) / std_dev
        else:
            data[col] = 0
    
    else:
        raise ValueError("Unexpected Argument for preprocessing_means")
    
    return data

def preprocess_batch_NNUE(data, col, preprocessing_means, metadata_location, chunk_count):
    if preprocessing_means is None:
        return data

    metadata = []
    if chunk_count is not None:
        metadata.append(f"Batch: {chunk_count}")

    if "divide" in preprocessing_means.lower():
        divider = float(preprocessing_means.split(":")[-1])
        data[col] /= divider
        metadata.append(f"divide: {divider}")

    elif "normalize" in preprocessing_means.lower():
        min_val = data[col].min()
        max_val = data[col].max()
        if min_val != max_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0
        metadata.append(f"normalization: {min_val}, {max_val}")

    elif "standardize" in preprocessing_means.lower():
        mean = data[col].mean()
        std_dev = data[col].std()
        if std_dev != 0:
            data[col] = (data[col] - mean) / std_dev
        else:
            data[col] = 0
        metadata.append(f"standardization: {mean}, {std_dev}")
    
    else:
        raise ValueError("Unexpected Argument for preprocessing_means")

    if metadata_location:
        os.makedirs(metadata_location, exist_ok=True)
        metadata_file = os.path.join(metadata_location, "metadata.txt")
        with open(metadata_file, "a") as file:
            for entry in metadata:
                file.write(f"{entry}\n")
    
    return data

def load_single_game_from_file_NNUE(file_path):
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            pgn_text = decompressed_file.read().decode("utf-8")
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            return game
        
def load_games_from_file_NNUE(file_path, num_games = 100):
    games = []
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            pgn_text = decompressed_file.read().decode("utf-8")
            pgn_io = io.StringIO(pgn_text)
            for i in range(num_games):
                game = chess.pgn.read_game(pgn_io)
                games.append(copy.deepcopy(game))
    return games

def display_first_100_moves_NNUE(game):
    board = game.board()
    boards = [board.copy()]
    
    for move_number, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        boards.append(board.copy())
        if move_number >= 100:
            break
    
    return boards

def is_board_same(reconstructed_board, board):
    for square in chess.SQUARES:
        original_piece = str(board.piece_at(square))
        reconstructed_piece = str(reconstructed_board.piece_at(square))
        if original_piece != reconstructed_piece:
            print("Error Found")
            print(reconstructed_board)
            print()
            print(board)
            return False
    return True

def process_game_NNUE(file_path, verbose = True):
    games = load_games_from_file_NNUE(file_path)
    for game in games:
        if game is None:
            print("No game found in file.")
            return
        boards = display_first_100_moves_NNUE(game)
        for move_number, board in enumerate(boards):
            nnue_rep = board_to_nnue(board)
            if verbose:
                print(f"\nNNUE Representation for move {move_number}:\n{nnue_rep[:100]}... [truncated]\n")
                print(f"Length of NNUE: {len(nnue_rep)}")
            
            reconstructed_board = nnue_to_board(nnue_rep)

            if verbose:
                print(f"Reconstructed board from NNUE (move {move_number}):\n{reconstructed_board}\n")
            assert is_board_same(reconstructed_board, board), f"Mismatch at move {move_number}!"

def preprocess_NNUE(file_path, max_num_games, verbose=1, save_interval=10, output_dir="Data", stockfish_path = "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe", depth = 8, time_limit = None, mate_score = 900, shuffle = True,
                    preprocessing_means = None, metadata_location = "Data", col = "Stockfish Score", batch_size = 64):
    data = []
    i = 0
    chunk_count = 0
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            while True:
                print("starting new chunk")
                chunk = decompressed_file.read(1024**3)
                if not chunk:
                    if verbose:
                        print("No more chunks.")
                    break
                
                pgn_text = chunk.decode("utf-8")
                pgn_io = io.StringIO(pgn_text)
                
                while i < max_num_games:
                    game = chess.pgn.read_game(pgn_io)
                    if game is None:
                        if verbose:
                            print("Chunk done!")
                        break
                    
                    try:
                        white_player = game.headers["White"]
                        black_player = game.headers["Black"]
                    except KeyError:
                        continue
                    
                    board = game.board()
                    moves = list(game.mainline_moves())
                    total_moves = len(moves)
                    data = initialize_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, data, white_player, black_player)
                    
                    for move_number, move in enumerate(moves, start=1):
                        board.push(move)
                        data = update_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, move_number, data, white_player, black_player)
                    
                    i += 1

                    if i % save_interval == 0:
                        chunk_count += 1
                        save_interval_NNUE(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_location, col, batch_size)
                        data = []
                    
                if i >= max_num_games:
                    break
    if data:
        chunk_count += 1
        save_interval_NNUE(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_location, col, batch_size)
    
    if verbose:
        print(f"Total number of processed games: {i}")

def preprocess_NNUE_test(file_path, max_num_games, verbose=1, save_interval=10, output_dir="Data", stockfish_path = "./stockfish/stockfish_windows/stockfish-windows-x86-64-avx2.exe", depth = 8, time_limit = None, mate_score = 900, shuffle = True,
                    preprocessing_means = None, metadata_summary=None, col = "Stockfish Score", batch_size = 64):
    data = []
    i = 0
    chunk_count = 0
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            while True:
                print("starting new chunk")
                chunk = decompressed_file.read(1024**3)
                if not chunk:
                    if verbose:
                        print("No more chunks.")
                    break
                
                pgn_text = chunk.decode("utf-8")
                pgn_io = io.StringIO(pgn_text)
                
                while i < max_num_games:
                    game = chess.pgn.read_game(pgn_io)
                    if game is None:
                        if verbose:
                            print("Chunk done!")
                        break
                    
                    try:
                        white_player = game.headers["White"]
                        black_player = game.headers["Black"]
                    except KeyError:
                        continue
                    
                    board = game.board()
                    moves = list(game.mainline_moves())
                    total_moves = len(moves)
                    data = initialize_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, data, white_player, black_player)
                    
                    for move_number, move in enumerate(moves, start=1):
                        board.push(move)
                        data = update_data_NNUE(board, engine, time_limit, depth, mate_score, total_moves, move_number, data, white_player, black_player)
                    
                    i += 1

                    if i % save_interval == 0:
                        chunk_count += 1
                        save_interval_NNUE_metadata(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_summary, col, batch_size)
                        data = []
                    
                if i >= max_num_games:
                    break
    if data:
        chunk_count += 1
        save_interval_NNUE_metadata(data, output_dir, chunk_count, verbose, shuffle, preprocessing_means, metadata_summary, col, batch_size)
    
    if verbose:
        print(f"Total number of processed games: {i}")

def read_csv_files_from_directory(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv.gz'):
            file_path = os.path.join(directory, filename)
            dataframes.append(file_path)
    return dataframes

def log_training_NNUE(results_str, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "Training.txt")
    with open(log_file, "a") as file:
        file.write(results_str + "\n")

def save_model(model, model_dir, model_name, input_tensor, export_onnx=False):
    if model_dir is not None and model_name is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_name += ".pth"
        model_path = os.path.join(model_dir, model_name)
        model = model.to("cpu")
        torch.save(model.state_dict(), model_path)

        if export_onnx:
            onnx_path = os.path.join(model_dir, model_name.split('.')[0] + ".onnx")
            torch.onnx.export(model, input_tensor, onnx_path, opset_version=11)


def train_epoch_NNUE(epoch, model, optimizer, train_data, track_accuracy, x_col, y_col, device, loss_fcn, log_dir, verbose):
    total_loss = 0.0
    total_MSE = 0.0
    total_MAE = 0.0
    total_examples = 0
    model.train()
    if track_accuracy:
        total_accuracy = 0.0
    
    for train_batch_path in train_data:
        optimizer.zero_grad()
        train_batch = pd.read_csv(train_batch_path, compression="gzip")
        total_examples += len(train_batch)
        y_train = np.expand_dims(train_batch[y_col].values.astype(np.float32), 1)
        X_train = np.array(train_batch[x_col].apply(ast.literal_eval).tolist(), dtype=np.float32)
        X_train = torch.Tensor(X_train).to(device)
        y_train = torch.Tensor(y_train).to(device)
        y_preds = model(X_train)
        loss = loss_fcn(y_preds, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        numpy_y_preds = y_preds.detach().cpu().numpy()
        numpy_y_train = y_train.detach().cpu().numpy()
        total_MSE += np.sum(np.square(numpy_y_train - numpy_y_preds))
        total_MAE += np.sum(np.abs(numpy_y_train - numpy_y_preds))
        if track_accuracy:
            binary_preds = np.where(numpy_y_preds >= 0.5, 1, 0)
            binary_preds = binary_preds.astype(np.int8)
            binary_expected = numpy_y_train.astype(np.int8)
            total_accuracy += np.sum(binary_preds == binary_expected)
        results_str = f"Epoch {epoch} Train Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)}"
    if track_accuracy:
        results_str += f" | Accuracy: {round(total_accuracy/total_examples, 4)}"
    if verbose:
        print("______________________________________________________________")
        print(results_str)
    log_training_NNUE(results_str, log_dir)
    if track_accuracy:
        results = (total_loss, total_MSE, total_MAE, total_accuracy)
    else:
        results = (total_loss, total_MSE, total_MAE)
    return model, optimizer, results

def test_epoch_NNUE(epoch, model, test, track_accuracy, x_col, y_col, device, loss_fcn, log_dir, verbose, test_type):
    total_loss = 0.0
    total_MSE = 0.0
    total_MAE = 0.0
    total_examples = 0
    model.eval()
    if track_accuracy:
        total_accuracy = 0.0
    
    for test_batch_path in test:
        test_batch = pd.read_csv(test_batch_path, compression="gzip")
        total_examples += len(test_batch)
        y_test = np.expand_dims(test_batch[y_col].values.astype(np.float32), 1)
        X_test = np.array(test_batch[x_col].apply(ast.literal_eval).tolist(), dtype=np.float32)
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        y_preds = model(X_test)
        loss = loss_fcn(y_preds, y_test)
        total_loss += loss.item()
        numpy_y_preds = y_preds.detach().cpu().numpy()
        numpy_y_test = y_test.detach().cpu().numpy()
        total_MSE += np.sum(np.square(numpy_y_test - numpy_y_preds))
        total_MAE += np.sum(np.abs(numpy_y_test - numpy_y_preds))
        if track_accuracy:
            binary_preds = np.where(numpy_y_preds >= 0.5, 1, 0)
            binary_preds = binary_preds.astype(np.int8)
            binary_expected = numpy_y_test.astype(np.int8)
            total_accuracy += np.sum(binary_preds == binary_expected)
    results_str = f"Epoch {epoch} {test_type} Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)}"
    if track_accuracy:
        results_str += f" | Accuracy: {round(total_accuracy/total_examples, 4)}"
    if verbose:
        print(results_str)
    log_training_NNUE(results_str, log_dir)
    if track_accuracy:
        results = (total_loss, total_MSE, total_MAE, total_accuracy)
    else:
        results = (total_loss, total_MSE, total_MAE)
    return total_loss, results

def preproess_NNUE_onnx_representation(X_data):
    single_value = X_data.iloc[0]
    np_array = np.array(single_value).astype(np.float32)
    np_array = np.expand_dims(np_array, 0)
    return np_array

def load_input_tensor(test, x_col, y_col):
    for test_batch_path in test:
        test_batch = pd.read_csv(test_batch_path, compression="gzip")
        X_test, y_test = test_batch[x_col], test_batch[y_col]
        X_test = X_test.apply(ast.literal_eval)
        X_test = preproess_NNUE_onnx_representation(X_test)
        break
    return X_test

import os
import re

def read_metadata_NNUE(metadata_location):
    metadata_file = os.path.join(metadata_location, "metadata.txt")
    divider_values = []
    normalization_mins = []
    normalization_maxs = []
    standardization_means = []
    standardization_stds = []
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    
    with open(metadata_file, "r") as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("divide:"):
                divider = float(line.split(":")[-1])
                divider_values.append(divider)
            
            elif line.startswith("normalization:"):
                min_val, max_val = map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line))
                normalization_mins.append(min_val)
                normalization_maxs.append(max_val)
            
            elif line.startswith("standardization:"):
                mean, std_dev = map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line))
                standardization_means.append(mean)
                standardization_stds.append(std_dev)
    
    metadata_summary = {}
    if divider_values:
        metadata_summary["average_divider"] = sum(divider_values) / len(divider_values)
    if normalization_mins and normalization_maxs:
        metadata_summary["average_min"] = sum(normalization_mins) / len(normalization_mins)
        metadata_summary["average_max"] = sum(normalization_maxs) / len(normalization_maxs)
    if standardization_means and standardization_stds:
        metadata_summary["average_mean"] = sum(standardization_means) / len(standardization_means)
        metadata_summary["average_std_dev"] = sum(standardization_stds) / len(standardization_stds)
    return metadata_summary

def train_NNUE(train_dir, val_dir, test_dir, model, optimizer, loss, epochs, early_callback, early_callback_epochs, model_dir, model_name, x_col, y_col, log_dir, verbose=1, track_accuracy = True, device = "cpu"):
    train_data = read_csv_files_from_directory(train_dir)
    val = read_csv_files_from_directory(val_dir)
    test = read_csv_files_from_directory(test_dir)
    best_loss = np.inf
    epochs_no_improvement = 0
    best_model = None
    best_train_results = None
    best_val_results = None
    if verbose:
        print(f"Successfuly read {len(train_data)} datframes from the {train_dir} directory")
        print(f"Successfuly read {len(val)} datframes from the {val_dir} directory")
        print(f"Successfuly read {len(test)} datframes from the {test_dir} directory")
    for epoch in range(epochs):
        model, optimizer, train_results = train_epoch_NNUE(epoch, model, optimizer, train_data, track_accuracy, x_col, y_col, device, loss, log_dir, verbose)
        curr_loss, val_results = test_epoch_NNUE(epoch, model, test, track_accuracy, x_col, y_col, device, loss, log_dir, verbose, test_type = "Val")
        if curr_loss < best_loss:
            best_loss = curr_loss
            epochs_no_improvement = 0
            best_model = copy.deepcopy(model)
            best_train_results = train_results
            best_val_results = val_results
        else:
            epochs_no_improvement += 1
        if early_callback and epochs_no_improvement == early_callback_epochs:
            if verbose:
                print("*****************")
                print("Early Callback")
                print("*****************")
            model = copy.deepcopy(best_model)
            break
    test_loss, test_results = test_epoch_NNUE(epoch, model, test, track_accuracy, x_col, y_col, device, loss, log_dir, verbose, test_type = "Test")
    if model_dir is not None and model_name is not None:
        input_tensor = load_input_tensor(test, x_col, y_col)
        input_tensor = torch.tensor(input_tensor)
        save_model(model, model_dir, model_name, input_tensor, export_onnx=False)
    if not early_callback:
        best_train_results = train_results
        best_val_results = val_results
    return model, best_train_results, best_val_results, test_results

def NNUE_end_to_end(train_game_file_path, val_game_file_path, test_game_file_path, train_max_num_games, val_max_num_games, test_max_num_games, save_interval, 
                    verbose_training, verbose_preprocessing, stockfish_path, stockfish_depth, time_limit, mate_score, shuffle, preprocessing_means, metadata_location, x_col, y_col,
                    log_dir, output_activation, train_dir, val_dir, test_dir, loss, epochs, early_callback, early_callback_epochs, model_dir, model_name,
                    device, track_accuracy, lr, batch_size, preprocess_from_none=True):
    
    if preprocess_from_none:
        prepare_data_training(train_game_file_path, train_max_num_games, verbose_preprocessing,
                              save_interval, train_dir, stockfish_path, stockfish_depth,time_limit, mate_score, shuffle, preprocessing_means, metadata_location,
                              y_col, batch_size, val_dir, val_game_file_path, val_max_num_games, test_game_file_path, test_max_num_games, test_dir)
    
    dataframes = read_csv_files_from_directory(train_dir)
    dataframes = pd.read_csv(dataframes[0], compression="gzip")
    input_data = np.array(dataframes[x_col].apply(ast.literal_eval).iloc[0])
    model = NNUEModel(input_data.shape[0], output_activation)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if verbose_training:
        print("\n\n-------------------------")
        print("Training Starts")
        print("-------------------------\n\n")
    best_train_results, best_val_results, test_results = train_NNUE(train_dir, val_dir, test_dir, model, optimizer, loss, epochs,
                                                                    early_callback, early_callback_epochs, model_dir, model_name, x_col, y_col, log_dir,
                                                                    verbose=verbose_training, track_accuracy = track_accuracy, device = device)
    return best_train_results, best_val_results, test_results

def prepare_data_training(train_game_file_path, train_max_num_games, verbose_preprocessing, save_interval, train_dir, stockfish_path, stockfish_depth,time_limit, mate_score, shuffle, preprocessing_means, metadata_location,
                        y_col, batch_size, val_dir, val_game_file_path, val_max_num_games, test_game_file_path, test_max_num_games, test_dir):
        
        preprocess_NNUE(train_game_file_path, train_max_num_games, verbose=verbose_preprocessing, save_interval=save_interval, output_dir=train_dir, stockfish_path=stockfish_path, depth=stockfish_depth, time_limit=time_limit, mate_score=mate_score, shuffle=shuffle,
                    preprocessing_means=preprocessing_means, metadata_location=metadata_location, col=y_col, batch_size=batch_size)
        metadata_summary = read_metadata_NNUE(metadata_location=metadata_location)
        preprocess_NNUE_test(val_game_file_path, val_max_num_games, verbose=verbose_preprocessing, save_interval=save_interval, output_dir=val_dir, stockfish_path=stockfish_path, depth=stockfish_depth, time_limit=time_limit, mate_score=mate_score, shuffle=shuffle,
                    preprocessing_means=preprocessing_means, metadata_summary=metadata_summary, col=y_col, batch_size=batch_size)
        preprocess_NNUE_test(test_game_file_path, test_max_num_games, verbose=verbose_preprocessing, save_interval=save_interval, output_dir=test_dir, stockfish_path=stockfish_path, depth=stockfish_depth, time_limit=time_limit, mate_score=mate_score, shuffle=shuffle,
                    preprocessing_means=preprocessing_means, metadata_summary=metadata_summary, col=y_col, batch_size=batch_size)
        return metadata_summary

def initalize_results_dict(track_accuracy):
    results_dict = {}
    keys = ["model_name", "model_path", "preprocessing_means", "y_col", "output_activation", "loss_fcn", "train_loss", "train_MSE", "train_MAE", "val_loss", "val_MSE", "val_MAE", "test_loss", "test_MSE", "test_MAE"]
    if track_accuracy:
        keys += ["train_accuracy", "val_accuracy", "test_accuracy"]
    for key in keys:
        results_dict[key] = []
    return results_dict

def update_results_dict(all_results, track_accuracy, model_dir, model_name, preprocessing_means, y_col, output_activation, loss_fcn, best_train_results, best_val_results, test_results):
    all_results["model_name"].append(model_name)
    all_results["model_path"].append(model_dir)
    all_results["preprocessing_means"].append(preprocessing_means)
    all_results["y_col"].append(y_col)
    all_results["output_activation"].append(output_activation)
    all_results["loss_fcn"].append(loss_fcn)

    if track_accuracy:
        train_loss, train_MSE, train_MAE, train_accuracy = best_train_results
        val_loss, val_MSE, val_MAE, val_accuracy = best_val_results
        test_loss, test_MSE, test_MAE, test_accuracy = test_results
    else:
        train_loss, train_MSE, train_MAE = best_train_results
        val_loss, val_MSE, val_MAE = best_val_results
        test_loss, test_MSE, test_MAE = test_results
    
    all_results["train_loss"].append(train_loss)
    all_results["train_MSE"].append(train_MSE)
    all_results["train_MAE"].append(train_MAE)
    all_results["val_loss"].append(val_loss)
    all_results["val_MSE"].append(val_MSE)
    all_results["val_MAE"].append(val_MAE)
    all_results["test_loss"].append(test_loss)
    all_results["test_MSE"].append(test_MSE)
    all_results["test_MAE"].append(test_MAE)
    if track_accuracy:
        all_results["train_accuracy"].append(train_accuracy)
        all_results["val_accuracy"].append(val_accuracy)
        all_results["test_accuracy"].append(test_accuracy)

    return all_results


def complete_experiments(train_game_file_path, val_game_file_path, test_game_file_path, train_max_num_games, val_max_num_games, test_max_num_games, save_interval, 
                    verbose_training, verbose_preprocessing, stockfish_path, stockfish_depth, time_limit, mate_score, shuffle, metadata_location, x_col, preprocessing_means_y_cols,
                    log_dir, output_activation_list, complete_train_dir, complete_val_dir, complete_test_dir, loss_list, epochs, early_callback, early_callback_epochs, models_dir, lr, batch_size,
                    results_save_path, device):
    
    for preprocessing_means, y_col in preprocessing_means_y_cols:
        train_dir = complete_train_dir + f"/{preprocessing_means}/{y_col}/"
        train_dir = train_dir.replace(":", "_")
        val_dir = complete_val_dir + f"/{preprocessing_means}/{y_col}/"
        val_dir = val_dir.replace(":", "_")
        test_dir = complete_test_dir + f"/{preprocessing_means}/{y_col}/"
        test_dir = test_dir.replace(":", "_")
        os.makedirs(train_dir, exist_ok=True)
        prepare_data_training(train_game_file_path=train_game_file_path, train_max_num_games=train_max_num_games, verbose_preprocessing=verbose_preprocessing,
                                save_interval=save_interval, train_dir=train_dir, stockfish_path=stockfish_path, stockfish_depth=stockfish_depth, time_limit=time_limit,
                                mate_score=mate_score, shuffle=shuffle, preprocessing_means=preprocessing_means, metadata_location=metadata_location,
                                y_col=y_col, batch_size=batch_size, val_dir=val_dir, val_game_file_path=val_game_file_path, val_max_num_games=val_max_num_games,
                                test_game_file_path=test_game_file_path, test_max_num_games=test_max_num_games, test_dir=test_dir)
        
    track_accuracy = False
    all_results = initalize_results_dict(track_accuracy=track_accuracy)
    for preprocessing_means, y_col in preprocessing_means_y_cols:
        train_dir = complete_train_dir + f"./{preprocessing_means}/{y_col}/"
        val_dir = complete_val_dir + f"./{preprocessing_means}/{y_col}/"
        test_dir = complete_test_dir + f"./{preprocessing_means}/{y_col}/"
        for output_activation in output_activation_list:
            for loss_fcn in loss_list:
                print(f"Training Model: {preprocessing_means} {y_col} {output_activation} {loss_fcn}")
                model_dir = f"./{models_dir}/"
                model_name = f"Model_{preprocessing_means}_{y_col}_{output_activation}_{loss_fcn}"
                os.makedirs(model_dir, exist_ok=True)
                best_train_results, best_val_results, test_results = NNUE_end_to_end(train_game_file_path=train_game_file_path, val_game_file_path=val_game_file_path, test_game_file_path=test_game_file_path,
                                                                                    train_max_num_games=train_max_num_games, val_max_num_games=val_max_num_games, test_max_num_games=test_max_num_games, save_interval=save_interval, 
                                                                                    verbose_training=verbose_training, verbose_preprocessing=verbose_preprocessing, stockfish_path=stockfish_path, stockfish_depth=stockfish_depth,
                                                                                    time_limit=time_limit, mate_score=mate_score, shuffle=shuffle, preprocessing_means=preprocessing_means, metadata_location=metadata_location,
                                                                                    x_col=x_col, y_col=y_col, log_dir=log_dir, output_activation=output_activation, train_dir=train_dir, val_dir=val_dir,
                                                                                    test_dir=test_dir, loss=loss_fcn, epochs=epochs, early_callback=early_callback, early_callback_epochs=early_callback_epochs,
                                                                                    model_dir=model_dir, model_name=model_name, device=device, track_accuracy=track_accuracy, lr=lr, batch_size=batch_size, preprocess_from_none=False)
                all_results = update_results_dict(all_results = all_results, track_accuracy = track_accuracy, model_dir=model_dir, model_name=model_name,
                                                    preprocessing_means=preprocessing_means, y_col=y_col, output_activation=output_activation, loss_fcn=loss_fcn, 
                                                    best_train_results=best_train_results, best_val_results=best_val_results, test_results=test_results)
    all_results
    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.sort_values(by='val_loss', ascending=False)
    all_results_df.to_csv(results_save_path, index=False, compression="gzip")
    return all_results_df