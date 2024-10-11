import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.onnx
import copy
from utils.generator_utils import *

def update_learning_rate(optimizer, curr_lr, epoch, update_epochs, learning_rate_multiplayer = 0.1, verbose = True):
    """
    Updates the learning rate of the optimizer based on the current epoch.

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be updated.
        curr_lr (float): The current learning rate before the update.
        epoch (int): The current epoch number.
        update_epochs (list): A list of epoch numbers at which the learning rate should be updated.
        learning_rate_multiplier (float) optional (default=0.1) The factor by which to multiply the current learning rate when an update is triggered.
        verbose (bool): optional, default - True

    Returns:
        float: The updated learning rate after applying the multiplier if the current epoch is in `update_epochs`.
        If the current epoch is not in `update_epochs`, the learning rate remains unchanged.
    """
    if epoch in update_epochs:
        curr_lr *= learning_rate_multiplayer
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        if verbose:
            print(f"Updated the learning rate, current learning rate: {curr_lr}")
    return curr_lr

def evaluate(loader, model, loss_func, num_hanging_values, device, verbose, log, log_file, epoch, name = "Val"):
    '''
    Evaluates the model on the provided data loader.

    Parameters:
        loader (DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): The neural network model.
        loss_func (callable): Loss function used for evaluation.
        num_hanging_values (int): Number of hanging values.
        device (str): Device to run the evaluation (e.g., "cpu" or "cuda").
        verbose (bool): Verbosity level.
        log (bool): Flag indicating whether to log evaluation progress.
        log_file (str): Path to the log file.
        epoch (int): Current epoch number.
        name (str, optional): Name of the evaluation phase (default is "Val").

    Returns:
        float: Total loss for the evaluation phase.
    '''
    model.eval()
    total_loss = 0.0
    total_MSE = 0.0
    total_MAE = 0.0
    total_accuracy = 0.0
    total_examples = 0
    with torch.no_grad():
        for X_train, y_train in loader:
            if X_train is None and y_train is None:
                break
            total_examples += len(y_train)
            hanging_vals, bitboards = separate_hanging_and_bitboards(X_train)
            hanging_vals = hanging_vals.astype(np.float32)
            hanging_vals = np.nan_to_num(hanging_vals, nan=0.0)
            hanging_vals = torch.Tensor(hanging_vals).to(device)
            bitboards = torch.Tensor(bitboards).to(device)
            y_train = y_train.astype(np.float32)
            y_train = y_train.reshape(-1, 1)
            y_train = torch.Tensor(y_train).to(device)
            y_preds = model(bitboards, hanging_vals)
            loss = loss_func(y_preds, y_train)
            total_loss += loss.item()
            numpy_y_preds = y_preds.detach().cpu().numpy()
            numpy_y_train = y_train.detach().cpu().numpy()
            total_MSE += np.sum(np.square(numpy_y_train - numpy_y_preds))
            total_MAE += np.sum(np.abs(numpy_y_train - numpy_y_preds))
            binary_preds = np.where(numpy_y_preds >= 0.5, 1, 0)
            binary_preds = binary_preds.astype(np.int8)
            binary_expected = numpy_y_train.astype(np.int8)
            total_accuracy += np.sum(binary_preds == binary_expected)
    if verbose:
        print(f"Epoch {epoch} {name} Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)}  | Accuracy: {round(total_accuracy/total_examples, 4)}")
    if log:
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} {name} Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)}  | Accuracy: {round(total_accuracy/total_examples, 4)}\n")
    return total_loss

import numpy as np

def separate_hanging_and_bitboards(X_train):
    num_columns = X_train.shape[1]
    hanging_vals = []
    bitboards = []

    for col in range(num_columns):
        first_element = X_train[0, col]
        if isinstance(first_element, (int, float)):
            col_data = X_train[:, col]
            hanging_vals.append(col_data)
        elif isinstance(first_element, np.ndarray):
            bitboards.append(np.stack(X_train[:, col]))
        else:
            raise ValueError(f"Unexpected data type in column {col}: {type(first_element)}")
        
    hanging_vals = np.column_stack(hanging_vals)
    bitboards = np.stack(bitboards, axis=1)
    batch_dim, mul, channel_dim, width, height = bitboards.shape
    bitboards = bitboards.reshape(batch_dim, mul*channel_dim, width, height)
    hanging_vals = np.nan_to_num(hanging_vals, nan=0.0)
    return hanging_vals, bitboards

def train_epoch(loader, model, optimizer, loss_func, num_hanging_values, device, verbose, log, log_file, epoch):
    '''
    Performs one epoch of training.

    Parameters:
        loader (DataLoader/Generator): DataLoader for the dataset.
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_func (callable): Loss function used for training.
        num_hanging_values (int): Number of hanging values.
        device (str): Device to run the training (e.g., "cpu" or "cuda").
        verbose (bool): Verbosity level.
        log (bool): Flag indicating whether to log training progress.
        log_file (str): Path to the log file.
        epoch (int): Current epoch number.

    Returns:
        float: Total loss for the epoch.
    '''
    model.train(True)
    total_loss = 0.0
    total_MSE = 0.0
    total_MAE = 0.0
    total_accuracy = 0.0
    total_examples = 0
    for X_train, y_train in loader:
        if X_train is None and y_train is None:
            break
        total_examples += len(y_train)
        optimizer.zero_grad()
        hanging_vals, bitboards = separate_hanging_and_bitboards(X_train)
        hanging_vals = hanging_vals.astype(np.float32)
        hanging_vals = np.nan_to_num(hanging_vals, nan=0.0)
        hanging_vals = torch.Tensor(hanging_vals).to(device)
        bitboards = torch.Tensor(bitboards).to(device)
        y_train = y_train.astype(np.float32)
        y_train = y_train.reshape(-1, 1)
        y_train = torch.Tensor(y_train).to(device)
        y_preds = model(bitboards, hanging_vals)
        loss = loss_func(y_preds, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        numpy_y_preds = y_preds.detach().cpu().numpy()
        numpy_y_train = y_train.detach().cpu().numpy()
        total_MSE += np.sum(np.square(numpy_y_train - numpy_y_preds))
        total_MAE += np.sum(np.abs(numpy_y_train - numpy_y_preds))
        binary_preds = np.where(numpy_y_preds >= 0.5, 1, 0)
        binary_preds = binary_preds.astype(np.int8)
        binary_expected = numpy_y_train.astype(np.int8)
        total_accuracy += np.sum(binary_preds == binary_expected)
    if verbose:
        print("______________________________________________________________")
        print(f"Epoch {epoch} Train Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)} | Accuracy: {round(total_accuracy/total_examples, 4)}")
    if log:
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} Train Loss: {round(total_loss/total_examples, 4)} | MSE: {round(total_MSE/total_examples, 4)} | MAE: {round(total_MAE/total_examples, 4)} | Accuracy: {round(total_accuracy/total_examples, 4)}\n")
    return total_loss

def train(train_loader_path, val_loader_path, test_loader_path, model, optimizer, loss_func, num_hanging_values, epochs, device, learning_rate, log = 1, log_file = "./Training_Logs/Training.txt", verbose = 1, val = False, early_callback = False, early_callback_epochs = 100,
          checkpoint = True, epochs_per_checkpoint = 4, break_after_checkpoint = True, checkpoint_filename = "./Models/PikeBot_Models/PikeBot_checkpoint.pth", change_learning_rate = True, update_epochs = None):
    '''
    Trains a neural network model with optional checkpointing and early callback.

    Parameters:
        train_loader_path (str): Path for loading a data generator/loader saved as a pickle object for training.
        val_loader_path (str): Path for loading a data generator/loader saved as a pickle object for evaluation.
        test_loader_path (str): Path for loading a data generator/loader saved as a pickle object for testing.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_func (callable): Loss function used for training.
        num_hanging_values (int): Number of hanging values.
        epochs (int): Number of epochs for training.
        device (str): Device to run the training (e.g., "cpu" or "cuda").
        learning_rate (float): Initial learning rate of the optimizer.
        log (bool): Flag indicating whether to log training progress (default is 1).
        log_file (str, optional): Path to the log file (default is "./Training_Logs/Training.txt").
        verbose (bool, optional): Optional printing of the training process.
        val (bool, optional): Flag indicating whether to validate during training (default is False).
        early_callback (bool, optional): Flag indicating whether to use early stopping callback (default is False).
        early_callback_epochs (int, optional): Number of epochs for early stopping (default is 100).
        checkpoint (bool, optional): Flag indicating whether to save checkpoints during training (default is True).
        epochs_per_checkpoint (int, optional): Number of epochs per checkpoint (default is 4).
        break_after_checkpoint (bool, optional): Flag indicating whether to break after saving a checkpoint (default is True).
        checkpoint_filename (str, optional): Path to save the model checkpoints (default is "./Models/PikeBot_Models/PikeBot.pth").
        change_learning_rate (bool, optional): Flag, if the learning rate should be updated during training, defaults to True.
        update_epochs (list, optional): Epoch numbers for which the learning rate should be updated, defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    '''
    if early_callback:
        best_val_loss = np.inf
        no_improvement_counter = 0
        best_model = None
    if log:
        if not os.path.exists(log_file):
            open(log_file, 'a').close()
    epoch = 0

    if checkpoint and os.path.exists(checkpoint_filename):
        if verbose:
            print("Checkpoint found. Resuming training from checkpoint...")
        checkpoint = torch.load(checkpoint_filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learning_rate = checkpoint['learning_rate']
        if early_callback:
            best_val_loss = checkpoint['best_val_loss']
            best_model = checkpoint['best_model']

    while epoch < epochs:
        if change_learning_rate:
            learning_rate = update_learning_rate(optimizer=optimizer, curr_lr=learning_rate, epoch=epoch, update_epochs=update_epochs)
        train_loader = efficent_load_object(train_loader_path)
        train_epoch(train_loader, model, optimizer, loss_func, num_hanging_values, device, verbose, log, log_file, epoch)
        del train_loader
        if val:
            val_loader = efficent_load_object(val_loader_path)
            val_loss = evaluate(val_loader, model, loss_func, num_hanging_values, device, verbose, log, log_file, epoch, name = "Val")
            del val_loader
            if early_callback:
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_counter += 1
                    if no_improvement_counter > early_callback_epochs:
                        model = copy.deepcopy(best_model)
                        if verbose:
                            print("*****************")
                            print("Early Callback")
                            print("*****************")
                        break
        epoch += 1
        if checkpoint and epoch % epochs_per_checkpoint == 0:
            if verbose:
                print(f"Epoch {epoch}: Saving checkpoint...")
            if early_callback:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_model': best_model,
                    'learning_rate' : learning_rate
                }, checkpoint_filename)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': learning_rate
                }, checkpoint_filename)
            if break_after_checkpoint:
                break

    test_loader = efficent_load_object(test_loader_path)
    test_loss = evaluate(test_loader, model, loss_func, num_hanging_values, device, verbose, log, log_file, epoch, name = "Test")
    del test_loader
    return model

def save_model(model, model_filename, onnx_filename, bitboard_input_shape, hanging_values_input_shape, opset_version = 11, device = "cpu"):
    '''
    This function saves a model in PyTorch and ONNX formats.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_filename (str): File path to save the PyTorch model.
        onnx_filename (str): File path to save the ONNX model.
        bitboard_input_shape (tuple): Shape of the bitboard input.
        hanging_values_input_shape (tuple): Shape of the hanging values input.
        opset_version (int, optional): ONNX opset version (default is 11).
        device (str, optional): Device to run the model (default is "cpu").
    '''
    model.eval()
    model = model.to(device)
    torch.save(model, model_filename)
    input_bitboard = torch.tensor(np.random.rand(*bitboard_input_shape), dtype = torch.float32)
    input_floats = torch.tensor(np.random.rand(*hanging_values_input_shape), dtype = torch.float32)
    torch.onnx.export(model, (input_bitboard, input_floats), onnx_filename, opset_version=opset_version)
    print("Model saved successfully!")

class ApplySqueezeExcitation(nn.Module):
    def __init__(self, reshape_size):
        super(ApplySqueezeExcitation, self).__init__()
        self.reshape_size = reshape_size

    def forward(self, inputs):
        x, excited = inputs
        batch_size = x.size(0)
        excited = excited.view(batch_size, 2, self.reshape_size, 1, 1)
        gammas, betas = excited[:, 0, :, :, :], excited[:, 1, :, :, :]
        return torch.sigmoid(gammas) * x + betas

class SqueezeExcitationV2(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SqueezeExcitationV2, self).__init__()
        self.se_ratio = se_ratio
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // se_ratio)
        self.fc2 = nn.Linear(channels // se_ratio, 2 * channels)

    def forward(self, inputs):
        pooled = self.global_avg_pool(inputs).view(inputs.size(0), -1)
        squeezed = F.relu(self.fc1(pooled))
        excited = self.fc2(squeezed)
        return ApplySqueezeExcitation(inputs.size(1))([inputs, excited])

class ConvBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3):
        super(ConvBlockV2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlockV2(nn.Module):
    def __init__(self, channels, se_ratio):
        super(ResidualBlockV2, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5)
        self.se = SqueezeExcitationV2(channels, se_ratio)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(x + out)

class Chess_Model(nn.Module):
    def __init__(self, bit_board_shape, num_float_inputs, channel_multiple):
        super(Chess_Model, self).__init__()
        self.num_channels = bit_board_shape[0]
        self.multiple = channel_multiple
        self.num_float_inputs = num_float_inputs
        
        #RESNET BLOCK 1
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)

        #RESNET BLOCK 2
        self.conv5 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels*self.multiple, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=self.num_channels*self.multiple, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.float_inputs_fc = nn.Linear(self.num_float_inputs, 512)

        dummy_input = torch.randn(1, *bit_board_shape)
        self.concat_size = self._get_concatenated_size(dummy_input)
        self.fc1 = nn.Linear(self.concat_size, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, bit_board, hanging_inputs):
        conv_x = self.pool(self.ResNetBlock1(bit_board))
        conv_x = self.pool(self.ResNetBlock2(conv_x))

        conv_x = conv_x.view(conv_x.size(0), -1)
        float_x = nn.functional.relu(self.float_inputs_fc(hanging_inputs))
        x = torch.cat((float_x, conv_x), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x =  torch.sigmoid(self.output_layer(x))
        return x
    
    def _get_concatenated_size(self, dummy_input):
        with torch.no_grad():
            conv_x = self.pool(self.ResNetBlock1(dummy_input))
            conv_x = self.pool(self.ResNetBlock2(conv_x))
            conv_x = conv_x.view(conv_x.size(0), -1)
        return conv_x.size(1) + 512

    
    def ResNetBlock1(self, x):
        conv_x1 = self.conv1(x)
        conv_x2 = self.conv2(conv_x1)
        added1 = conv_x1 + conv_x2
        conv_x3 =self.conv3(added1)
        added2 = conv_x1 + conv_x2 + conv_x3
        conv_x4 = self.conv4(added2)
        return conv_x4 + x
    
    def ResNetBlock2(self, x):
        conv_x1 = self.conv5(x)
        conv_x2 = self.conv6(conv_x1)
        added1 = conv_x1 + conv_x2
        conv_x3 =self.conv7(added1)
        added2 = conv_x1 + conv_x2 + conv_x3
        conv_x4 = self.conv8(added2)
        return conv_x4 + x
    
    def ResNetBlock3(self, x):
        conv_x1 = self.conv9(x)
        conv_x2 = self.conv10(conv_x1)
        added1 = conv_x1 + conv_x2
        conv_x3 =self.conv11(added1)
        added2 = conv_x1 + conv_x2 + conv_x3
        conv_x4 = self.conv12(added2)
        return conv_x4 + x

class ChessModel_V2(nn.Module):
    def __init__(self, bit_board_shape, num_float_inputs, residual_blocks, residual_filters, se_ratio):
        super(ChessModel_V2, self).__init__()
        self.RESIDUAL_FILTERS = residual_filters
        self.RESIDUAL_BLOCKS = residual_blocks
        self.SE_ratio = se_ratio
        self.num_channels = bit_board_shape[0]
        self.num_float_inputs = num_float_inputs

        self.relu = nn.ReLU()
        self.conv_block = ConvBlockV2(self.num_channels, self.RESIDUAL_FILTERS)
        self.residual_blocks = nn.Sequential(*[ResidualBlockV2(self.RESIDUAL_FILTERS, self.SE_ratio) for _ in range(self.RESIDUAL_BLOCKS)])
        self.conv_val = ConvBlockV2(self.RESIDUAL_FILTERS, 32, filter_size=1)

        dummy_input = torch.randn(1, *bit_board_shape)
        self.concatenated_size = self._get_concatenated_size(dummy_input)
        self.fc1 = nn.Linear(self.concatenated_size, 512)
        self.fc2 = nn.Linear(512, 4096)
        self.fc3 = nn.Linear(4096, 32)
        self.output = nn.Linear(32, 1)
    
    def _get_concatenated_size(self, dummy_input):
        with torch.no_grad():
            flow = self.conv_block(dummy_input)
            flow = self.residual_blocks(flow)
            conv_val = self.conv_val(flow)
            h_conv_val_flat = conv_val.view(conv_val.size(0), -1)
            return h_conv_val_flat.size(1) + self.num_float_inputs

    def forward(self, bit_board, hanging_inputs):
        flow = self.conv_block(bit_board)
        flow = self.residual_blocks(flow)
        conv_val = self.conv_val(flow)
        h_conv_val_flat = conv_val.view(conv_val.size(0), -1)
        x = torch.cat((h_conv_val_flat, hanging_inputs), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x
    
def count_parameters(model):
    '''
    Counts the number of parameters in the model.

    Parameters:
        model (torch.nn.Module): The neural network model.

    Returns:
        Int: Number of model parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model(model, loss_func, num_hanging_values, device, test_generator_path="./Generators/test_generator.pkl"):
    '''
    Tests the model's performance using the TEST_GENERATOR and prints the accuracy and loss on the test set.

    Parameters:
        model (torch.nn.Module): The neural network model.
        loss_func (callable): Loss function used for evaluation.
        num_hanging_values (int): Number of hanging values.
        device (str): Device to run the evaluation (e.g., "cpu" or "cuda").
        test_generator_path (str, optional): Test generator path (default is "./Generators/test_generator.pkl").
    '''
    model.eval()
    test_loader = efficent_load_object(test_generator_path)
    total_loss = 0.0
    total_MSE = 0.0
    total_MAE = 0.0
    total_accuracy = 0.0
    total_examples = 0
    with torch.no_grad():
        for X_train, y_train in test_loader:
            if X_train is None and y_train is None:
                break
            total_examples += len(y_train)
            hanging_vals, bitboards = separate_hanging_and_bitboards(X_train)
            hanging_vals = hanging_vals.astype(np.float32)
            hanging_vals = np.nan_to_num(hanging_vals, nan=0.0)
            hanging_vals = torch.Tensor(hanging_vals).to(device)
            bitboards = torch.Tensor(bitboards).to(device)
            y_train = y_train.astype(np.float32)
            y_train = y_train.reshape(-1, 1)
            y_train = torch.Tensor(y_train).to(device)
            y_preds = model(bitboards, hanging_vals)
            loss = loss_func(y_preds, y_train)
            total_loss += loss.item()
            numpy_y_preds = y_preds.detach().cpu().numpy()
            numpy_y_train = y_train.detach().cpu().numpy()
            total_MSE += np.sum(np.square(numpy_y_train - numpy_y_preds))
            total_MAE += np.sum(np.abs(numpy_y_train - numpy_y_preds))
            binary_preds = np.where(numpy_y_preds >= 0.5, 1, 0)
            binary_preds = binary_preds.astype(np.int8)
            binary_expected = numpy_y_train.astype(np.int8)
            total_accuracy += np.sum(binary_preds == binary_expected)
    avr_loss = round(total_loss/total_examples, 4)
    avr_MSE = round(total_MSE/total_examples, 4)
    avr_MAE = round(total_MAE/total_examples, 4)
    avr_accuracy = round(total_accuracy/total_examples, 4)
    print(f"Testing Complete, Loss: {avr_loss} | MSE: {avr_MSE} | MAE: {avr_MAE} | Accuracy: {avr_accuracy}")