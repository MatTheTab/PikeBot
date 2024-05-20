import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.onnx
import copy
from utils.generator_utils import *

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
            hanging_vals = X_train[:, :num_hanging_values]
            bitboards = X_train[:, num_hanging_values]
            bitboards = np.stack(bitboards)
            hanging_vals = hanging_vals.astype(np.float32)
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
        hanging_vals = X_train[:, :num_hanging_values]
        bitboards = X_train[:, num_hanging_values]
        bitboards = np.stack(bitboards)
        hanging_vals = hanging_vals.astype(np.float32)
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

def train(train_loader, val_loader, test_loader, model, optimizer, loss_func, num_hanging_values, epochs, device, log = 1, log_file = "./Training_Logs\\Training.txt", verbose = 1, val = False, early_callback = False, early_callback_epochs = 100,
          checkpoint = True, epochs_per_checkpoint = 4, break_after_checkpoint = True, checkpoint_filename = "./Models\\PikeBot_Models\\PikeBot_checkpoint.pth"):
    '''
    Trains a neural network model with optional checkpointing and early callback.

    Parameters:
        train_loader (DataLoader/Generator): DataLoader for the training dataset.
        val_loader (DataLoader/Generator): DataLoader for the validation dataset.
        test_loader (DataLoader/Generator): DataLoader for the test dataset.
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_func (callable): Loss function used for training.
        num_hanging_values (int): Number of hanging values.
        epochs (int): Number of epochs for training.
        device (str): Device to run the training (e.g., "cpu" or "cuda").
        log (bool): Flag indicating whether to log training progress (default is 1).
        log_file (str, optional): Path to the log file (default is "./Training_Logs\\Training.txt").
        verbose (bool, optional): Optional printing of the training process.
        val (bool, optional): Flag indicating whether to validate during training (default is False).
        early_callback (bool, optional): Flag indicating whether to use early stopping callback (default is False).
        early_callback_epochs (int, optional): Number of epochs for early stopping (default is 100).
        checkpoint (bool, optional): Flag indicating whether to save checkpoints during training (default is True).
        epochs_per_checkpoint (int, optional): Number of epochs per checkpoint (default is 4).
        break_after_checkpoint (bool, optional): Flag indicating whether to break after saving a checkpoint (default is True).
        checkpoint_filename (str, optional): Path to save the model checkpoints (default is "./Models\\PikeBot_Models\\PikeBot.pth").

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
        if early_callback:
            best_val_loss = checkpoint['best_val_loss']
            best_model = checkpoint['best_model']

    while epoch < epochs:
        train_epoch(train_loader, model, optimizer, loss_func, num_hanging_values, device, verbose, log, log_file, epoch)
        if val:
            val_loss = evaluate(val_loader, model, loss_func, num_hanging_values, device, verbose, log, log_file, epoch, name = "Val")
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
                    'best_model': best_model
                }, checkpoint_filename)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_filename)
            if break_after_checkpoint:
                break

    test_loss = evaluate(test_loader, model, loss_func, num_hanging_values, device, verbose, log, log_file, epoch, name = "Test")
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

class Chess_Model(nn.Module):
    def __init__(self, bit_board_shape, num_float_inputs, channel_multiple, concatenated_size):
        super(Chess_Model, self).__init__()
        self.num_channels = bit_board_shape[0]
        self.multiple = channel_multiple
        self.num_float_inputs = num_float_inputs
        self.concat_size = concatenated_size
        
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
    
def count_parameters(model):
    '''
    Counts the number of parameters in the model.

    Parameters:
        model (torch.nn.Module): The neural network model.

    Returns:
        Int: Number of model parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)