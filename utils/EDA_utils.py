from utils.preprocess_utils import *
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column_name, bins=50, title=None, xlabel=None, ylabel='Relative Frequency'):
    """
    Plots a histogram for a specified column in the DataFrame with relative frequency on the y-axis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    bins (int): The number of bins for the histogram.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name].dropna(), bins=bins, color='blue', alpha=0.7, edgecolor='black', density=True)
    plt.title(title if title else f'Histogram of {column_name}')
    plt.xlabel(xlabel if xlabel else column_name)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_smoothed_histogram(df, column_name, bins=50, title=None, xlabel=None, ylabel='Relative Frequency', kde=True):
    """
    Plots a smoothed histogram (KDE plot) for a specified column in the DataFrame with relative frequency on the y-axis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    bins (int): The number of bins for the histogram.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    kde (bool): Flag, if overlay a Kernel Density Estimate should be present (default is True).
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name].dropna(), bins=bins, kde=kde, color='blue', stat='density', edgecolor='black')
    plt.title(title if title else f'Smoothed Histogram of {column_name}')
    plt.xlabel(xlabel if xlabel else column_name)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_categorical_distribution(df, column_name, title=None, xlabel=None, ylabel='Relative Frequency'):
    """
    Plots the distribution of a categorical column in the DataFrame with relative frequency on the y-axis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the categorical column to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    """
    plt.figure(figsize=(10, 6))
    category_counts = df[column_name].value_counts(normalize=True)
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title(title if title else f'Distribution of {column_name}')
    plt.xlabel(xlabel if xlabel else column_name)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

def create_chessboard_position_heatmap(df, array_column, title = 'Relative Frequency of Occupied Spaces for The Current Move'):
    """
    Creates a heatmap showing the relative frequency of occupied spaces on an 8x8 chessboard.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numpy arrays.
    array_column (str): The name of the column with numpy arrays of shape (72, 8, 8).
    """
    board_frequency = np.zeros((8, 8), dtype=int)
    num_arrays = len(df)
    for array in df[array_column]:
        slices = array[:12, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency += combined_board
    relative_frequency = board_frequency / num_arrays
    plt.figure(figsize=(8, 6))
    sns.heatmap(relative_frequency, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True)
    plt.title(title)
    plt.xlabel('File')
    plt.ylabel('Rank')
    plt.show()

def create_chessboard_attack_heatmap(df, array_column, title = 'Relative Frequency of Attacked Spaces for The Current Move'):
    """
    Creates a heatmap showing the relative frequency of occupied spaces on an 8x8 chessboard.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numpy arrays.
    array_column (str): The name of the column with numpy arrays of shape (72, 8, 8).
    """
    board_frequency = np.zeros((8, 8), dtype=int)
    num_arrays = len(df)
    for array in df[array_column]:
        slices = array[12:, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency += combined_board
    relative_frequency = board_frequency / num_arrays
    plt.figure(figsize=(8, 6))
    sns.heatmap(relative_frequency, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True)
    plt.title(title)
    plt.xlabel('File')
    plt.ylabel('Rank')
    plt.show()

def plot_violin_by_category(df, numeric_column, category_column, title=None, xlabel=None, ylabel=None):
    """
    Creates a violin plot showing the distribution of values in a numeric column, divided by categories in another column.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    numeric_column (str): The name of the numeric column to plot.
    category_column (str): The name of the categorical column to divide the data.
    title (str): The title of the plot (optional).
    xlabel (str): The label for the x-axis (optional).
    ylabel (str): The label for the y-axis (optional).
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[category_column], y=df[numeric_column], palette='Set2')
    plt.title(title if title else f'Distribution of {numeric_column} by {category_column}')
    plt.xlabel(xlabel if xlabel else category_column)
    plt.ylabel(ylabel if ylabel else numeric_column)
    plt.grid(True)
    plt.show()

def plot_comparison_histograms(df, column_name, bins=50, title=None, xlabel=None, ylabel='Relative Frequency'):
    """
    Creates a subplot with two histograms comparing the distribution of a column for 'human' values 1 and 0.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    bins (int): The number of bins for the histograms.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    """
    df_human_1 = df[df['human'] == 1]
    df_human_0 = df[df['human'] == 0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    axs[0].hist(df_human_1[column_name].dropna(), bins=bins, color='blue', alpha=0.7, edgecolor='black', density=True)
    axs[0].set_title(f'{title} (Human = 1)' if title else f'Histogram of {column_name} (Human = 1)')
    axs[0].set_xlabel(xlabel if xlabel else column_name)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(True)
    
    axs[1].hist(df_human_0[column_name].dropna(), bins=bins, color='orange', alpha=0.7, edgecolor='black', density=True)
    axs[1].set_title(f'{title} (Human = 0)' if title else f'Histogram of {column_name} (Human = 0)')
    axs[1].set_xlabel(xlabel if xlabel else column_name)
    axs[1].set_ylabel(ylabel)
    axs[1].grid(True)
    
    plt.suptitle(title if title else f'Comparison of {column_name} Distribution by Human Status')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_comparison_smoothed_histograms(df, column_name, bins=50, title=None, xlabel=None, ylabel='Relative Frequency', kde=True):
    """
    Creates a subplot with two smoothed histograms comparing the distribution of a column for 'human' values 1 and 0.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    bins (int): The number of bins for the histogram.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    kde (bool): Whether to overlay a Kernel Density Estimate (default is True).
    """
    df_human_1 = df[df['human'] == 1]
    df_human_0 = df[df['human'] == 0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    sns.histplot(df_human_1[column_name].dropna(), bins=bins, kde=kde, color='blue', stat='density', edgecolor='black', ax=axs[0])
    axs[0].set_title(f'{title} (Human = 1)' if title else f'Smoothed Histogram of {column_name} (Human = 1)')
    axs[0].set_xlabel(xlabel if xlabel else column_name)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(True)
    
    sns.histplot(df_human_0[column_name].dropna(), bins=bins, kde=kde, color='orange', stat='density', edgecolor='black', ax=axs[1])
    axs[1].set_title(f'{title} (Human = 0)' if title else f'Smoothed Histogram of {column_name} (Human = 0)')
    axs[1].set_xlabel(xlabel if xlabel else column_name)
    axs[1].set_ylabel(ylabel)
    axs[1].grid(True)
    
    plt.suptitle(title if title else f'Comparison of {column_name} Distribution by Human Status')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_comparison_categorical_distributions(df, column_name, title=None, xlabel=None, ylabel='Relative Frequency'):
    """
    Creates a subplot with two bar plots comparing the distribution of a categorical column for 'human' values 1 and 0.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the categorical column to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    """
    df_human_1 = df[df['human'] == 1]
    df_human_0 = df[df['human'] == 0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    category_counts_1 = df_human_1[column_name].value_counts(normalize=True)
    sns.barplot(x=category_counts_1.index, y=category_counts_1.values, palette='viridis', ax=axs[0])
    axs[0].set_title(f'{title} (Human = 1)' if title else f'Distribution of {column_name} (Human = 1)')
    axs[0].set_xlabel(xlabel if xlabel else column_name)
    axs[0].set_ylabel(ylabel)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    category_counts_0 = df_human_0[column_name].value_counts(normalize=True)
    sns.barplot(x=category_counts_0.index, y=category_counts_0.values, palette='viridis', ax=axs[1])
    axs[1].set_title(f'{title} (Human = 0)' if title else f'Distribution of {column_name} (Human = 0)')
    axs[1].set_xlabel(xlabel if xlabel else column_name)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(title if title else f'Comparison of {column_name} Distribution by Human Status')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_comparison_chessboard_heatmaps_positions(df, array_column, title=None):
    """
    Creates a subplot with two heatmaps comparing the relative frequency of occupied spaces on an 8x8 chessboard 
    for 'human' values 1 and 0.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numpy arrays.
    array_column (str): The name of the column with numpy arrays of shape (72, 8, 8).
    title (str): The title of the plot (optional).
    """
    board_frequency_1 = np.zeros((8, 8), dtype=int)
    board_frequency_0 = np.zeros((8, 8), dtype=int)
    df_human_1 = df[df['human'] == 1]
    df_human_0 = df[df['human'] == 0]
    
    num_arrays_1 = len(df_human_1)
    for array in df_human_1[array_column]:
        slices = array[:12, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency_1 += combined_board
    
    num_arrays_0 = len(df_human_0)
    for array in df_human_0[array_column]:
        slices = array[:12, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency_0 += combined_board
    
    relative_frequency_1 = board_frequency_1 / num_arrays_1 if num_arrays_1 > 0 else np.zeros((8, 8))
    relative_frequency_0 = board_frequency_0 / num_arrays_0 if num_arrays_0 > 0 else np.zeros((8, 8))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    sns.heatmap(relative_frequency_1, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True, ax=axs[0])
    axs[0].set_title(f'{title} (Human = 1)' if title else 'Relative Frequency of Occupied Spaces (Human = 1)')
    axs[0].set_xlabel('File')
    axs[0].set_ylabel('Rank')
    
    sns.heatmap(relative_frequency_0, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True, ax=axs[1])
    axs[1].set_title(f'{title} (Human = 0)' if title else 'Relative Frequency of Occupied Spaces (Human = 0)')
    axs[1].set_xlabel('File')
    axs[1].set_ylabel('Rank')
    
    plt.suptitle(title if title else 'Comparison of Occupied Spaces on the 8x8 Chessboard by Human Status')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_comparison_chessboard_heatmaps_attacks(df, array_column, title=None):
    """
    Creates a subplot with two heatmaps comparing the relative frequency of occupied spaces on an 8x8 chessboard 
    for 'human' values 1 and 0.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numpy arrays.
    array_column (str): The name of the column with numpy arrays of shape (72, 8, 8).
    title (str): The title of the plot (optional).
    """
    board_frequency_1 = np.zeros((8, 8), dtype=int)
    board_frequency_0 = np.zeros((8, 8), dtype=int)
    
    df_human_1 = df[df['human'] == 1]
    df_human_0 = df[df['human'] == 0]
    num_arrays_1 = len(df_human_1)
    for array in df_human_1[array_column]:
        slices = array[12:, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency_1 += combined_board
    
    num_arrays_0 = len(df_human_0)
    for array in df_human_0[array_column]:
        slices = array[12:, :, :]
        combined_board = np.any(slices, axis=0).astype(int)
        board_frequency_0 += combined_board
    
    relative_frequency_1 = board_frequency_1 / num_arrays_1 if num_arrays_1 > 0 else np.zeros((8, 8))
    relative_frequency_0 = board_frequency_0 / num_arrays_0 if num_arrays_0 > 0 else np.zeros((8, 8))
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.heatmap(relative_frequency_1, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True, ax=axs[0])
    axs[0].set_title(f'{title} (Human = 1)' if title else 'Relative Frequency of Attacked Spaces (Human = 1)')
    axs[0].set_xlabel('File')
    axs[0].set_ylabel('Rank')
    
    sns.heatmap(relative_frequency_0, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5, square=True, ax=axs[1])
    axs[1].set_title(f'{title} (Human = 0)' if title else 'Relative Frequency of Attacked Spaces (Human = 0)')
    axs[1].set_xlabel('File')
    axs[1].set_ylabel('Rank')
    
    plt.suptitle(title if title else 'Comparison of Attacked Spaces on the 8x8 Chessboard by Human Status')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_multiple_histograms(df, columns, bins=50, title=None, xlabel=None, ylabel='Relative Frequency'):
    """
    Creates subplots with histograms for each specified column in the DataFrame, with relative frequency on the y-axis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list of str): The list of column names to plot.
    bins (int): The number of bins for the histograms.
    title (str): The title of the overall figure.
    xlabel (str): The label for the x-axis (default is None).
    ylabel (str): The label for the y-axis (default is 'Relative Frequency').
    """
    num_columns = len(columns)
    cols_per_row = 2
    rows = (num_columns + cols_per_row - 1) // cols_per_row
    fig, axs = plt.subplots(rows, cols_per_row, figsize=(10, 5 * rows), sharey=True)
    axs = axs.flatten()
    
    for i, column in enumerate(columns):
        axs[i].hist(df[column].dropna(), bins=bins, color='blue', alpha=0.7, edgecolor='black', density=True)
        axs[i].set_title(f'Histogram of {column}')
        axs[i].set_xlabel(xlabel if xlabel else column)
        axs[i].set_ylabel(ylabel)
        axs[i].grid(True)

    for j in range(num_columns, len(axs)):
        axs[j].axis('off')
    
    plt.suptitle(title if title else 'Histograms of Columns')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_comparison_violin(df_list, column_name, df_labels, title=None, xlabel=None, ylabel='Distribution'):
    """
    Creates a violin plot comparing the distributions of values in a specified column across multiple DataFrames.
    
    Parameters:
    df_list (list of pandas.DataFrame): List of DataFrames to compare.
    column_name (str): The name of the column to plot.
    df_labels (list of str): Labels for each DataFrame, to be used in the plot.
    title (str): The title of the plot (optional).
    xlabel (str): The label for the x-axis (optional).
    ylabel (str): The label for the y-axis (default is 'Distribution').
    """
    if len(df_list) != len(df_labels):
        raise ValueError("The number of DataFrames and labels must be the same.")
    
    combined_df = pd.concat(
        [df.assign(Source=label) for df, label in zip(df_list, df_labels)],
        ignore_index=True
    )
    
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Source', y=column_name, data=combined_df, palette='Set2')
    
    plt.title(title if title else f'Comparison of {column_name} Distributions')
    plt.xlabel(xlabel if xlabel else 'DataFrame')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_correlation_heatmap(df, columns=None, title='Correlation Heatmap'):
    """
    Creates a heatmap of correlations between specified attributes.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list of str): List of column names to include in the heatmap (default is None, which includes all columns).
    title (str): The title of the heatmap (default is 'Correlation Heatmap').
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel('Attributes')
    plt.ylabel('Attributes')
    plt.show()

def plot_correlation_with_human(df, numerical_columns, categorical_column='human', title=None, xlabel=None, ylabel='Correlation'):
    """
    Creates a bar chart showing the correlation of numerical variables with the categorical variable 'human'.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    numerical_columns (list of str): List of numerical column names to check correlation with 'human'.
    categorical_column (str): The name of the categorical column to check correlation with (default is 'human').
    title (str): The title of the plot (optional).
    xlabel (str): The label for the x-axis (optional).
    ylabel (str): The label for the y-axis (default is 'Correlation').
    """
    correlations = {}
    df[categorical_column] = df[categorical_column].astype(float)
    
    for column in numerical_columns:
        correlation = df[[column, categorical_column]].corr().iloc[0, 1]
        correlations[column] = correlation
    
    correlations_df = pd.DataFrame(list(correlations.items()), columns=['Variable', 'Correlation'])
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Variable', y='Correlation', data=correlations_df, color='grey')
    plt.title(title if title else f'Correlation with {categorical_column}')
    plt.xlabel(xlabel if xlabel else 'Numerical Variable')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()