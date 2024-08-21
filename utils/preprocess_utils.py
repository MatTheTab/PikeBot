from utils.data_utils import *

def change_column_types(df, columns, types):
    '''
    Changes data types of chosen columns in a DataFrame.

    Params:
    - df (pd.Dataframe): Dataframe to process.
    - columns (list): Columns to process.
    - types (list): New column types.
    '''
    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].astype(types[i])
    return df

def map_string_val(df, columns, str_vals):
    '''
    Changes chosen string values in specified columns to binary variables. If value is the same as in str_vals, then 1 otherwise 0.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to process.
        - str_vals (list): String values to map to 1, others to 0.
    '''
    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].apply(lambda x: 1 if x==str_vals[i] else 0)
    return df

def one_hot_encode_col(df, columns, recognised_vals):
    '''
    Performs one-hot encoding on chosen columns in a DataFrame.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to one-hot encode.
        - recognised_vals (list of lists): Discreet buckets for one-hot encoding.
    '''
    for col, vals in zip(columns, recognised_vals):
        for val in vals:
            df[val] = df[col].str.lower().str.contains(val).astype(int)
    df = df.drop(columns=columns)
    return df

def manually_divide(df, columns, values):
    '''
    Divides values in specified columns by provided divisor.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to divide.
        - values (list): Divisor values.
    '''
    for i in range(len(columns)):
        df[columns[i]] = df[columns[i]].astype(np.float32)
        df[columns[i]] /= values[i]
    return df

def calculate_mean(column, directory):
    '''
    Calculates the mean of a specified column from all batches present in a chosen directory. Currently not used in the preprocess function.

    Params:
        - column (str): Name of the column.
        - directory (str): Directory containing data batches.
    '''
    files = os.listdir(directory)
    npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
    num_examples = 0
    sum_val = 0
    for file in npy_gz_files:
        df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
        sum_val += df[column].sum()
        num_examples += len(df[column])
    return sum_val/num_examples

def calculate_max(column, directory):
    '''
    Calculates the max value from all batches present in a directory. Currently not used in the preprocess function.

    Params:
        - column (str): Name of the column.
        - directory (str): Directory containing batches.
    '''
    files = os.listdir(directory)
    npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
    max_val = -99999
    for file in npy_gz_files:
        df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
        temp_val = df[column].max()
        if temp_val > max_val:
            max_val = temp_val
    return max_val

def calculate_min(column, directory):
    '''
    Calculates the min value from all batches present in a directory. Currently not used in the preprocess function.

    Params:
        - column (str): Name of the column.
        - directory (str): Directory containing batches.
    '''
    files = os.listdir(directory)
    npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
    min_val = 99999
    for file in npy_gz_files:
        df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
        temp_val = df[column].min()
        if temp_val < min_val:
            min_val = temp_val
    return min_val

def calculate_num_examples(column, directory):
    '''
    Calculates the total number of examples from all batches present in a directory. Currently not used in the preprocess function.

    Params:
        - column (str): Name of the column.
        - directory (str): Directory containing batches.
    '''
    files = os.listdir(directory)
    npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
    num_examples = 0
    for file in npy_gz_files:
        df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
        num_examples += len(df[column])
    return num_examples

def calculate_standard_deviation(column, mean, num_examples, directory):
    '''
    Calculates the standard deviation value from all batches present in a directory. Currently not used in the preprocess function.

    Params:
        - column (str): Name of the column.
        - directory (str): Directory containing batches.
    '''
    files = os.listdir(directory)
    npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
    std_div = 0
    for file in npy_gz_files:
        df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
        sqr_diff = np.square(df[column] - mean)
        std_div += sqr_diff.sum()
    return np.sqrt(std_div/num_examples)

def normalize(df, columns, mins, maxs):
    '''
    Normalizes specified columns in a DataFrame using given min-max values. Currently not used in the preprocess function.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to normalize.
        - mins (list): Minimum values for normalization.
        - maxs (list): Maximum values for normalization.
    '''
    for i in range(len(columns)):
        df[columns[i]] = (df[columns[i]] - mins[i])/(maxs[i]-mins[i])
    return df

def denormalize(df, columns, mins, maxs, batch_number):
    '''
    Denormalizes specified columns in a DataFrame using given min-max values. Currently not used in the preprocess function.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (dict): Columns to denormalize.
        - mins (list): Minimum values for denormalization.
        - maxs (list): Maximum values for denormalization.
    '''
    for column in columns:
        try:
            df[column] = df[column]*(maxs[batch_number][column]-mins[batch_number][column]) + mins[batch_number][column]
        except:
            pass
    return df

def standardize(df, columns, means, stds):
    '''
    Standardizes specified columns in a DataFrame using given mean and standard deviation values. Currently not used in the preprocess function.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to Standardize.
        - mins (list): Minimum values for Standardize.
        - maxs (list): Maximum values for Standardize.
    '''
    for i in range(len(columns)):
        df[columns[i]] = (df[columns[i]] - means[i])/stds[i]
    return df

def destandardize(df, columns, means, stds, batch_number):
    '''
    Destandardizes specified columns in a DataFrame using given mean and standard deviation values. Currently not used in the preprocess function.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to Destandardize.
        - mins (list): Minimum values for Destandardize.
        - maxs (list): Maximum values for Destandardize.
    '''
    for column in columns:
        try:
            df[column] = (df[column]*stds[batch_number][column]) + means[batch_number][column]
        except:
            pass
    return df

def batch_normalize(df, columns):
    '''
    Normalizes specified columns in a DataFrame based on data from a single batch. More efficent than normalize.
    Aside of the chnaged df also returns lists of minimum and maximal values for all chosen columns for this particular batch.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to normalize.
    '''
    mins = {}
    maxs = {}
    for col in columns:
        min = df[col].min()
        max = df[col].max()
        df[col] = (df[col] - min)/(max-min)
        mins[col] = min
        maxs[col] = max
    return df, mins, maxs

def batch_standardize(df, columns):
    '''
    Standardizes specified columns in a DataFrame based on data from a single batch. More efficent than standardize.
    Aside of the chnaged df also returns lists of mean and standard deviation values for all chosen columns for this particular batch.

    Params:
        - df (DataFrame): DataFrame to process.
        - columns (list): Columns to normalize.
    '''
    means = {}
    stds = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean)/(std)
        means[col] = mean
        stds[col] = std
    return df, means, stds

def preprocess(df, columns_to_change_types, new_types, columns_to_map, str_vals, columns_to_one_hot, key_strings,
               columns_to_divide, divide_val, cols_to_normalize, cols_to_standardize, columns_to_drop):
        '''
        Preprocesses a DataFrame by performing various transformations including changing column types, mapping string values to binary indicators, one-hot encoding, manual division, normalization, standardization, and dropping columns.

        Params:
                - df (DataFrame): DataFrame to preprocess.
                - columns_to_change_types (list): Columns to change types.
                - new_types (list): New types for specified columns.
                - columns_to_map (list): Columns to map string values to binary indicators.
                - str_vals (list): String values to map to 1, others to 0.
                - columns_to_one_hot (list): Columns to one-hot encode.
                - key_strings (list of lists): Recognised values for each column to encode.
                - columns_to_divide (list): Columns to divide.
                - divide_val (list): Divisor values for specified columns.
                - cols_to_normalize (list): Columns to normalize.
                - cols_to_standardize (list): Columns to standardize.
                - columns_to_drop (list): Columns to drop.

        Returns:
                - df (DataFrame): Preprocessed DataFrame.
                - mins (list): Minimum values used for normalization.
                - maxs (list): Maximum values used for normalization.
                - means (list): Means used for standardization.
                - stds (list): Standard deviations used for standardization.
        '''
        df = change_column_types(df, columns_to_change_types, new_types)
        df = map_string_val(df, columns_to_map, str_vals)
        df = one_hot_encode_col(df, columns_to_one_hot, key_strings)
        df = manually_divide(df, columns_to_divide, divide_val)
        df, mins, maxs = batch_normalize(df, cols_to_normalize)
        df, means, stds = batch_standardize(df, cols_to_standardize)
        df = df.drop(columns = columns_to_drop)
        return df, mins, maxs, means, stds

def copy_columns_file(columns, target_directory, columns_file):
    '''
    Save column names to a file.

    Params:
            - columns (list): Column names to be saved
            - target_directory (str): String representing the target directory to which the file should be saved
            - columns_file (str): String showing the target name of the column names file
    '''
    target_file = f"{target_directory}\\{columns_file}"
    with open(target_file, 'w') as f:
        for column in columns:
            f.write("%s\n" % column)

def preprocess_all(save_name, target_directory, directory, columns_to_change_types, new_types, columns_to_map, str_vals, columns_to_one_hot, key_strings,
               columns_to_divide, divide_val, cols_to_normalize, cols_to_standardize, columns_to_drop):
        '''
        Preprocesses all data batches present in the chosen directory by performing various transformations including changing column types, mapping string values to binary indicators, one-hot encoding, manual division, normalization, standardization, and dropping columns.

        Params:
                - save_name (str): Name under which the new batches should be saved _i will be added automatically, where i is the batch number
                - target_directory (str): Target directory for the new data
                - directory (str): Directory where the original batched data is located
                - columns_to_change_types (list): Columns to change types.
                - new_types (list): New types for specified columns.
                - columns_to_map (list): Columns to map string values to binary indicators.
                - str_vals (list): String values to map to 1, others to 0.
                - columns_to_one_hot (list): Columns to one-hot encode.
                - key_strings (list of lists): Recognised values for each column to encode.
                - columns_to_divide (list): Columns to divide.
                - divide_val (list): Divisor values for specified columns.
                - cols_to_normalize (list): Columns to normalize.
                - cols_to_standardize (list): Columns to standardize.
                - columns_to_drop (list): Columns to drop.

        Returns:
                - df (DataFrame): Preprocessed DataFrame.
                - mins (list): Minimum values used for normalization.
                - maxs (list): Maximum values used for normalization.
                - means (list): Means used for standardization.
                - stds (list): Standard deviations used for standardization.
        '''
        files = os.listdir(directory)
        npy_gz_files = [file for file in files if file.endswith('.npy.gz')]
        mins_list = []
        maxs_list = []
        means_list = []
        stds_list = []
        copied_columns = False
        for i in range(len(npy_gz_files)):
            file = npy_gz_files[i]
            df = read(data_file = f"{directory}\\{file}", column_names_file = f"{directory}\\column_names.txt")
            df = change_column_types(df, columns_to_change_types, new_types)
            df = map_string_val(df, columns_to_map, str_vals)
            df = one_hot_encode_col(df, columns_to_one_hot, key_strings)
            df = manually_divide(df, columns_to_divide, divide_val)
            df, mins, maxes = batch_normalize(df, cols_to_normalize)
            df, means, stds = batch_standardize(df, cols_to_standardize)
            df = df.drop(columns=columns_to_drop)
            if not copied_columns:
                copy_columns_file(list(df.columns), target_directory, "column_names.txt")
                copied_columns = True
            mins_list.append(mins)
            maxs_list.append(maxes)
            means_list.append(means)
            stds_list.append(stds)
            file_save_location = f"{target_directory}\\{save_name}_{i}.npy"
            np.save(file_save_location, df.to_numpy())
            compress_file(file_save_location)
        return mins_list, maxs_list, means_list, stds_list, npy_gz_files

def read_metadata_txt(file_path, separator="---"):
    """
    Reads metadata from a text file where dictionaries are delineated by a specified separator. Converts key-value pairs in the text file into a list of dictionaries with float values.

    Params:
        - file_path (str): Path to the text file containing the metadata.
        - separator (str, optional): String that delineates the end of each dictionary. Defaults to "---".

    Returns:
        - metadata_list (list): A list of dictionaries containing the metadata. Each dictionary represents a set of key-value pairs from the text file.
    """
    metadata_list = []
    with open(file_path, 'r') as f:
        current_dict = {}
        for line in f:
            line = line.strip()
            if line == separator:
                metadata_list.append(current_dict)
                current_dict = {}
            elif line:
                key, value = line.split(': ')
                current_dict[key] = float(value)
        if current_dict:
            metadata_list.append(current_dict)
    return metadata_list

def depreprocess(dataset_type, metadata_path, processed_df, columns_to_divide, divide_val, cols_to_normalize, cols_to_standardize, batch_number):
    """
    Reverses the preprocessing steps on a given DataFrame using stored metadata. This includes reversing normalization, standardization, and division operations.

    Params:
        - dataset_type (str): Type of the dataset ("train", "val", or "test").
        - metadata_path (str): Path to the directory containing the metadata files.
        - processed_df (DataFrame): The DataFrame that was processed and now needs to be depreprocessed.
        - columns_to_divide (list): List of columns that were divided during preprocessing.
        - divide_val (list): List of divisor values corresponding to the columns_to_divide.
        - cols_to_normalize (list): List of columns that were normalized during preprocessing.
        - cols_to_standardize (list): List of columns that were standardized during preprocessing.
        - batch_number (int): Batch number used to retrieve corresponding metadata for normalization and standardization.

    Returns:
        - result_df (DataFrame): The DataFrame after reversing the preprocessing steps.
    """
    result_df = processed_df.copy()
    npy_gz_files = np.load(f"{metadata_path}/npy_gz_files_{dataset_type}.npy")

    if dataset_type == "train":
        mins_list = read_metadata_txt(f"{metadata_path}/train_mins_list.txt")
        maxs_list = read_metadata_txt(f"{metadata_path}/train_maxs_list.txt")
        means_list = read_metadata_txt(f"{metadata_path}/train_means_list.txt")
        stds_list = read_metadata_txt(f"{metadata_path}/train_stds_list.txt")
    elif dataset_type == "val":
        mins_list = read_metadata_txt(f"{metadata_path}/train_mins_list.txt")
        maxs_list = read_metadata_txt(f"{metadata_path}/train_maxs_list.txt")
        means_list = read_metadata_txt(f"{metadata_path}/train_means_list.txt")
        stds_list = read_metadata_txt(f"{metadata_path}/train_stds_list.txt")
    else:
        mins_list = read_metadata_txt(f"{metadata_path}/train_mins_list.txt")
        maxs_list = read_metadata_txt(f"{metadata_path}/train_maxs_list.txt")
        means_list = read_metadata_txt(f"{metadata_path}/train_means_list.txt")
        stds_list = read_metadata_txt(f"{metadata_path}/train_stds_list.txt")

    for i in range(len(columns_to_divide)):
        col = columns_to_divide[i]
        multi = divide_val[i]
        result_df[[col]] = result_df[[col]]* multi

    result_df = denormalize(result_df, cols_to_normalize, mins_list, maxs_list, batch_number)
    result_df = destandardize(result_df, cols_to_standardize, means_list, stds_list, batch_number)
    return result_df