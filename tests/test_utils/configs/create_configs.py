import os
import re

STOCKISH_PATH="D:/Program Files/Stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"

assert os.path.exists(STOCKISH_PATH)

for model_path, preprocessing_parameters_path, in [
    ('Models/PikeBot_Models/PikeBot.pth', "model_loading_data/preprocessing_parameters.json"),
    ('Models/PikeBot_Models/500k/PikeBot500k.pth', "model_loading_data/preprocessing_parameters.json"),
    ('Models/PikeBot_Models/small_500k/PikeBotSmall500k.pth', "model_loading_data/preprocessing_parameters.json"),
]:
    if not os.path.exists(preprocessing_parameters_path):
        print(f"Following file does not exist: {preprocessing_parameters_path} please add it in the proper location")
        continue
    if not os.path.exists(model_path):
        print(f"Following file does not exist: {model_path} please add it in the proper location")
        continue
    new_file_name = re.findall('(\w+)\.pth$', model_path)
    if not new_file_name:
        print(f"Not a correct model path: {model_path}")
        continue
    new_file_name = new_file_name[0]
    with open(os.path.join('tests/test_utils/configs', f'{new_file_name}-config.yaml'), 'w') as output_file:
        output_file.write(f"stockfish_path: {STOCKISH_PATH}\n")
        output_file.write(f"model_path: {model_path}\n")
        output_file.write(f"preprocessing_parameters_path: {preprocessing_parameters_path}")