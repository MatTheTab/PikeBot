# PikeBot
Repository for an AI-based chess bot designed to exploit human gameplay weaknesses, set up traps, and play much more aggressively than the vast majority of the currently available chess bots.
PikeBot utilizes score evaluations generated by Stockfish to make intelligent, human-like decisions, resulting in more dynamic and exciting chess games. 
What sets PikeBot apart from other chess bots is its commitment to utilizing human strategic weaknesses rather than pursuing optimal gameplay, leading to more engaging gameplay.

## Play with PikeBot
https://lichess.org/@/PikeBot

## Test your bot recognition skills
https://site-ngqt.onrender.com

## Example Data
https://drive.google.com/drive/folders/1ZiqnOgf4XJErX2aGR9IvxpPPwCLDET2g?usp=sharing <br>

---

## Developer Notes
 - Data -> raw PGN data, needs to be converted to .npy files <br>
 - Preprocessed_Data -> data read from PGN files <br>
 - Model_Data -> Data which can be directly fed into the model, normalized/standardized and generally cleaned data from Preprocessed_Data <br>

Project directory structure:
```
Pikebot/
├── Data/
│   ├── Train/
│   ├── Test/
│   └── Val/
├── Preprocessed_Data/
│   ├── Train/
│   ├── Test/
│   └── Val/
├── Model_Data/
│   ├── Metadata/
│   ├── Train/
│   ├── Test/
│   └── Val/
├── Small_Data/
│   ├── Train/
│   ├── Test/
│   └── Val/
├── Small_Preprocessed_Data/
│   ├── Train/
│   ├── Test/
│   └── Val/
├── stockfish/
├── Generators/
├── utils/
│   ├── data_utils
│   └── chess_utils
├── Models/
│   ├── PikeBot_Models/
│   ├── Pretrained/
│   └── Training_Results/
└── *All Notebooks*
```
Small_Data is for smaller PGN files (~100 000 games) and can be used for testing some ideas without running the code on bigger PGN files. Directories with 'Preprocessed' in the name show output of data-saving functions, i.e. effects of transformation from PGN files to .txt, .csv and .npy files, for more info see documentation of data_utils functions and Data_reading_tutorial.ipynb file.

Default dataframe columns explanation:

| Column Name | Explanation | Expected Values |
| --------------- | --------------- | --------------- |
| human    | if the move was made by a human or randomly, expected model output   | True/False  |
| player    | name of the player or "bot" if the move was made by a bot    | name of the player like: "cixon123"    |
| elo    | player's ELO score   | integer value, e.g. 1397    |
| color    | color of the player    | "White" or "Black"   |
| event    | event of the game    |  Event as defined in Lichess, e.g. Rated Classical game   |
| clock    | how much time is left in the game    | floating point number representing time left in a game, e.g. 64.0    |
| stockfish_score_depth_{depth}    | score for a currently playing player judged using stockfish of chosen depth in {depth}    | Stockfish score as int, e.g. 744    |
| stockfish_difference_depth_{move_num}    | difference between last human-made position as evaluated by stockfish and the position after making a move, i.e. how advantageous/disadvantageous a move was    |  Stockfish score as int, e.g. 744   |
| past_move_{move_num}    | past move saved as bitboard to provide context for the model    | bitboard representing board state from past turns, shaped (76, 8, 8)    |
| current_move    | currently performed move, expected input to the model along with optional context from previous columns    | bitboard representing board state after the current human/bot move, shaped (76, 8, 8)    |
| current_move_str    | Forsyth–Edwards Notation (FEN) representation of the current board state    | String representing the board state (example shortened here) - , e.g. r2q1rk1/...    |

