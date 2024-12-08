import sys
import chess.pgn
from utils.utils import get_paths
from evaluation_utils import load_pgn, set_engine_history, get_move_evaluation_pairs
from utils.pikeBot_chess_utils import PikeBotModelWrapper, Pikebot

def add_move_evaluation(game: chess.pgn.Game, engine: Pikebot, model: PikeBotModelWrapper):
    set_engine_history(engine, [])

    node = game
    is_white = "pikebot" in game.headers["White"].lower()
    is_black = "pikebot" in game.headers["Black"].lower()

    if not is_black and not is_white:
        print("Can't recognize Pikebot")
        sys.exit(1)

    if is_black and is_white:
        print("Both players have 'PikeBot' in their name")
        sys.exit(1)
    try:
        if is_white:
            elo = game.headers["BlackElo"]
        else:
            elo = game.headers["WhiteElo"]
        elo = int(elo)
    except:
        elo = 1100

    move_number = 1
    board = game.board()
    engine.save_to_history(board)

    evaluations = {board.fen(): -1}
    moves = {board.fen(): chess.Move.from_uci('e2e4')}
    for move in board.legal_moves:
        board.push(move)
        evaluations[board.fen()] = -1
        moves[board.fen()] = move
        board.pop()
    all_predictions = list()
    while node.variations:
        next_node = node.variations[0] 
        move = next_node.move
        board.push(move)
        engine.save_to_history(board)
        if is_white == board.turn:
            evaluations, moves = get_move_evaluation_pairs(engine, elo, board, model)
            all_predictions.extend(((p[0], p[1], moves[p[0]].uci()) for p in evaluations.items()))
        else:
            predictions = [(key, value) for key, value in evaluations.items()]
            predictions.sort(key=lambda x: -x[1])
            predictions = [(moves[p[0]].uci(), p[1]) for p in predictions]
            comment = f"Prediction for the move {evaluations[board.fen()]:.3f}, top 5 moves: {predictions[:5]}"
            node.comment = comment

        node = next_node
        move_number += 1

    # Print the annotated game
    return str(game), all_predictions
