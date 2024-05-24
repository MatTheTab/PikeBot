"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult, Limit
from lib.config import Configuration
from lib.engine_wrapper import COMMANDS_TYPE, OPTIONS_TYPE
from lib.engine_wrapper import MinimalEngine, MOVE
from typing import Any
import logging
from lib.model import Game

import sys
import os
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.chess_utils import ChessBot, Uniform_model, mean_aggr, max_aggr
from utils.pikeBot_chess_utils import Pikebot, PikeBotModelWrapper



# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


    
class PikeBotEngine(ExampleEngine):
    """Get a random move."""

    def __init__(self, commands: list[str], options: dict[str, Any], stderr: int | None, draw_or_resign: Configuration, game: Game | None = None, name: str | None = None, **popen_args: str) -> None:
        super().__init__(commands, options, stderr, draw_or_resign, game, name, **popen_args)
        self.opponent_elo = 1500

        if game:
            if game.opponent_color == "white":
                self.opponent_elo = game.white.rating
            else:
                self.opponent_elo = game.black.rating

            with open('pikeBot-config.yaml') as config_file:
                config = yaml.safe_load(config_file)

                stockfish_path = config['stockfish_path']
                model_path = config["model_path"]
                preprocessing_parameters_path = config["preprocessing_parameters_path"]

            model = PikeBotModelWrapper(model_path, preprocessing_parameters_path)
            self.chessBot = Pikebot( model=model, 
                                    aggregate=mean_aggr,
                                    stockfish_path=stockfish_path,
                                    color=game.my_color,
                                    opponents_elo = self.opponent_elo
                                    )
            

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        print(args)
        return PlayResult(self.chessBot.get_best_move(board), None)