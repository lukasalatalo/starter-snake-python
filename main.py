# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
from gamestate import GameState
from mcts import MCTS


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "lukaschris",
        "color": "#FFFF00",
        "head": "default",
        "tail": "default",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    body = [list(coord.values()) for coord in game_state['you']['body']]
    food = [list(coord.values()) for coord in game_state['board']['food']]
    hazards = [
        list(coord.values()) for coord in game_state['board']['hazards']
    ]
    print("----------------------")
    width = game_state['board']['width']
    height = game_state['board']['height']
    my_snake_id = game_state['you']['id']
    health = game_state['you']['health']
    snakes = game_state['board']['snakes']
    gs = GameState(body, food, hazards, width, height, my_snake_id, snakes, health, game_state['you'])
    mcts = MCTS(gs, game_state['game']['timeout'])

    next_move = mcts.MCTSSearch()
    if next_move == '':

        is_move_safe = {"up": True, "down": True, "left": True, "right": True}
        my_head = game_state["you"]["body"][0]  # Coordinates of your head
        my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"
        if my_neck["x"] < my_head[
                "x"]:  # Neck is left of head, don't move left
            is_move_safe["left"] = False

        elif my_neck["x"] > my_head[
                "x"]:  # Neck is right of head, don't move right
            is_move_safe["right"] = False

        elif my_neck["y"] < my_head[
                "y"]:  # Neck is below head, don't move down
            is_move_safe["down"] = False

        elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
            is_move_safe["up"] = False

        safe_moves = []
        for move, isSafe in is_move_safe.items():
            if isSafe:
                safe_moves.append(move)
        
        if len(safe_moves) == 0:
            print(
                f"MOVE {game_state['turn']}: No safe moves detected! Moving down"
            )
            
            return {"move": "down"}

        # Choose a random move from the safe ones
        next_move = random.choice(safe_moves)
    print(f"MOVE {game_state['turn']}: {next_move}")



    print_reward_map1(mcts)
    return {"move": next_move}
    
def print_reward_map1(self):
    print("Reward Map:")
    for row in self.reward_map:
        row_str = "|"
        for value in row:
            if value < -50000:
                row_str += " X |"
            elif value < 0:
                row_str += " S |"
            elif value > 1:
                row_str += " O |"
            else:
                row_str += "   |"
        print(row_str)
# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
