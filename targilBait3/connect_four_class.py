# -*- coding: utf-8 -*-
"""Connect four
    ID: 208246090
    Name: Nir Ben Boaz

"""
import math
import random


class ConnectFour:
    # Some constants that I want to use to fill the board.
    RED = 1
    YELLOW = -1
    EMPTY = 0

    # Game status constants
    RED_WIN = 1
    YELLOW_WIN = -1
    DRAW = 0
    ONGOING = 2

    def __init__(self):
        self.board = [[self.EMPTY for _ in range(6)] for _ in range(7)]
        self.heights = [0 for _ in range(7)]  # The column heights in the board.
        self.player = self.RED
        self.status = self.ONGOING

    def legal_moves(self):
        return [i for i in range(7) if self.heights[i] < 6]

    def make(self, move):  # Assumes that 'move' is a legal move
        self.board[move][self.heights[move]] = self.player
        self.heights[move] += 1
        # Check if the current move results in a winner:
        if self.winning_move(move):
            self.status = self.player
        elif len(self.legal_moves()) == 0:
            self.status = self.DRAW
        else:
            self.player = self.other(self.player)

    def other(self, player):
        return self.RED if player == self.YELLOW else self.YELLOW

    def unmake(self, move):
        self.heights[move] -= 1
        self.board[move][self.heights[move]] = self.EMPTY
        self.player = self.other(self.player)
        self.status = self.ONGOING

    def clone(self):
        clone = ConnectFour()
        clone.board = [col[:] for col in self.board]  # Deep copy columns
        clone.heights = self.heights[:]  # Deep copy heights
        clone.player = self.player
        clone.status = self.status
        return clone

    def winning_move(self, move):
        # Checks if the move that was just made wins the game.
        # Assumes that the player who made the move is still the current player.

        col = move
        row = self.heights[col] - 1  # Row of the last placed piece
        player = self.board[col][row]  # Current player's piece

        # Check all four directions: horizontal, vertical, and two diagonals
        # Directions: (dx, dy) pairs for all 4 possible win directions
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 0
            x, y = col + dx, row + dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = col - dx, row - dy
            while 0 <= x < 7 and 0 <= y < 6 and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 3:
                return True
        return False


    def immediate_win(self):
        legal = self.legal_moves()

        for move in legal:
            clone = self.clone()
            clone.make(move)
            if clone.status == self.player:
                # print(f"Instant win found at column {move}")
                return move
        return None


    def immediate_threat(self):
        legal = self.legal_moves()

        opponent = self.other(self.player)
        for move in legal:
            clone = self.clone()
            clone.player = opponent
            clone.make(move)
            if clone.status == opponent:
                return move
        return None


    def __str__(self):
        """
        Returns a string representation of the board.
        'R' for RED, 'Y' for YELLOW, '.' for EMPTY.
        """
        rows = []
        for r in range(5, -1, -1):  # From top row to bottom
            row = []
            for c in range(7):
                if self.board[c][r] == self.RED:
                    row.append("R")
                elif self.board[c][r] == self.YELLOW:
                    row.append("Y")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)


class MCTSNode:

    def __init__(self, p, parent=None, move=None):
        self.p = p
        self.parent = parent
        self.move = move
        self.untried_moves = p.legal_moves()
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_game_ends(self):
        return self.p.status != self.p.ONGOING

    def is_fully_expand(self):
        return len(self.untried_moves) == 0

    def utc(self, c=1.4):  ## Tzur said c = sqrt(2) is best, dont know why
        if self.visits == 0:
            return float('inf')
        return abs(self.wins) / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        ##  uses utc to find the best child inside children list
        children_weights = [child.utc() for child in self.children]
        max_child_weight = max(children_weights)
        max_child_idx = children_weights.index(max_child_weight)
        return self.children[max_child_idx]

    def add_child(self, move, p):
        # FIX: Instantiate the class correctly
        child = MCTSNode(p, parent=self, move=move)

        if move in self.untried_moves:
            self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1

        # Who made the move that led to this node?
        # The state 'self.p' is waiting for the NEXT player.
        # So the previous player (who made 'self.move') is other(self.p.player).
        mover = self.p.other(self.p.player)

        if result == mover:
            self.wins += 1  # The move led to a win for the mover
        elif result == self.p.DRAW:
            self.wins += 0.5  # A draw is better than a loss


class MCTSPlayer:


    def choose_move(self, game, iterations):
        # win/lose Heuristics
        #  check for immediate win
        move = game.immediate_win()
        # check for immediate threat
        if move is None:
            move = game.immediate_threat()

        if move is not None:
            return move

        root = MCTSNode(game)

        for _ in range(iterations):
            node = self.selection(root)
            if not node.is_game_ends():
                node = self.expansion(node)
            result = self.simulation(node)
            self.backpropagation(node, result)

        if not root.children:
            return random.choice(game.legal_moves())

        most_visited_child = max(root.children, key=lambda child: child.visits)
        return most_visited_child.move

    def selection(self, node):
        while node.is_fully_expand() and not node.is_game_ends():
            node = node.best_child()
        return node

    def expansion(self, node):
        move = random.choice(node.untried_moves)
        clone = node.p.clone()
        clone.make(move)
        return node.add_child(move, clone)

    def simulation(self, node):
        clone = node.p.clone()
        while clone.status == clone.ONGOING:
            move = random.choice(clone.legal_moves())
            clone.make(move)
        return clone.status

    def backpropagation(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent


# Main function for debugging purposes: Allows two humans to play the game.
def main():
    """
    Runs the game: Human (Red) vs AI (Yellow).
    """
    game = ConnectFour()
    ai_player = MCTSPlayer()

    # Configuration
    AI_COLOR = game.RED
    HUMAN_COLOR = game.YELLOW
    AI_ITERATIONS = 50000

    print("Welcome to Connect Four!")
    print("The AI is RED (R).  You are YELLOW (Y). ")
    print("---------------------------------------")

    while game.status == game.ONGOING:
        print("\n" + str(game))

        if game.player == HUMAN_COLOR:
            # Human's Turn
            print("\n[Your Turn]")
            try:
                move = int(input("Enter a column (0-6): "))
                if move not in game.legal_moves():
                    print("Illegal move. Try again.")
                    continue
                game.make(move)
            except ValueError:
                print("Invalid input. Enter a number between 0 and 6.")
            except IndexError:
                print("Move out of bounds. Try again.")

        else:
            # AI's Turn
            print(f"\n[AI Turn] Thinking ({AI_ITERATIONS} iterations)...")
            move = ai_player.choose_move(game, AI_ITERATIONS)
            print(f"AI chose column {move}")
            game.make(move)

    # Game Over
    print("\n" + str(game))
    if game.status == game.RED:
        print("\nRED  wins!")
    elif game.status == game.YELLOW:
        print("\nYELLOW  wins!")
    else:
        print("\nIt's a draw!")


if __name__ == "__main__":
    main()