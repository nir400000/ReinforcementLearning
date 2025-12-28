import random
import numpy as np

class DiceBlackjack:
    DICE = 6
    THRESHOLD = 11
    HIT = 1
    STAY = 0
    LOST = -1


    def __init__(self):
        self.sum = 0
        self.status = self.HIT


    def roll_dice(self):
        return random.randint(1, 6)


    def make(self, move: int):
        if move == self.STAY:
            #print("Stay")
            self.status = self.STAY
            return

        else:
            #print("Hit")
            self.sum += self.roll_dice()
            if self.sum > self.THRESHOLD:
                self.sum = 0
                self.status = self.LOST
            elif self.sum == self.THRESHOLD:
                self.status = self.STAY
        return


    def clone(self):
        c = DiceBlackjack()
        c.sum = self.sum
        c.status = self.status

    def is_game_ended(self):
        if self.status == self.HIT:
            return False
        return True

class QTable:
    def __init__(self, threshold=11, epsilon=0.1, training_episodes=10000, evaluation_episodes=1000):
        self.threshold = threshold
        self.epsilon = epsilon
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        # States: possible sums (0 to threshold)
        # Actions: 0=STAY, 1=HIT
        self.n_states = threshold + 1
        self.n_actions = 2
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.N = np.zeros_like(self.q_table)

    def choose_action(self, state, evaluate = False):
        if evaluate == False and np.random.rand() < self.epsilon :
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def train(self):
        for episode in range(self.training_episodes):
            game = DiceBlackjack()
            state = game.sum
            while not game.is_game_ended():
                action = self.choose_action(state)
                game.make(action)

                # Q-learning update
                self.N[state,action] += 1
                n = self.N[state,action]
                g = self.q_table[state, action]
                self.q_table[state, action] += (1/n) * (game.sum - g)
                state = game.sum



    def evaluate(self):
        total_reward = 0
        for _ in range(self.evaluation_episodes):
            game = DiceBlackjack()
            state = game.sum
            while not game.is_game_ended():
                action = self.choose_action(state, evaluate= True)
                game.make(action)
                state = game.sum
            total_reward += game.sum
        return total_reward/self.evaluation_episodes

    def print_policy(self):
        print("Optimal policy (sum: action):")
        for s in range(self.n_states):
            action = np.argmax(self.q_table[s])
            print(f"Sum {s}: {'HIT' if action == 1 else 'STAY'}")


def main_play():
    print("Welcome to Dice blackjack!\n")
    while(True):
        print("initializing new game:\n")
        game = DiceBlackjack()
        while not game.is_game_ended():
            move = int(input("Press 0 to stay or 1 to hit : "))
            game.make(move)
            print(game.sum,game.is_game_ended())

        print("game ended with ",game.sum," points")

def main_QTable():
    print("Training Q-learning agent for Dice Blackjack...")
    q_agent = QTable(training_episodes=20000, evaluation_episodes=100)
    q_agent.train()
    print("\nEvaluating learned policy...")
    avg_reward = q_agent.evaluate()
    print("\nLearned policy:")
    q_agent.print_policy()
    print("average reward:", avg_reward)


if __name__ == "__main__":
    # main_play()
    main_QTable()
"""Optimal policy (sum: action):
Sum 0: HIT
Sum 1: HIT
Sum 2: HIT
Sum 3: HIT
Sum 4: HIT
Sum 5: HIT
Sum 6: HIT
Sum 7: STAY
Sum 8: STAY
Sum 9: STAY
Sum 10: STAY
Sum 11: STAY
average reward: 7.86"""
