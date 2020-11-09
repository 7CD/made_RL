import gym
from gym import spaces
from gym.utils import seeding
import random

def cmp(a, b):
    return float(a > b) - float(a < b)


polovinki_scores = {1: -1, 2: 0.5, 3: 1, 4: 1, 5: 1.5,\
                    6: 1, 7: 0.5, 8: 0, 9: -0.5, 10: -1}


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
exposed_cards_score = 0

def draw_card():
    card = deck.pop()
    global exposed_cards_score
    exposed_cards_score += polovinki_scores[card]
    return card


def draw_hand():
    return [draw_card(), draw_card()]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, counting_score_thresh, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        global exposed_cards_score
        exposed_cards_score = 0
        self.dealer = []
        self.counting_score_thresh = counting_score_thresh
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        elif action == 2:  # double: players bet is doubled, player takes last card
            self.player.append(draw_card())
            done = True
            if is_bust(self.player):
                reward = -1.
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card())
                reward = cmp(score(self.player), score(self.dealer))
                if self.natural and is_natural(self.player) and reward == 1.:
                    reward = 1.5
            reward *= 2.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        global exposed_cards_score
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player),\
                exposed_cards_score >= self.counting_score_thresh)

    def reset(self):
        global deck
        global exposed_cards_score
        if len(deck) < 15:
            deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
            random.shuffle(deck)
            exposed_cards_score = 0
        if len(self.dealer) > 1: 
            exposed_cards_score += polovinki_scores[self.dealer[1]]
        self.dealer = draw_hand()
        # one dealers card is hidden till the end of the play
        exposed_cards_score -= polovinki_scores[self.dealer[1]]
        self.player = draw_hand()
        return self._get_obs()