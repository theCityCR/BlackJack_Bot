from agents.rule_agent.rule_agent import RuleAgent
from game import Action, BlackjackGame, GameState


def make_state(
    player_value,
    dealer_upcard,
    usable_ace=False,
    can_double=False,
    can_split=False,
    is_split_hand=False,
    active_hand_index=0,
    num_hands=1,
):
    return GameState(
        player_value=player_value,
        dealer_upcard=dealer_upcard,
        usable_ace=usable_ace,
        can_double=can_double,
        can_split=can_split,
        is_split_hand=is_split_hand,
        active_hand_index=active_hand_index,
        num_hands=num_hands,
    )


# =========================
# Hard totals
# =========================

def test_hard_16_hits_against_dealer_10():
    agent = RuleAgent()
    state = make_state(16, 10)

    assert agent.choose_action(state) == Action.HIT


def test_hard_16_stands_against_dealer_6():
    agent = RuleAgent()
    state = make_state(16, 6)

    assert agent.choose_action(state) == Action.STAND


def test_hard_12_hits_against_dealer_2():
    agent = RuleAgent()
    state = make_state(12, 2)

    assert agent.choose_action(state) == Action.HIT


def test_hard_12_stands_against_dealer_4():
    agent = RuleAgent()
    state = make_state(12, 4)

    assert agent.choose_action(state) == Action.STAND


def test_hard_17_stands():
    agent = RuleAgent()
    state = make_state(17, 10)

    assert agent.choose_action(state) == Action.STAND


# =========================
# Soft totals
# =========================

def test_soft_17_hits_when_double_not_available():
    agent = RuleAgent()
    state = make_state(17, 10, usable_ace=True)

    assert agent.choose_action(state) == Action.HIT


def test_soft_18_stands_against_dealer_6():
    agent = RuleAgent()
    state = make_state(18, 6, usable_ace=True)

    assert agent.choose_action(state) == Action.STAND


def test_soft_18_hits_against_dealer_10():
    agent = RuleAgent()
    state = make_state(18, 10, usable_ace=True)

    assert agent.choose_action(state) == Action.HIT


def test_soft_19_stands():
    agent = RuleAgent()
    state = make_state(19, 10, usable_ace=True)

    assert agent.choose_action(state) == Action.STAND


# =========================
# Double decisions
# =========================

def test_hard_11_doubles_when_allowed():
    agent = RuleAgent()
    state = make_state(11, 10, can_double=True)

    assert agent.choose_action(state) == Action.DOUBLE


def test_hard_10_doubles_against_dealer_9():
    agent = RuleAgent()
    state = make_state(10, 9, can_double=True)

    assert agent.choose_action(state) == Action.DOUBLE


def test_hard_10_does_not_double_against_dealer_10():
    agent = RuleAgent()
    state = make_state(10, 10, can_double=True)

    assert agent.choose_action(state) == Action.HIT


def test_hard_9_doubles_against_dealer_3_to_6():
    agent = RuleAgent()

    for dealer in [3, 4, 5, 6]:
        state = make_state(9, dealer, can_double=True)
        assert agent.choose_action(state) == Action.DOUBLE


def test_soft_18_doubles_against_dealer_3_to_6():
    agent = RuleAgent()

    for dealer in [3, 4, 5, 6]:
        state = make_state(18, dealer, usable_ace=True, can_double=True)
        assert agent.choose_action(state) == Action.DOUBLE


def test_soft_18_does_not_double_against_dealer_2():
    agent = RuleAgent()
    state = make_state(18, 2, usable_ace=True, can_double=True)

    assert agent.choose_action(state) == Action.STAND


# =========================
# Split decisions
# =========================

def test_splits_aces():
    agent = RuleAgent()

    # Pair of aces has value 12 with usable ace.
    state = make_state(
        player_value=12,
        dealer_upcard=10,
        usable_ace=True,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.SPLIT


def test_splits_8s():
    agent = RuleAgent()

    state = make_state(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.SPLIT


def test_does_not_split_10s():
    agent = RuleAgent()

    state = make_state(
        player_value=20,
        dealer_upcard=6,
        usable_ace=False,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.STAND


def test_does_not_split_5s_prefers_double():
    agent = RuleAgent()

    state = make_state(
        player_value=10,
        dealer_upcard=6,
        usable_ace=False,
        can_double=True,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.DOUBLE


def test_splits_9s_against_dealer_9():
    agent = RuleAgent()

    state = make_state(
        player_value=18,
        dealer_upcard=9,
        usable_ace=False,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.SPLIT


def test_does_not_split_9s_against_dealer_7():
    agent = RuleAgent()

    state = make_state(
        player_value=18,
        dealer_upcard=7,
        usable_ace=False,
        can_split=True,
    )

    assert agent.choose_action(state) == Action.STAND


# =========================
# Full episode tests
# =========================

def test_play_episode_returns_numeric_reward():
    game = BlackjackGame()
    agent = RuleAgent()

    reward = agent.play_episode(game)

    assert isinstance(reward, (int, float))
    assert game.done is True


def test_play_episode_can_run_multiple_times():
    game = BlackjackGame()
    agent = RuleAgent()

    rewards = [agent.play_episode(game) for _ in range(100)]

    assert all(isinstance(reward, (int, float)) for reward in rewards)
    assert game.done is True


def test_play_episode_reward_can_include_blackjack_or_double_or_split_values():
    game = BlackjackGame()
    agent = RuleAgent()

    rewards = [agent.play_episode(game) for _ in range(500)]

    # Rewards are no longer only {-1, 0, 1} because:
    # - natural blackjack pays 1.5
    # - double can produce +/-2
    # - split can sum multiple hand rewards
    assert all(isinstance(reward, (int, float)) for reward in rewards)