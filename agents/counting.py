"""Hi-Lo true-count helpers from remaining-rank count vectors."""

from __future__ import annotations

from config import NUM_DECKS
from game import ShoeObservation

# Tags aligned with count_vector order [A, 2, 3, 4, 5, 6, 7, 8, 9, 10].
HI_LO_TAGS: tuple[int, ...] = (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1)


def full_shoe_count_vector(num_decks: int = NUM_DECKS) -> tuple[int, ...]:
    """Remaining counts for an unshuffled complete shoe."""
    if num_decks <= 0:
        raise ValueError("num_decks must be positive")
    return tuple([4 * num_decks] * 9 + [16 * num_decks])


def running_count(
    count_vector: tuple[int, ...] | list[int],
    *,
    num_decks: int = NUM_DECKS,
) -> float:
    """Hi-Lo running count implied by cards already dealt from the shoe."""
    if len(count_vector) != 10:
        raise ValueError("count_vector must have length 10")

    full = full_shoe_count_vector(num_decks)
    total = 0.0
    for remaining, initial, tag in zip(count_vector, full, HI_LO_TAGS):
        dealt = initial - remaining
        if dealt < 0:
            raise ValueError("count_vector exceeds a full shoe for this num_decks")
        total += tag * dealt
    return total


def true_count(
    count_vector: tuple[int, ...] | list[int],
    cards_remaining: int | None = None,
    *,
    num_decks: int = NUM_DECKS,
) -> float:
    """Hi-Lo true count = running count / decks remaining."""
    remaining = (
        int(cards_remaining)
        if cards_remaining is not None
        else int(sum(count_vector))
    )
    if remaining <= 0:
        return 0.0
    decks_remaining = remaining / 52.0
    return running_count(count_vector, num_decks=num_decks) / decks_remaining


def true_count_from_shoe(
    shoe: ShoeObservation,
    *,
    num_decks: int = NUM_DECKS,
) -> float:
    """True count from a pre-deal :class:`ShoeObservation`."""
    return true_count(
        shoe.count_vector,
        shoe.cards_remaining,
        num_decks=num_decks,
    )
