"""Custom exceptions for the Monopoly game engine.

This module defines all exception types used throughout the game engine.
"""


class MonopolyError(Exception):
    """Base exception for all Monopoly game errors."""

    pass


class InvalidActionError(MonopolyError):
    """Raised when an action fails validation."""

    pass


class InsufficientFundsError(MonopolyError):
    """Raised when a player doesn't have enough money."""

    pass


class GameOverError(MonopolyError):
    """Raised when trying to perform actions after the game has ended."""

    pass


class InvalidPlayerError(MonopolyError):
    """Raised when an invalid player ID is used."""

    pass


class InvalidPropertyError(MonopolyError):
    """Raised when an invalid property position is used."""

    pass


class NotYourTurnError(MonopolyError):
    """Raised when a player tries to act out of turn."""

    pass


class PropertyNotOwnedError(MonopolyError):
    """Raised when trying to operate on an unowned property."""

    pass


class PropertyAlreadyOwnedError(MonopolyError):
    """Raised when trying to buy an already-owned property."""

    pass


class CannotBuildError(MonopolyError):
    """Raised when building is not allowed on a property."""

    pass


class CannotMortgageError(MonopolyError):
    """Raised when mortgaging is not allowed."""

    pass


class BankruptcyError(MonopolyError):
    """Raised when a player goes bankrupt."""

    pass
