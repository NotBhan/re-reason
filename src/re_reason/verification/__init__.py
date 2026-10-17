"""Verification package contracts and transaction commit gate."""

from re_reason.verification.base import BaseVerifier
from re_reason.verification.transaction_gate import (
    CandidateRecord,
    DuplicateCandidateError,
    InvalidCandidateStateError,
    InvalidRankingScoreError,
    TransactionCommitGate,
    TransactionDecision,
    TransactionGateError,
    TransactionOutcome,
    UnverifiedCandidateError,
    VerificationOutcome,
)

__all__ = [
    "BaseVerifier",
    "CandidateRecord",
    "DuplicateCandidateError",
    "InvalidCandidateStateError",
    "InvalidRankingScoreError",
    "TransactionCommitGate",
    "TransactionDecision",
    "TransactionGateError",
    "TransactionOutcome",
    "UnverifiedCandidateError",
    "VerificationOutcome",
]
