# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""The approval gate of `openjd run` (STEP_APPROVALS, RFC 0200 "Local runs").

`openjd run` is a local, single-user runner, so it decides a Step's approval itself:

* a Step named with `--approve-step` is approved in advance;
* otherwise, with an interactive terminal on stdin, it asks there and waits;
* otherwise the approval counts as rejected, so a non-interactive run never
  approves anything.

The template's `expiresAfterSeconds` is not applied: RFC 0200 gives local runs no
expiry, and a prompt only waits as long as the person running the CLI does.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Iterable, Optional, TextIO

from ..._common._approvals import terminal_safe


class ApprovalDecision(str, Enum):
    """How a Step's approval gate was decided in a local run."""

    APPROVED_IN_ADVANCE = "APPROVED_IN_ADVANCE"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NO_TERMINAL = "NO_TERMINAL"

    @property
    def approved(self) -> bool:
        return self in (ApprovalDecision.APPROVED_IN_ADVANCE, ApprovalDecision.APPROVED)

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS = {
    ApprovalDecision.APPROVED_IN_ADVANCE: "approved in advance (--approve-step)",
    ApprovalDecision.APPROVED: "approved at the prompt",
    ApprovalDecision.REJECTED: "rejected at the prompt",
    ApprovalDecision.NO_TERMINAL: "not approved (no interactive terminal, and no --approve-step)",
}


def _is_interactive(stream: Optional[TextIO]) -> bool:
    # stdin is None under pythonw, and may be closed or detached.
    try:
        return stream is not None and stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


class StepApprovalGate:
    """Decides the approval gates of a local run.

    Args:
        approved_in_advance: The names of the Steps given with `--approve-step`.
        prompt_to_stderr: Write the prompt to stderr instead of stdout. Use it when
            stdout carries machine-readable output; the Session log, which holds the
            approval request, is then not printed, so the request is written along
            with the prompt.
    """

    def __init__(
        self, *, approved_in_advance: Iterable[str] = (), prompt_to_stderr: bool = False
    ) -> None:
        self._approved_in_advance = frozenset(approved_in_advance)
        self._prompt_to_stderr = prompt_to_stderr
        # The stream of the prompt that is waiting for an answer, if any.
        self._prompting_on: Optional[TextIO] = None

    def decide(self, step_name: str, request: list[str]) -> ApprovalDecision:
        """Decides the approval gate of the Step named `step_name`.

        Args:
            step_name: The name of the Step whose gate has been reached.
            request: The approval request, as lines of terminal-safe text.

        Raises:
            KeyboardInterrupt: If the person at the terminal interrupts the prompt.
        """
        if step_name in self._approved_in_advance:
            return ApprovalDecision.APPROVED_IN_ADVANCE
        # Look stdin up now, not when the gate is made, so that it is the stream in
        # use when the gate is reached.
        stdin = sys.stdin
        if not _is_interactive(stdin):
            return ApprovalDecision.NO_TERMINAL
        out = sys.stderr if self._prompt_to_stderr else sys.stdout
        if self._prompt_to_stderr:
            for line in request:
                print(line, file=out)
        self._prompting_on = out
        try:
            return self._ask(stdin, out, step_name)
        finally:
            self._prompting_on = None

    def end_prompt_line(self) -> None:
        """Ends the line of a prompt that is waiting for an answer, if there is one,
        so that what is written next (such as the log of an interruption) starts on a
        line of its own."""
        if self._prompting_on is not None:
            self._prompting_on.write("\n")
            self._prompting_on.flush()

    @staticmethod
    def _ask(stdin: TextIO, out: TextIO, step_name: str) -> ApprovalDecision:
        # There is no default answer: an approval is never given, and a rejection
        # never stops the Steps that depend on it, because of a stray Enter key.
        question = (
            f"Approve Step '{terminal_safe(step_name)}'? Enter 'y' to approve or 'n' to reject: "
        )
        while True:
            out.write(question)
            out.flush()
            line = stdin.readline()
            if not line:
                # End of input: nobody is left to approve.
                out.write("\n")
                return ApprovalDecision.REJECTED
            answer = line.strip().lower()
            if answer in ("y", "yes"):
                return ApprovalDecision.APPROVED
            if answer in ("n", "no"):
                return ApprovalDecision.REJECTED
            out.write("Please answer 'y' or 'n'.\n")
