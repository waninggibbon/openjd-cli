# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Shared helpers for the STEP_APPROVALS extension (RFC 0200, Template Schemas §3.7).

A Step (or Step Template) may have an `approval`: a person must approve it before
any of its Tasks run. A Step with an `approval` and no `script` is an approval-only
Step, which runs nothing and completes successfully once it is approved.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from openjd.model.v2023_09 import StepApproval

# Unicode bidirectional formatting characters. In a terminal they can reorder the
# text around them, for example to make a link target read as something else.
_BIDI_CONTROLS = frozenset(
    chr(code) for code in (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A))
)


def step_approval(step: Any) -> Optional["StepApproval"]:
    """The `approval` of a Job's Step or of a Step Template, or None if it has none.

    Read with getattr because openjd-model releases without STEP_APPROVALS have no
    `approval` field. Those releases reject templates that use the extension, so a
    Step from them never has an approval.
    """
    return getattr(step, "approval", None)


def is_approval_only(step: Any) -> bool:
    """Whether a Job's Step or a Step Template is approval-only: it has an `approval`
    and nothing to run."""
    return step_approval(step) is not None and bool(getattr(step, "is_approval_only", False))


def terminal_safe(text: str) -> str:
    """Returns template-supplied text in a form that is safe to write to a terminal.

    Approval messages, link labels and link targets come from the template author
    (and from Job Parameter values), and the person reading them is asked to trust
    them (RFC 0200, Security Considerations). Control characters could move the
    cursor, recolor or overwrite earlier output, and bidirectional formatting
    characters could reorder it, so both are shown as escape sequences instead.
    Newlines and tabs are kept, and a CRLF line ending becomes a newline.
    """
    text = text.replace("\r\n", "\n")
    return "".join(
        (
            ch.encode("unicode_escape").decode("ascii")
            if ch not in "\n\t" and (unicodedata.category(ch) == "Cc" or ch in _BIDI_CONTROLS)
            else ch
        )
        for ch in text
    )


def format_approval(
    *,
    message: str,
    links: Iterable[tuple[str, str]] = (),
    expires_after_seconds: Optional[int] = None,
    indent: str = "",
    expiry_note: str = "",
) -> list[str]:
    """Formats a resolved approval for a person to read, as terminal-safe lines that
    each start with `indent`: the message, each link's label and full target (not
    only its label, as RFC 0200 recommends), and the expiry if the template sets
    one, followed by `expiry_note`.

    The message loses the trailing blank lines that a YAML block scalar leaves.
    """
    lines = [f"{indent}Message:"]
    lines.extend(f"{indent}  {line}" for line in terminal_safe(message).rstrip().splitlines())
    links = list(links)
    if links:
        lines.append(f"{indent}Links:")
        lines.extend(
            f"{indent}  - {terminal_safe(label)}: {terminal_safe(target)}"
            for label, target in links
        )
    if expires_after_seconds is not None:
        lines.append(f"{indent}Expires after: {expires_after_seconds} seconds{expiry_note}")
    return lines


def format_step_approval(approval: "StepApproval", **kwargs: Any) -> list[str]:
    """:func:`format_approval` for a Job Step's resolved approval."""
    return format_approval(
        message=approval.message,
        links=[(link.label, link.target) for link in approval.links or []],
        expires_after_seconds=approval.expiresAfterSeconds,
        **kwargs,
    )
