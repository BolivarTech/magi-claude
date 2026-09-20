# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-09-19
"""Plan how a bundle that does not fit magi-rs's input ceiling is split (MS4b).

The ceiling is a transport fact measured in MS0: ``consult`` rejects any stdin
above 262 144 bytes, instantly and fail-closed. Everything in this module is
PURE -- it decides, it never reads a file, launches a process or renders a
byte -- so that the decision can be recorded before the first evaluation and
checked by a reader without re-running anything (CLAUDE.local.md §6.2).

The four rules it codifies:

1. A bundle that fits under the HEADROOM is not segmented: segmentation is a
   fallback for a context limit, never the default mode.
2. The ceiling is measured in BYTES (UTF-8), and no segment is filled past
   ``HEADROOM`` of it -- the system prompt and the model's reasoning live in
   the same window.
3. Segments are cut at section boundaries only, in the bundle's order; a
   section that alone exceeds the headroom is REFUSED, never truncated.
4. The plan is data (``as_map``), written before anything runs.

Example:
    >>> plan = BundleSegmenter.plan(sections, CONSULT_INPUT_CEILING_BYTES)
    >>> if plan.is_segmented:
    ...     write_map(plan.as_map())
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: The stdin ceiling of ``magi-rs consult``, inclusive, measured in MS0
#: (262 145 bytes is rejected with ``input_invalid``).
CONSULT_INPUT_CEILING_BYTES = 262_144

#: The fraction of the ceiling a segment may fill (§6.2.2: ~60-70 %, measured in
#: bytes). The remainder is headroom for the system prompt, the model's own
#: reasoning and tokenisation slack. Inclusive: a bundle exactly AT the bound fits.
HEADROOM = 0.65


class SegmentTooLargeError(Exception):
    """One section alone exceeds the headroom and cannot be split safely.

    A SISTER of ``Exception``, never of ``ValidationError``: fail-closed, and a
    retry does not make the section smaller. The message names the section,
    its size and the bound it must fit under, so the operator knows what to
    split.
    """


@dataclass(frozen=True)
class Segment:
    """One self-contained slice of a bundle, evaluated as its own gate run.

    Attributes:
        order: 1-based position in the deterministic evaluation order.
        section_names: The sections it holds, in the bundle's original order.
        size_bytes: Sum of each section's ``body`` encoded as UTF-8, PLUS
            ``overhead_bytes`` per section. Body-only would understate what is
            actually sent.
        defines: Shared interface names this segment OWNS (§6.2.3), taken from
            each section's own ``defines`` list. The segmenter does not infer
            ownership -- the bundle builder declares it.
        references: Names defined by OTHER sections that this segment's bodies
            mention. The datum an integration pass is decided on: a split with
            a definer on one side and a reference on the other cut through a
            contract.
    """

    order: int
    section_names: list[str]
    size_bytes: int
    defines: list[str]
    references: list[str]


@dataclass(frozen=True)
class SegmentPlan:
    """The decision, taken BEFORE anything is evaluated.

    Attributes:
        segments: The segments in evaluation order.
        is_segmented: False when one segment holds everything (or nothing).
        ceiling_bytes: The transport ceiling the plan was computed against.
        headroom: The fraction of it a segment may fill.
    """

    segments: list[Segment]
    is_segmented: bool
    ceiling_bytes: int
    headroom: float

    def as_map(self) -> dict[str, Any]:
        """The map recorded in ``magi-report.json`` before the first evaluation.

        Returns:
            ``{"segments": [{"order", "section_names", "size_bytes", "defines",
            "references"}, ...], "ceiling_bytes": int, "headroom": float}`` --
            every number the reader needs to check the split without
            re-running the segmenter.
        """
        return {
            "segments": [
                {
                    "order": segment.order,
                    "section_names": list(segment.section_names),
                    "size_bytes": segment.size_bytes,
                    "defines": list(segment.defines),
                    "references": list(segment.references),
                }
                for segment in self.segments
            ],
            "ceiling_bytes": self.ceiling_bytes,
            "headroom": self.headroom,
        }


class BundleSegmenter:
    """The pure planner: decides whether to segment, and how."""

    @staticmethod
    def plan(
        sections: list[dict[str, Any]],
        ceiling_bytes: int,
        overhead_bytes: int = 0,
    ) -> SegmentPlan:
        """Decide whether to segment, and how.

        Greedy, in bundle order: sections are appended to the current segment
        while it stays under the headroom, and a new segment is opened at the
        first section that would cross it. Order is the bundle's own, so the
        plan is deterministic for the same input.

        Args:
            sections: ``{"name": str, "body": str, "defines": list[str]}`` each.
            ceiling_bytes: The measured transport ceiling (262 144, from MS0).
            overhead_bytes: Bytes the RENDERER adds per section -- its heading
                and separator. Passed IN rather than assumed, so this function
                stays pure and never has to know how the bundle is formatted.
                Zero is the honest default: a caller that does not measure its
                own overhead should not have one invented for it.

        Returns:
            The plan. ``is_segmented`` is False when one segment holds
            everything.

        Raises:
            SegmentTooLargeError: One section alone exceeds the headroom.

        Complexity:
            O(n·d) in the number of sections times the number of declared
            names (the reference scan is a substring search per name), single
            pass, once per run.
        """
        bound = int(ceiling_bytes * HEADROOM)
        sizes = [len(str(s["body"]).encode("utf-8")) + overhead_bytes for s in sections]
        for section, size in zip(sections, sizes):
            if size > bound:
                raise SegmentTooLargeError(
                    f"section {section['name']!r} is {size} bytes on its own, over the "
                    f"{bound}-byte headroom ({HEADROOM:.0%} of the {ceiling_bytes}-byte "
                    "ceiling); split it before running -- it is never truncated"
                )
        owners = {name: s["name"] for s in sections for name in s.get("defines", ())}

        groups: list[list[int]] = []
        current: list[int] = []
        current_size = 0
        for index, size in enumerate(sizes):
            if current and current_size + size > bound:
                groups.append(current)
                current, current_size = [], 0
            current.append(index)
            current_size += size
        if current:
            groups.append(current)

        segments = [
            Segment(
                order=order,
                section_names=[sections[i]["name"] for i in group],
                size_bytes=sum(sizes[i] for i in group),
                defines=[n for i in group for n in sections[i].get("defines", ())],
                references=_references(sections, group, owners),
            )
            for order, group in enumerate(groups, start=1)
        ]
        return SegmentPlan(
            segments=segments,
            is_segmented=len(segments) > 1,
            ceiling_bytes=ceiling_bytes,
            headroom=HEADROOM,
        )


def _references(
    sections: list[dict[str, Any]], group: list[int], owners: dict[str, str]
) -> list[str]:
    """Names owned by sections OUTSIDE ``group`` that a body inside it mentions.

    Args:
        sections: All sections.
        group: Indexes of the sections in the segment.
        owners: Declared name -> the section that defines it.

    Returns:
        The referenced names, in declaration order, without duplicates.
    """
    inside = {sections[i]["name"] for i in group}
    bodies = "\n".join(str(sections[i]["body"]) for i in group)
    return [
        name for name, owner in owners.items() if owner not in inside and mentions(name, bodies)
    ]


def mentions(name: str, text: str) -> bool:
    """Whether ``text`` names ``name`` as a whole word.

    Plain containment would let ``Pin`` match ``Pinout`` or ``opinion`` -- and
    a false reference buys a real, paid integration run for nothing.

    Args:
        name: A declared interface name.
        text: The text to search.

    Returns:
        True when the name appears delimited by word boundaries.
    """
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None
