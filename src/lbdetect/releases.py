"""Manually curated timeline of LLM and coding-assistant releases.

These are *annotations*, never labels. Adoption lags release by weeks to months,
so alignment code treats each entry as a weak prior with a wide window.

`weight` is a rough guess at how much developer-facing text the release touched
(reach x how much it changed default phrasing), on a 0-1 scale. It only ever
re-weights a ranking; no result should depend on the exact numbers.

`generation` groups releases that land close enough together that this project
does not try to separate them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Release:
    date: str  # YYYY-MM-DD
    name: str
    kind: str  # chat | api | coding_assistant | agent
    weight: float
    generation: str


RELEASES: list[Release] = [
    Release("2021-06-29", "GitHub Copilot technical preview", "coding_assistant", 0.25, "copilot-1"),
    Release("2022-06-21", "GitHub Copilot general availability", "coding_assistant", 0.35, "copilot-1"),
    Release("2022-11-30", "ChatGPT (GPT-3.5)", "chat", 1.00, "gpt-3.5"),
    Release("2023-03-14", "GPT-4 / Claude 1", "chat", 0.85, "gpt-4"),
    Release("2023-03-22", "GitHub Copilot X announcement", "coding_assistant", 0.30, "gpt-4"),
    Release("2023-07-11", "Claude 2", "chat", 0.35, "mid-2023"),
    Release("2023-11-06", "GPT-4 Turbo", "api", 0.40, "late-2023"),
    Release("2023-12-06", "Gemini 1.0", "chat", 0.30, "late-2023"),
    Release("2024-03-04", "Claude 3 (Opus/Sonnet)", "chat", 0.60, "claude-3"),
    Release("2024-05-13", "GPT-4o", "chat", 0.70, "gpt-4o"),
    Release("2024-06-20", "Claude 3.5 Sonnet", "chat", 0.75, "claude-3.5"),
    Release("2024-09-12", "OpenAI o1-preview", "chat", 0.40, "reasoning-1"),
    Release("2024-10-22", "Claude 3.5 Sonnet (new) + computer use", "chat", 0.45, "claude-3.5"),
    Release("2024-11-19", "Cursor / AI editors mainstream adoption", "coding_assistant", 0.45, "editors"),
    Release("2025-02-24", "Claude 3.7 Sonnet + Claude Code preview", "agent", 0.70, "claude-code"),
    Release("2025-04-16", "OpenAI Codex CLI / o3", "agent", 0.50, "agents-1"),
    Release("2025-05-22", "Claude 4 (Opus/Sonnet)", "agent", 0.80, "claude-4"),
    Release("2025-08-07", "GPT-5", "chat", 0.70, "gpt-5"),
    Release("2025-09-29", "Claude Sonnet 4.5", "agent", 0.55, "claude-4.5"),
    Release("2026-01-01", "Agentic coding tools broadly default", "agent", 0.50, "agents-2"),
]

# Coarse eras used for pre/post comparisons and for the date-aware score.
ERAS = {
    "pre_llm": ("2018-01", "2022-10"),
    "chatgpt": ("2022-12", "2023-12"),
    "gpt4o_claude3": ("2024-01", "2024-12"),
    "agentic": ("2025-01", "2026-07"),
}

PRE_LLM_END = "2022-10"  # last month treated as clean baseline


def release_months() -> list[tuple[str, Release]]:
    return [(r.date[:7], r) for r in RELEASES]


def generations() -> dict[str, list[Release]]:
    out: dict[str, list[Release]] = {}
    for r in RELEASES:
        out.setdefault(r.generation, []).append(r)
    return out


def nearest(period: str, adoption_lag_months: float = 1.5) -> tuple[Release, float]:
    """Nearest release to a 'YYYY-MM' period, in months, after shifting releases
    forward by an assumed adoption lag. Positive distance = period is after."""
    from .util import month_index

    t = month_index(period)
    best, best_d = RELEASES[0], 1e9
    for r in RELEASES:
        d = t - (month_index(r.date[:7]) + adoption_lag_months)
        if abs(d) < abs(best_d):
            best, best_d = r, d
    return best, best_d
