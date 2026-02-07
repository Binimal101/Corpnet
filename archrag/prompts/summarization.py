"""Prompts for community summarisation."""

COMMUNITY_SUMMARY_SYSTEM = (
    "You are a concise labeler. "
    "Reply with ONLY the label — nothing else. "
    "Never exceed 10 words. Never start with 'The community' or 'This community'."
)

COMMUNITY_SUMMARY_PROMPT = """\
Write a label (MAX 10 words) describing what these members share.
Rules:
- 10 words or fewer, period.
- No filler phrases like "centers around", "revolves around", "is focused on".
- Just name the topic directly, e.g. "Einstein's contributions to relativity and physics".
- Do NOT reference hierarchy, levels, communities, or anything outside the list.

Members:
{entities_text}

Label:
"""
