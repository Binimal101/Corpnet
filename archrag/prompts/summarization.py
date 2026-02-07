"""Prompts for community summarisation."""

COMMUNITY_SUMMARY_SYSTEM = (
    "You produce very short community labels. "
    "Output only the summary — no preamble, no bullet points, no commentary."
)

COMMUNITY_SUMMARY_PROMPT = """\
Below are the members of a single community. Write ONE short sentence (max 15 words) \
that captures what these members have in common. Do NOT mention other communities, \
hierarchy levels, or how this group relates to anything outside it. \
Just state the shared topic or theme.

Members:
{entities_text}

One-sentence summary:
"""
