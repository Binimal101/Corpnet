"""LLM prompts for MemoryNote construction, linking, and evolution.

Based on the A-Mem paper (arXiv 2502.12110) Section 3:
- P_s1: Note construction (keywords, context, tags)
- P_s2: Link generation (determine connections)
- P_s3: Memory evolution (update existing notes)
"""

# ── P_s1: Note Construction ──────────────────────────────────────────────────

NOTE_CONSTRUCTION_SYSTEM = """You are an AI memory system that analyzes content and extracts structured metadata.
Your goal is to create rich, searchable representations of information for later retrieval."""

NOTE_CONSTRUCTION_PROMPT = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // Several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are names of speakers or timestamps
        // At least three keywords, but avoid redundancy
    ],
    "context": 
        // One sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // Several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but avoid redundancy
    ]
}}

Content for analysis:
{content}"""


# ── P_s2: Link Generation ────────────────────────────────────────────────────

LINK_GENERATION_SYSTEM = """You are an AI memory evolution agent responsible for managing connections in a knowledge base.
Analyze memories to identify meaningful relationships that would help with future retrieval."""

LINK_GENERATION_PROMPT = """Analyze the new memory note and determine which existing memories should be linked to it.

The new memory:
- Content: {new_content}
- Context: {new_context}
- Keywords: {new_keywords}

Candidate memories to potentially link:
{candidates}

For each candidate, decide if a meaningful connection exists based on:
- Shared concepts or topics
- Causal or temporal relationships
- Complementary information
- Contradictions or updates

Return a JSON object:
{{
    "links": [
        {{
            "note_id": "id of the candidate note",
            "relation_type": "type of relationship (e.g., 'related_to', 'extends', 'contradicts', 'similar_topic')",
            "reason": "brief explanation of why this link is meaningful"
        }}
    ]
}}

Only include links that provide genuine value for retrieval. Do not force connections."""


# ── P_s3: Memory Evolution ───────────────────────────────────────────────────

MEMORY_EVOLUTION_SYSTEM = """You are an AI memory evolution agent responsible for keeping a knowledge base current.
When new information arrives, analyze whether existing memories should be updated."""

MEMORY_EVOLUTION_PROMPT = """A new memory has been added to the system. Analyze whether the nearest neighbor memories should evolve.

The new memory:
- Content: {new_content}
- Context: {new_context}
- Keywords: {new_keywords}
- Tags: {new_tags}

Nearest neighbor memories:
{neighbors}

For each neighbor, determine if its context or tags should be updated based on the new information.
Updates should:
- Strengthen connections by adding relevant tags
- Refine context descriptions with new understanding
- NOT change the original content

Return a JSON object:
{{
    "should_evolve": true/false,
    "updates": [
        {{
            "note_id": "id of the note to update",
            "new_context": "updated context string (or null to keep existing)",
            "new_tags": ["updated", "tag", "list"] // or null to keep existing
        }}
    ]
}}

Only propose updates that genuinely improve the memory organization."""
