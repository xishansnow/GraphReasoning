"""Prompt Template Manager for LLM-based tasks.

This module manages system prompts and user prompts for LLM interactions,
allowing centralized maintenance and easy customization of prompts.
"""

from typing import Dict, Any, Optional


class PromptTemplate:
    """Single prompt template with variable substitution support."""

    def __init__(self, name: str, system_prompt: str, user_prompt: str):
        """Initialize a prompt template.

        Args:
            name: Template identifier.
            system_prompt: System role/context prompt.
            user_prompt: User-facing task prompt with optional {placeholders}.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def render(self, **kwargs) -> tuple:
        """Render template with variable substitution.

        Args:
            **kwargs: Variables to substitute in the prompts.

        Returns:
            Tuple (system_prompt, user_prompt) with variables substituted.
        """
        sys_rendered = self._safe_format(self.system_prompt, **kwargs)
        usr_rendered = self._safe_format(self.user_prompt, **kwargs)
        return sys_rendered, usr_rendered

    @staticmethod
    def _safe_format(text: str, **kwargs) -> str:
        """Safely format a string, only substituting known kwargs.
        
        Ignores format placeholders that are not in kwargs.
        """
        try:
            if kwargs and "{" in text:
                return text.format(**kwargs)
            return text
        except (KeyError, ValueError):
            return text


class PromptTemplateRegistry:
    """Registry for managing multiple prompt templates."""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Retrieve a template by name."""
        return self._templates.get(name)

    def list_templates(self) -> list:
        """List all registered template names."""
        return list(self._templates.keys())

    def render(self, template_name: str, **kwargs) -> tuple:
        """Render a template by name.

        Args:
            template_name: Name of the template to render.
            **kwargs: Variables for substitution.

        Returns:
            Tuple (system_prompt, user_prompt).

        Raises:
            ValueError: If template not found.
        """
        template = self.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found. Available: {self.list_templates()}")
        return template.render(**kwargs)


# ============================================================================
# Built-in Graph Generation Prompts
# ============================================================================

PROMPT_GRAPH_MAKER_INITIAL = PromptTemplate(
    name="graph_maker_initial",
    system_prompt=(
        "You are a network ontology graph maker who extracts terms and their relations from context, using category theory. "
        "Your task is to extract the ontology of key concepts in the given context, including materials, systems, and methods.\n\n"
        "Format output as JSON triplets: [{'node_1': 'concept', 'node_2': 'related', 'edge': 'relationship'}, ...]\n\n"
        "Examples:\n"
        "Context: 'Alice is Marc's mother.'\n"
        "[{'node_1': 'Alice', 'node_2': 'Marc', 'edge': 'is mother of'}]\n\n"
        "Produce around 10 triplets reflecting consistent ontologies."
    ),
    user_prompt="Context: ```{input}```\n\nOutput: ",
)

PROMPT_GRAPH_FORMAT = PromptTemplate(
    name="graph_format",
    system_prompt=(
        "You respond in JSON format:\n"
        "[{'node_1': 'concept', 'node_2': 'related', 'edge': 'relationship'}, ...]"
    ),
    user_prompt=(
        "Read this context: ```{input}```\n"
        "Read this ontology: ```{ontology}```\n\n"
        "Improve the ontology by renaming nodes with consistent field-standard labels."
    ),
)

PROMPT_GRAPH_FIX_FORMAT = PromptTemplate(
    name="graph_fix_format",
    system_prompt=(
        "You respond in JSON triplet format:\n"
        "[{'node_1': 'concept', 'node_2': 'related', 'edge': 'relationship'}, ...]"
    ),
    user_prompt="Fix this ontology to proper JSON format: ```{ontology}```",
)

PROMPT_GRAPH_ADD_TRIPLETS = PromptTemplate(
    name="graph_add_triplets",
    system_prompt=(
        "You are a network ontology graph maker extracting terms and relations from context.\n"
        "Format output as JSON triplets in the specified format."
    ),
    user_prompt=(
        "Read context: ```{input}```\n"
        "Read ontology: ```{ontology}```\n\n"
        "Add new triplets to the original list, keeping same JSON format. Return original + new triplets."
    ),
)

PROMPT_GRAPH_REFINE = PromptTemplate(
    name="graph_refine",
    system_prompt=(
        "You respond in JSON triplet format:\n"
        "[{'node_1': 'concept', 'node_2': 'related', 'edge': 'relationship'}, ...]"
    ),
    user_prompt=(
        "Read context: ```{input}```\n"
        "Read ontology: ```{ontology}```\n\n"
        "Revise the ontology with consistent and concise node/edge labels."
    ),
)

# Historical Knowledge Graph Prompts
PROMPT_HISTORICAL_ENTITY_EXTRACTION = PromptTemplate(
    name="historical_entity_extraction",
    system_prompt=(
        "You are a historical knowledge extraction specialist. Extract named entities from historical text "
        "including persons, places, events, organizations, and artifacts. "
        "For each entity, provide type and description."
    ),
    user_prompt=(
        "Extract historical entities from this text:\n```{text}```\n\n"
        "Format: [{'entity': 'name', 'type': 'person|place|event|organization|artifact', 'description': '...'}]"
    ),
)

PROMPT_HISTORICAL_RELATION_EXTRACTION = PromptTemplate(
    name="historical_relation_extraction",
    system_prompt=(
        "You are a historical relation extraction specialist. Extract relationships between entities "
        "from historical text. Include causality, temporal, and spatial relations."
    ),
    user_prompt=(
        "Extract relations from this historical text:\n```{text}```\n\n"
        "Known entities: {entities}\n\n"
        "Format: [{'source': 'entity1', 'target': 'entity2', 'relation': 'type', 'description': '...'}]"
    ),
)

PROMPT_HISTORICAL_EVENT_TIMELINE = PromptTemplate(
    name="historical_event_timeline",
    system_prompt=(
        "You are a historical timeline expert. Order historical events chronologically "
        "and identify cause-effect relationships between them."
    ),
    user_prompt=(
        "Create a timeline from these historical facts:\n```{text}```\n\n"
        "Format: [{'date': 'YYYY-MM-DD or period', 'event': 'description', 'significance': '...'}]"
    ),
)

# ============================================================================
# Default Registry (initialized with built-in templates)
# ============================================================================

_default_registry = PromptTemplateRegistry()
_default_registry.register(PROMPT_GRAPH_MAKER_INITIAL)
_default_registry.register(PROMPT_GRAPH_FORMAT)
_default_registry.register(PROMPT_GRAPH_FIX_FORMAT)
_default_registry.register(PROMPT_GRAPH_ADD_TRIPLETS)
_default_registry.register(PROMPT_GRAPH_REFINE)
_default_registry.register(PROMPT_HISTORICAL_ENTITY_EXTRACTION)
_default_registry.register(PROMPT_HISTORICAL_RELATION_EXTRACTION)
_default_registry.register(PROMPT_HISTORICAL_EVENT_TIMELINE)


def get_registry() -> PromptTemplateRegistry:
    """Get the default template registry."""
    return _default_registry


def render_prompt(template_name: str, **kwargs) -> tuple:
    """Convenience function to render a prompt from the default registry.

    Args:
        template_name: Name of the template.
        **kwargs: Variables for substitution.

    Returns:
        Tuple (system_prompt, user_prompt).
    """
    return _default_registry.render(template_name, **kwargs)


def add_custom_template(name: str, system_prompt: str, user_prompt: str) -> None:
    """Add a custom prompt template to the default registry.

    Args:
        name: Unique template identifier.
        system_prompt: System prompt text.
        user_prompt: User prompt text (may contain {placeholders}).
    """
    template = PromptTemplate(name, system_prompt, user_prompt)
    _default_registry.register(template)


def list_available_templates() -> list:
    """List all available template names."""
    return _default_registry.list_templates()
