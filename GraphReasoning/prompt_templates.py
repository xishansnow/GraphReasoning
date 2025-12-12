"""
Prompt Template Manager for Graph Generation.

This module manages system prompts and user prompts for LLM-based graph generation,
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
        # Use safe_format to handle braces that are not placeholders
        sys_rendered = self._safe_format(self.system_prompt, **kwargs)
        usr_rendered = self._safe_format(self.user_prompt, **kwargs)
        return sys_rendered, usr_rendered

    @staticmethod
    def _safe_format(text: str, **kwargs) -> str:
        """Safely format a string, only substituting known kwargs.
        
        Ignores format placeholders that are not in kwargs.
        """
        try:
            # Only try to format if there are kwargs and text contains {
            if kwargs and "{" in text:
                return text.format(**kwargs)
            return text
        except (KeyError, ValueError):
            # If formatting fails, return original text
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
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```). Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods. \n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms "
        "and the relation between them, like the following: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        "   }, {...}\n"
        "]\n\n"
        "Examples:\n"
        "Context: ```Alice is Marc's mother.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        "   }\n"
        "]\n\n"
        "Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
        "[\n"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        "   },\n"
        "   {\n"
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        "   },\n"
        "   {\n"
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        "   }\n"
        "]\n\n"
        "Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent ontologies.\n"
    ),
    user_prompt="Context: ```{input}```\n\nOutput: ",
)

PROMPT_GRAPH_FORMAT = PromptTemplate(
    name="graph_format",
    system_prompt=(
        "You respond in this format:\n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        '   }, {...} ]\n'
    ),
    user_prompt=(
        "Read this context: ```{input}```.\n"
        "Read this ontology: ```{ontology}```\n\n"
        "Improve the ontology by renaming nodes so that they have consistent labels that are widely used in the field of materials science."
    ),
)

PROMPT_GRAPH_FIX_FORMAT = PromptTemplate(
    name="graph_fix_format",
    system_prompt=(
        "You respond in this format:\n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        '   }, {...} ]\n'
    ),
    user_prompt="Context: ```{ontology}```\n\nFix to make sure it is proper format.",
)

PROMPT_GRAPH_ADD_TRIPLETS = PromptTemplate(
    name="graph_add_triplets",
    system_prompt=(
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by ```). Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context, including well-defined and widely used names of materials, systems, methods.\n\n"
        "Format your output as a list of JSON triplets."
    ),
    user_prompt=(
        "Insert new triplets into the original ontology. Read this context: ```{input}```.\n"
        "Read this ontology: ```{ontology}```\n\n"
        "Insert additional triplets to the original list, in the same JSON format. Repeat original AND new triplets.\n"
    ),
)

PROMPT_GRAPH_REFINE = PromptTemplate(
    name="graph_refine",
    system_prompt=(
        "You respond in this format:\n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
        '   }, {...} ]\n'
    ),
    user_prompt=(
        "Read this context: ```{input}```.\n"
        "Read this ontology: ```{ontology}```\n\n"
        "Revise the ontology by renaming nodes and edges so that they have consistent and concise labels."
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
