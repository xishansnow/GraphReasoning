"""
LLM (Large Language Model) Integration Package

This package encapsulates all LLM-related functionality including:
- Agent-based reasoning and conversation systems
- Multi-provider LLM support (OpenAI, DeepSeek, Ollama, etc.)
- OpenAI-specific utilities and image processing
- Prompt template management for reproducible LLM interactions
- Integration with LLamaIndex for advanced RAG patterns

Modules:
--------
agents.py
    Conversation agents and multi-turn reasoning systems
    - ConversationAgent, ZephyrLlamaCppChat
    - LLamaIndex-based agents for RAG

llm_providers.py
    Unified interface for multiple LLM providers
    - get_generate_fn(): Factory function for any provider
    - Supported: OpenAI, DeepSeek, Qwen, Ollama, LM Studio, llama.cpp, HF Transformers

openai_tools.py
    OpenAI-specific utilities
    - generate_OpenAIGPT: Direct OpenAI API calls
    - reason_over_image_OpenAI: Vision capabilities
    - DALL-E image generation

prompt_templates.py
    Prompt template management and registry
    - PromptTemplate: Individual template with variable substitution
    - PromptTemplateRegistry: Central registry of all templates
    - Built-in templates for common tasks
    - add_custom_template(): Add new templates at runtime
"""

# Import core components for convenient access
from Llms.agents import (
    ConversationAgent,
    ZephyrLlamaCppChat,
    get_entire_conversation,
    read_and_summarize,
)

# Import LLamaIndex agents only if available
try:
    from Llms.agents import (
        ConversationAgent_LlamaIndex,
        read_and_summarize_LlamaIndex,
        get_answer_LlamaIndex,
        get_chat_engine_from_index_LlamaIndex,
    )
except ImportError:
    pass

from Llms.llm_providers import (
    get_generate_fn,
)

from Llms.openai_tools import (
    generate_OpenAIGPT,
    reason_over_image_OpenAI,
    reason_over_image_and_graph_via_triples,
    develop_prompt_from_text_and_generate_image,
    get_answer,
    is_url,
)

from Llms.prompt_templates import (
    PromptTemplate,
    PromptTemplateRegistry,
    get_registry,
    render_prompt,
    add_custom_template,
    list_available_templates,
    PROMPT_GRAPH_MAKER_INITIAL,
    PROMPT_GRAPH_FORMAT,
    PROMPT_GRAPH_FIX_FORMAT,
    PROMPT_GRAPH_ADD_TRIPLETS,
    PROMPT_GRAPH_REFINE,
    PROMPT_HISTORICAL_ENTITY_EXTRACTION,
    PROMPT_HISTORICAL_RELATION_EXTRACTION,
    PROMPT_HISTORICAL_EVENT_TIMELINE,
)

__all__ = [
    # agents
    "ConversationAgent",
    "ZephyrLlamaCppChat",
    "get_entire_conversation",
    "read_and_summarize",
    # llm_providers
    "get_generate_fn",
    # openai_tools
    "generate_OpenAIGPT",
    "reason_over_image_OpenAI",
    "reason_over_image_and_graph_via_triples",
    "develop_prompt_from_text_and_generate_image",
    "get_answer",
    "is_url",
    # prompt_templates
    "PromptTemplate",
    "PromptTemplateRegistry",
    "get_registry",
    "render_prompt",
    "add_custom_template",
    "list_available_templates",
    "PROMPT_GRAPH_MAKER_INITIAL",
    "PROMPT_GRAPH_FORMAT",
    "PROMPT_GRAPH_FIX_FORMAT",
    "PROMPT_GRAPH_ADD_TRIPLETS",
    "PROMPT_GRAPH_REFINE",
    "PROMPT_HISTORICAL_ENTITY_EXTRACTION",
    "PROMPT_HISTORICAL_RELATION_EXTRACTION",
    "PROMPT_HISTORICAL_EVENT_TIMELINE",
]

