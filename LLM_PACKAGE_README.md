# LLM Integration Package

The `llms` package at the project root encapsulates all Large Language Model (LLM)-related functionality.

## Structure

```
llms/
├── __init__.py              # Package initialization with convenient exports
├── agents.py                # Conversation agents and multi-turn reasoning
├── llm_providers.py         # Unified interface for multiple LLM providers
├── openai_tools.py          # OpenAI-specific utilities
└── prompt_templates.py      # Prompt template management and registry
```

## Modules

### `agents.py` - Conversation Agents
Provides conversation agents for multi-turn reasoning and interaction:
- **`ConversationAgent`**: Multi-turn conversation with context management
- **`ZephyrLlamaCppChat`**: Chat interface for Zephyr model via llama.cpp
- **`ConversationAgent_LlamaIndex`**: LLamaIndex-based agents with RAG support (optional)

**Example:**
```python
from llms.agents import ConversationAgent

agent = ConversationAgent(
    chat_model=my_model,
    name="Expert",
    instructions="You are a helpful expert."
)
response = agent.reply("What is machine learning?")
```

### `llm_providers.py` - Multi-Provider Support
Unified interface for multiple LLM providers:

**Supported providers:**
- `openai`: Official OpenAI API
- `deepseek`: DeepSeek API (OpenAI-compatible)
- `qwen`: Alibaba Qwen API (OpenAI-compatible)
- `ollama`: Local Ollama server
- `lm_studio`: Local LM Studio server
- `llama_cpp`: Local llama.cpp via guidance
- `transformers`: Local HuggingFace transformers

**Example:**
```python
from Llms import get_generate_fn

# OpenAI
gen = get_generate_fn('openai', {
    'api_key': 'sk-...',
    'model': 'gpt-4'
})

# Local Ollama
gen = get_generate_fn('ollama', {
    'base_url': 'http://localhost:11434/v1',
    'model': 'llama2'
})

response = gen(
    system_prompt="You are helpful.",
    prompt="Hello!"
)
```

### `openai_tools.py` - OpenAI Utilities
OpenAI-specific functionality:
- **`generate_OpenAIGPT()`**: Direct text generation
- **`reason_over_image_OpenAI()`**: Vision capabilities
- **`develop_prompt_from_text_and_generate_image()`**: DALL-E image generation
- **`get_answer()`**: Multi-modal Q&A with images

**Example:**
```python
from Llms import generate_OpenAIGPT

response = generate_OpenAIGPT(
    system_prompt="You are a scientist.",
    prompt="Explain quantum computing.",
    openai_api_key="sk-...",
    gpt_model="gpt-4"
)
```

### `prompt_templates.py` - Prompt Management
Centralized prompt template management:
- **`PromptTemplate`**: Individual template with variable substitution
- **`PromptTemplateRegistry`**: Central registry of all templates
- **`render_prompt()`**: Render template with variables
- **`add_custom_template()`**: Add custom templates at runtime

**Built-in templates:**
- `graph_maker_initial`: Extract ontology from text
- `graph_format`: Format ontology consistently
- `graph_fix_format`: Fix malformed ontology
- `graph_add_triplets`: Add triplets to ontology
- `graph_refine`: Refine ontology labels
- `historical_entity_extraction`: Extract historical entities
- `historical_relation_extraction`: Extract historical relations
- `historical_event_timeline`: Create event timeline

**Example:**
```python
from Llms import render_prompt, add_custom_template

# Use built-in template
sys_prompt, user_prompt = render_prompt(
    'graph_maker_initial',
    input='Silk is a protein fiber...'
)

# Add custom template
add_custom_template(
    name='my_template',
    system_prompt='You are helpful.',
    user_prompt='Analyze: {text}'
)

# Use custom template
sys_prompt, user_prompt = render_prompt('my_template', text='...')
```

## Quick Start

### Installation

No special installation needed - the `llms` package is part of the project.

### Basic Usage

```python
# Generate text with multiple provider support
from Llms import get_generate_fn

# Any provider, unified interface
gen = get_generate_fn('openai', {'api_key': 'sk-...', 'model': 'gpt-4'})
response = gen(system_prompt="Help me.", prompt="How do I...?")

# Use prompts
from Llms import render_prompt
sys_p, user_p = render_prompt('graph_maker_initial', input='Your text here')

# Agents
from llms.agents import ConversationAgent
agent = ConversationAgent(model, name="Expert", instructions="You help.")
response = agent.reply("Your question?")
```

## Integration with GraphReasoning

For backward compatibility, the old import paths still work with deprecation warnings:

```python
# Old way (deprecated but works)
from Llms.llm_providers import get_generate_fn
from Llms.prompt_templates import render_prompt
from Llms.agents import ConversationAgent

# New way (recommended)
from Llms import get_generate_fn, render_prompt
from llms.agents import ConversationAgent
```

## Configuration

### OpenAI
```python
config = {
    'api_key': 'sk-...',
    'model': 'gpt-4-turbo',
    'organization': 'org-...'  # optional
}
gen = get_generate_fn('openai', config)
```

### Local Ollama
```python
config = {
    'base_url': 'http://localhost:11434/v1',
    'model': 'llama2',
    'api_key': ''  # optional
}
gen = get_generate_fn('ollama', config)
```

### HuggingFace Transformers
```python
config = {
    'model': 'Qwen/Qwen2.5-7B-Instruct'
}
gen = get_generate_fn('transformers', config)
```

## Key Features

✅ **Multi-Provider Support**: Unified interface for any LLM provider
✅ **Conversation Agents**: Multi-turn reasoning with context management
✅ **Template Management**: Centralized, reusable prompt templates
✅ **Vision Capabilities**: Image analysis with OpenAI Vision
✅ **RAG Integration**: LLamaIndex support for document-based Q&A
✅ **Backward Compatibility**: Old imports still work with deprecation warnings

## Best Practices

1. **Use the new `llms` package** for new code
2. **Store API keys** in environment variables, not in code
3. **Use templates** for consistency across prompts
4. **Leverage agents** for complex multi-turn interactions
5. **Handle import errors** gracefully for optional dependencies
