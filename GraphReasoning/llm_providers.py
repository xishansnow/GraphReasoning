from typing import Callable, Optional, Dict, Any

# Unified signature for `generate()` callables
# generate(system_prompt: str, prompt: str, **kwargs) -> str


def get_generate_fn(provider: str, config: Optional[Dict[str, Any]] = None) -> Callable:
    """Return a unified generate() callable for the given provider.

    Supported providers:
    - openai: Official OpenAI API
    - deepseek: OpenAI-compatible API (base_url required)
    - qwen: OpenAI-compatible API (base_url required)
    - llama_cpp: Local llama.cpp via guidance
    - transformers: Local HuggingFace transformers text generation

    The returned function accepts: system_prompt, prompt, and optional kwargs like
    temperature, max_tokens, timeout, image_path (ignored unless supported).
    """
    config = config or {}
    provider = provider.lower()

    # OpenAI official API
    if provider == "openai":
        # Uses generate_OpenAIGPT from openai_tools
        from GraphReasoning.openai_tools import generate_OpenAIGPT

        def generate(system_prompt: str, prompt: str, **kwargs) -> str:
            return generate_OpenAIGPT(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 2048),
                timeout=kwargs.get("timeout", 120),
                frequency_penalty=kwargs.get("frequency_penalty", 0),
                presence_penalty=kwargs.get("presence_penalty", 0),
                top_p=kwargs.get("top_p", 1.0),
                openai_api_key=config.get("api_key", ""),
                gpt_model=config.get("model", "gpt-4-turbo"),
                organization=config.get("organization", ""),
            )

        return generate

    # Custom OpenAI-compatible servers
    if provider in ("deepseek", "qwen"):
        # OpenAI-compatible servers: require base_url and api_key
        import requests

        base_url = config.get("base_url")
        api_key = config.get("api_key")
        model = config.get("model")
        if not base_url or not api_key or not model:
            raise ValueError("For provider deepseek/qwen, base_url, api_key, and model are required")

        def generate(system_prompt: str, prompt: str, **kwargs) -> str:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "top_p": kwargs.get("top_p", 1.0),
            }
            url = base_url.rstrip("/") + "/chat/completions"
            resp = requests.post(url, json=payload, headers=headers, timeout=kwargs.get("timeout", 120))
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        return generate

    # Local OpenAI-compatible servers: ollama, lm_studio
    if provider in ("ollama", "lm_studio"):
        # Both expose OpenAI-compatible /v1/chat/completions endpoints locally.
        # Example configs:
        #   Ollama:   base_url='http://localhost:11434/v1', api_key='ollama'
        #   LMStudio: base_url='http://localhost:1234/v1',  api_key='lm-studio'
        import requests

        base_url = config.get("base_url")
        api_key = config.get("api_key", "")  # Some local servers accept any token
        model = config.get("model")
        if not base_url or not model:
            raise ValueError("For provider ollama/lm_studio, base_url and model are required")

        def generate(system_prompt: str, prompt: str, **kwargs) -> str:
            headers = {
                "Content-Type": "application/json",
            }
            # Include Authorization only if provided
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "top_p": kwargs.get("top_p", 1.0),
            }
            url = base_url.rstrip("/") + "/chat/completions"
            resp = requests.post(url, json=payload, headers=headers, timeout=kwargs.get("timeout", 120))
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        return generate

    # Local llama.cpp via guidance
    if provider == "llama_cpp":
        # Local llama.cpp via guidance
        try:
            from guidance.models import LlamaCpp
        except Exception as e:
            raise RuntimeError("guidance LlamaCpp not available. Install guidance and llama.cpp.") from e

        # Minimal wrapper around guidance model
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("llama_cpp requires model_path to gguf file")

        # Instantiate once and close over it
        llm = LlamaCpp(model=model_path)

        def generate(system_prompt: str, prompt: str, **kwargs) -> str:
            # Simple concatenation; advanced chat formatting can be added as needed
            full_prompt = f"{system_prompt}\n\n{prompt}"
            out = llm(full_prompt, temperature=kwargs.get("temperature", 0.2))
            return str(out)

        return generate

    # Local HF transformers text-generation
    if provider == "transformers":
        # Local HF transformers text-generation
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = config.get("model")
        if not model_name:
            raise ValueError("transformers requires 'model' (e.g., 'Qwen/Qwen2.5-7B-Instruct')")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        def generate(system_prompt: str, prompt: str, **kwargs) -> str:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            inputs = tokenizer(full_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 512),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generate

    raise ValueError(f"Unknown provider: {provider}")
