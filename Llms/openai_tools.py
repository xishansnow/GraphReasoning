"""OpenAI-specific utilities for LLM interactions.

This module provides:
- Direct OpenAI API integration for text generation
- Vision capabilities for image analysis
- DALL-E image generation
- Graph reasoning with images
"""

from openai import OpenAI
import base64
import requests
from datetime import datetime
import openai

try:
    from GraphTools import print_node_pairs_edge_title
except ImportError:
    def print_node_pairs_edge_title(graph):
        """Fallback stub if graph_tools not available."""
        return []

try:
    from GraphReasoning.utils import make_dir_if_needed
except ImportError:
    def make_dir_if_needed(path):
        """Fallback stub if utils not available."""
        import os
        os.makedirs(path, exist_ok=True)

try:
    from GraphReasoning.graph_analysis import *
except ImportError:
    pass


def generate_OpenAIGPT(
    system_prompt='You are a materials scientist.',
    prompt="Describe the best options to design abrasive materials.",
    temperature=0.2,
    max_tokens=2048,
    timeout=120,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.,
    openai_api_key='',
    gpt_model='gpt-4-vision-preview',
    organization='',
) -> str:
    """Generate text using OpenAI GPT models.
    
    Args:
        system_prompt: System role/context prompt
        prompt: User prompt/query
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        frequency_penalty: Penalize repeated tokens
        presence_penalty: Penalize new tokens
        top_p: Nucleus sampling parameter
        openai_api_key: OpenAI API key
        gpt_model: Model name (e.g., 'gpt-4-turbo', 'gpt-4-vision-preview')
        organization: OpenAI organization ID
        
    Returns:
        Generated text response
    """
    client = openai.OpenAI(api_key=openai_api_key, organization=organization)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=gpt_model,
        timeout=timeout,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
    )
    return chat_completion.choices[0].message.content


def reason_over_image_OpenAI(
    system_prompt='You are a scientist.',
    prompt='Carefully analyze this image.',
    image_path='',
    temperature=0.2,
    max_tokens=2048,
    timeout=120,
    frequency_penalty=0,
    presence_penalty=0,
    openai_api_key='',
    gpt_model='gpt-4-vision-preview',
    organization='',
    top_p=1.,
    verbatim=False,
) -> str:
    """Reason over images using OpenAI's vision capabilities.
    
    Args:
        system_prompt: System role/context
        prompt: Question/instruction about the image
        image_path: Path to local image file
        temperature: Sampling temperature
        max_tokens: Maximum response length
        timeout: Request timeout
        frequency_penalty: Token repetition penalty
        presence_penalty: New token penalty
        openai_api_key: OpenAI API key
        gpt_model: Vision model (e.g., 'gpt-4-vision-preview')
        organization: OpenAI organization
        top_p: Nucleus sampling
        verbatim: Print full prompt if True
        
    Returns:
        Analysis/reasoning about the image
    """
    if verbatim:
        print("Prompt: ", prompt)
        
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    payload = {
        "model": gpt_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )
   
    if verbatim:
        from IPython.display import display, Markdown
        display(Markdown(response.json()['choices'][0]['message']['content']))

    return response.json()['choices'][0]['message']['content']


def reason_over_image_and_graph_via_triples(
    path_graph,
    generate,
    image_path='',
    keyword_1="music and sound",
    keyword_2="apples",
    include_keywords_as_nodes=True,
    inst_prepend='',
    instruction='Now, reason over them and propose a research hypothesis.',
    verbatim=False,
    N_limit=None,
    temperature=0.3,
    keywords_separator=' --> ',
    system_prompt='You are a scientist who uses logic and reasoning.',
    max_tokens=4096,
    prepend='You are given a set of information from a graph. You analyze these logically through reasoning.\n\n',
    save_files=True,
    data_dir='./',
    visualize_paths_as_graph=True,
    display_graph=True,
    words_per_line=2,
) -> tuple:
    """Reason over knowledge graphs and images combined.
    
    Args:
        path_graph: Knowledge graph object
        generate: Generate function (callable)
        image_path: Path to image file
        keyword_1: First concept
        keyword_2: Second concept
        include_keywords_as_nodes: Include keywords in reasoning
        inst_prepend: Instructions to prepend
        instruction: Main instruction
        verbatim: Print debugging info
        N_limit: Limit number of nodes
        temperature: Sampling temperature
        keywords_separator: Separator between keywords
        system_prompt: System role
        max_tokens: Max response length
        prepend: Prepend to task
        save_files: Save output files
        data_dir: Output directory
        visualize_paths_as_graph: Visualize results
        display_graph: Display graph visualization
        words_per_line: Formatting parameter
        
    Returns:
        Tuple of (response, graph, filename, graphML)
    """
    print("Reason over graph and image: ", image_path)
    
    make_dir_if_needed(data_dir)
    task = inst_prepend + ''
    
    join_strings = lambda strings: '\n'.join(strings)
    join_strings_newline = lambda strings: '\n'.join(strings)

    node_list = print_node_pairs_edge_title(path_graph)
    if N_limit is not None:
        node_list = node_list[:N_limit]

    if verbatim:
        print("Node list: ", node_list)
        
    if include_keywords_as_nodes:
        task += f"Analysis of relationships between {keyword_1} and {keyword_2}.\n\n"
    
    task += f"Knowledge graph nodes and relations:\n\nFormat: node_1, relationship, node_2\n\nData:\n\n{join_strings_newline(node_list)}\n\n"
    task += f"{instruction}"
     
    if verbatim:
        print("Task:\n", task)
    
    response = generate(
        system_prompt=system_prompt,
        prompt=task,
        max_tokens=max_tokens,
        temperature=temperature,
        image_path=image_path,
    )
    
    if verbatim:
        from IPython.display import display, Markdown
        display(Markdown("**Response:** " + response))

    return response, path_graph


def develop_prompt_from_text_and_generate_image(
    response,
    generate_fn,
    image_dir_name='./image_temp/',
    number_imgs=1,
    size="1024x1024",
    show_img=True,
    max_tokens=2048,
    temperature=0.3,
    quality='hd',
    style='vivid',
    direct_prompt=None,
    openai_api_key='',
    gpt_model='gpt-4-0125-preview',
    organization='',
    dalle_model="dall-e-3",
    system_prompt="You make prompts for DALL-E 3.",
) -> list:
    """Generate images using DALL-E based on text description.
    
    Args:
        response: Text description to generate image from
        generate_fn: Function to generate image prompt
        image_dir_name: Directory to save images
        number_imgs: Number of images to generate
        size: Image size (e.g., '1024x1024')
        show_img: Display generated images
        max_tokens: Max tokens for prompt generation
        temperature: Sampling temperature
        quality: Image quality ('standard' or 'hd')
        style: Image style ('vivid' or 'natural')
        direct_prompt: Direct DALL-E prompt if provided
        openai_api_key: OpenAI API key
        gpt_model: GPT model for prompt generation
        organization: OpenAI organization
        dalle_model: DALL-E model name
        system_prompt: System prompt for prompt generation
        
    Returns:
        List of image file paths
    """
    import os
    from PIL import Image
    from io import BytesIO
    from IPython.display import display, Image as DisplayImage
    from pathlib import Path
    
    image_dir = os.path.join(os.curdir, image_dir_name)
    make_dir_if_needed(image_dir)
    img_list = []
    
    if direct_prompt is None:
        task = f'''Consider this description: {response}

Develop a detailed prompt for DALL-E 3 to visualize this. 
The prompt should describe key features clearly. Do NOT include any text in the image.
'''
        response = generate_fn(
            system_prompt=system_prompt,
            prompt=task,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        from IPython.display import display, Markdown
        display(Markdown("Image prompt:\n\n" + response))
    else:
        response = direct_prompt
        from IPython.display import display, Markdown
        display(Markdown("Image prompt:\n\n" + response))
     
    client = openai.OpenAI(api_key=openai_api_key, organization=organization)
    generation_response = client.images.generate(
        model=dalle_model,
        prompt=response,
        n=number_imgs,
        style=style,
        quality=quality,
        size=size,
        response_format="b64_json",
    )
    
    for index, image_dict in enumerate(generation_response.data):
        image_data = base64.b64decode(image_dict.b64_json)
        time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_file = os.path.join(
            image_dir_name,
            f"generated_image_{time_part}_{response[:32]}_{index}.png"
        )
        with open(image_file, mode="wb") as png:
            png.write(image_data)
        if show_img:
            display(DisplayImage(data=image_data))
        img_list.append(image_file)
     
    return img_list


def is_url(val) -> bool:
    """Check if value is a URL string."""
    return isinstance(val, str) and val.startswith("http")


def get_answer(
    query='What is shown in this image?',
    model="gpt-4o",
    image=None,
    payload=None,
    max_tokens=1024,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    init_instr="Look at this image: ",
    display_image=False,
    system='You are a helpful assistant.',
    api_key='',
) -> tuple:
    """Get answer from OpenAI model with optional image.
    
    Args:
        query: Question to ask
        model: Model to use (e.g., 'gpt-4o')
        image: Image path or PIL Image
        payload: Previous conversation payload
        max_tokens: Max response tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-k sampling
        init_instr: Initial instruction
        display_image: Display image if provided
        system: System prompt
        api_key: OpenAI API key
        
    Returns:
        Tuple of (response_text, updated_payload)
    """
    from transformers.image_utils import load_image
    from io import BytesIO
    from IPython.display import display
    
    base64_image = None
    if image is not None:
        if is_url(image):
            image = load_image(image)
        else:
            image = load_image(image)
            
        if display_image:
            display(image)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_byte_array = buffered.getvalue()
        base64_image = base64.b64encode(img_byte_array).decode("utf-8")
       
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if payload is None:
        if base64_image is not None:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
        else:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}]
                    }
                ],
                "max_tokens": max_tokens
            }
    else:
        payload['messages'].append({
            "role": "user",
            "content": [{"type": "text", "text": query}]
        })

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response_dict = response.json()
    message_content = response_dict['choices'][0]['message']['content']

    payload['messages'].append({
        "role": "assistant",
        "content": [{"type": "text", "text": message_content}]
    })

    return message_content, payload
