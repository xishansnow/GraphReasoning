"""
Prompt Template Usage Examples and Documentation.

This file shows how to use and customize the prompt template system.
"""

from GraphReasoning.prompt_templates import (
    render_prompt,
    add_custom_template,
    list_available_templates,
    get_registry,
)


def example_1_list_templates():
    """Example 1: List all available templates."""
    print("Available templates:")
    for name in list_available_templates():
        print(f"  - {name}")


def example_2_render_template():
    """Example 2: Render a template with variables."""
    sys_prompt, user_prompt = render_prompt(
        "graph_maker_initial",
        input="Silk is a strong material made of protein fibers.",
    )
    print("System Prompt:")
    print(sys_prompt)
    print("\nUser Prompt:")
    print(user_prompt)


def example_3_add_custom_template():
    """Example 3: Add a custom prompt template."""
    custom_sys = "You are an expert in materials science and knowledge graphs."
    custom_usr = (
        "Analyze the following material description and extract key properties:\n\n"
        "Material: {material_name}\n"
        "Description: {description}\n\n"
        "Output properties as JSON."
    )

    add_custom_template("materials_properties", custom_sys, custom_usr)

    # Now use it
    sys_p, usr_p = render_prompt(
        "materials_properties",
        material_name="Spider Silk",
        description="A protein fiber with exceptional strength and elasticity.",
    )
    print("Custom Template System Prompt:")
    print(sys_p)
    print("\nCustom Template User Prompt:")
    print(usr_p)


def example_4_inspect_registry():
    """Example 4: Inspect registry and modify templates."""
    registry = get_registry()

    print("All registered templates:")
    for name in registry.list_templates():
        template = registry.get(name)
        print(f"\n{name}:")
        print(f"  System: {template.system_prompt[:80]}...")
        print(f"  User: {template.user_prompt[:80]}...")


def example_5_batch_rendering():
    """Example 5: Render multiple templates for a workflow."""
    input_text = "Graphene is a single layer of carbon atoms."

    steps = [
        ("graph_maker_initial", {"input": input_text}),
        ("graph_format", {"input": input_text, "response": "{}"}),
        ("graph_fix_format", {"response": "{}"}),
    ]

    for template_name, kwargs in steps:
        sys_p, usr_p = render_prompt(template_name, **kwargs)
        print(f"\n=== {template_name} ===")
        print(f"System: {sys_p[:100]}...")
        print(f"User: {usr_p[:100]}...")


# ============================================================================
# Integration with graphPrompt
# ============================================================================

"""
The graphPrompt function now uses render_prompt() to fetch prompts dynamically:

    from GraphReasoning.prompt_templates import render_prompt
    
    def graphPrompt(input: str, generate, metadata={}, repeat_refine=0, verbatim=False):
        # Step 1: Initial generation
        sys_prompt, user_prompt = render_prompt("graph_maker_initial", input=input)
        response = generate(system_prompt=sys_prompt, prompt=user_prompt)
        
        # Step 2: Format improvement
        sys_prompt, user_prompt = render_prompt("graph_format", 
                                                input=input, response=response)
        response = generate(system_prompt=sys_prompt, prompt=user_prompt)
        
        # ... more steps ...
        
        # Results handling (JSON parsing, etc.)
        return parsed_results

Benefits:
- Centralized prompt management: all prompts in one place
- Easy customization: modify templates without touching graphPrompt code
- Version control: track prompt changes in git
- A/B testing: register alternative templates and switch between them
- Reuse: use same prompts in different contexts
"""


# ============================================================================
# Tips for Customization
# ============================================================================

"""
1. **Modify Built-in Templates:**
   
   from GraphReasoning.prompt_templates import PromptTemplate, get_registry
   
   registry = get_registry()
   custom_template = PromptTemplate(
       "graph_maker_initial",  # Same name overwrites!
       system_prompt="Your custom system prompt here",
       user_prompt="Your custom user prompt with {input} placeholder"
   )
   registry.register(custom_template)

2. **Add New Custom Templates:**
   
   from GraphReasoning.prompt_templates import add_custom_template
   
   add_custom_template(
       "my_custom_template",
       system_prompt="...",
       user_prompt="... with {variable} placeholders ..."
   )

3. **Domain-Specific Templates:**
   
   # Create a separate module for your domain:
   # my_domain_prompts.py
   
   from GraphReasoning.prompt_templates import (
       PromptTemplate,
       add_custom_template
    )
   
   # Add domain-specific templates
   add_custom_template("biomedical_graph", ...)
   add_custom_template("chemical_graph", ...)

4. **Load Templates from Configuration:**
   
   import json
   from GraphReasoning.prompt_templates import add_custom_template
   
   with open("prompts_config.json") as f:
       config = json.load(f)
       for name, prompts in config.items():
           add_custom_template(
               name,
               system_prompt=prompts["system"],
               user_prompt=prompts["user"]
           )

5. **Template Variables:**
   
   Templates support Python format string syntax: {variable_name}
   
   Example:
       "Please analyze this text: {input}"
       "Improve this response: {response}"
       "Focus on material: {material_type}"
   
   When rendering:
       render_prompt("template_name", 
                    input="...",
                    response="...", 
                    material_type="...")
"""


if __name__ == "__main__":
    print("=" * 70)
    print("Prompt Template Examples")
    print("=" * 70)

    print("\n### Example 1: List Templates ###")
    example_1_list_templates()

    print("\n### Example 2: Render Template ###")
    example_2_render_template()

    print("\n### Example 3: Add Custom Template ###")
    example_3_add_custom_template()

    print("\n### Example 4: Inspect Registry ###")
    example_4_inspect_registry()

    print("\n### Example 5: Batch Rendering ###")
    example_5_batch_rendering()

    print("\n" + "=" * 70)
