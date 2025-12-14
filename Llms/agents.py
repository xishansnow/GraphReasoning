"""Agent-based reasoning and conversation systems.

This module provides conversation agents and multi-turn reasoning capabilities
for LLM-based systems, including support for guidance-based local models
and LLamaIndex integration for RAG patterns.
"""

try:
    import transformers
    from transformers import logging
    logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from GraphReasoning.utils import *
except ImportError:
    pass

try:
    from guidance import models, gen, select, system, user, assistant
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False

newline = "\n"

try:
    from IPython.display import display, Markdown
except ImportError:
    display = None
    Markdown = None

try:
    import markdown2
    import pdfkit
except ImportError:
    pass


class ZephyrLlamaCppChat:
    """Chat interface for Zephyr model via llama.cpp + guidance."""
    
    def __init__(self, base_model):
        """Initialize with a guidance LlamaCpp model."""
        if not GUIDANCE_AVAILABLE:
            raise ImportError("guidance package required. Install with: pip install guidance")
        self.model = base_model
    
    def get_role_start(self, role_name, **kwargs):
        """Get role start token."""
        if role_name == "user":
            return "<|user|>\n"
        elif role_name == "assistant":
            return "<|assistant|>\n"
        elif role_name == "system":
            return "<|system|>\n"
        return ""
            
    def get_role_end(self, role_name=None):
        """Get role end token."""
        return "</s>"


if GUIDANCE_AVAILABLE:
    class ConversationAgent:
        """Multi-turn conversation agent with context management."""
        
        def __init__(
            self,
            chat_model,
            name: str,
            instructions: str,
            context_turns: int = 2,
            temperature: float = 0.1,
        ):
            """Initialize conversation agent.
            
            Args:
                chat_model: guidance-based chat model
                name: Agent name
                instructions: System instructions/role
                context_turns: Number of turns to keep in context
                temperature: Sampling temperature
            """
            self._chat_model = chat_model
            self._name = name
            self._instructions = instructions
            self._my_turns = []
            self._interlocutor_turns = []
            self._went_first = False
            self._context_turns = context_turns
            self.temperature = temperature

        @property
        def name(self) -> str:
            """Get agent name."""
            return self._name
        
        def get_conv(self) -> list:
            """Get conversation history."""
            return self._my_turns
            
        def reply(self, interlocutor_reply: str = None) -> str:
            """Generate reply to interlocutor.
            
            Args:
                interlocutor_reply: Previous interlocutor message or None to start
                
            Returns:
                Generated response
            """
            if interlocutor_reply is None:
                self._my_turns = []
                self._interlocutor_turns = []
                self._went_first = True
            else:
                self._interlocutor_turns.append(interlocutor_reply)

            # Get trimmed history
            my_hist = self._my_turns[-(self._context_turns):]
            interlocutor_hist = self._interlocutor_turns[-self._context_turns:]

            # Set up the system prompt
            curr_model = self._chat_model
            with system():
                curr_model += f"{self._instructions}"
            
            # Replay the last few turns
            for i in range(len(my_hist)):
                with user():
                    curr_model += interlocutor_hist[i]
                with assistant():
                    curr_model += my_hist[i]

            if len(interlocutor_hist) > 0:
                with user():
                    curr_model += interlocutor_hist[-1]
            
            with assistant():
                curr_model += gen(name='response', max_tokens=1024, temperature=self.temperature)

            self._my_turns.append(curr_model['response'])
            return curr_model['response']
else:
    class ConversationAgent:
        """Stub for ConversationAgent when guidance is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ConversationAgent requires guidance package. Install with: pip install guidance"
            )


# LLamaIndex-based Agents
try:
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.core.embeddings import resolve_embed_model
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.chat_engine import SimpleChatEngine
    
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


if LLAMAINDEX_AVAILABLE:
    
    def get_chat_engine_from_index_LlamaIndex(
        llm,
        index,
        chat_token_limit: int = 2500,
        verbose: bool = False,
        chat_mode: str = "context",
        system_prompt: str = 'You are a helpful assistant.',
    ):
        """Create chat engine from index.
        
        Args:
            llm: LLM instance
            index: Vector store index
            chat_token_limit: Token limit for chat memory
            verbose: Enable verbose output
            chat_mode: Chat mode ('context', 'condense', etc.)
            system_prompt: System prompt
            
        Returns:
            Chat engine instance
        """
        memory = ChatMemoryBuffer.from_defaults(token_limit=chat_token_limit)
        chat_engine = index.as_chat_engine(
            llm=llm,
            chat_mode=chat_mode,
            memory=memory,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        return chat_engine
        
    
    def get_answer_LlamaIndex(
        llm,
        q: str,
        system_prompt: str = "You are an expert.",
        chat_engine=None,
        max_new_tokens: int = 1024,
        messages_to_prompt=None,
        chat_token_limit: int = 2500,
        chat_mode: str = "context",
        completion_to_prompt=None,
        index=None,
        verbose: bool = False,
    ) -> tuple:
        """Get answer using LLamaIndex chat engine.
        
        Args:
            llm: LLM instance
            q: Question
            system_prompt: System prompt
            chat_engine: Existing chat engine or None
            max_new_tokens: Max tokens in response
            messages_to_prompt: Custom message formatter
            chat_token_limit: Chat memory token limit
            chat_mode: Chat mode
            completion_to_prompt: Custom completion formatter
            index: Vector index for RAG
            verbose: Enable verbose output
            
        Returns:
            Tuple of (response_text, chat_engine)
        """
        if chat_engine is None:
            if index is not None:
                chat_engine = get_chat_engine_from_index_LlamaIndex(
                    llm,
                    index,
                    chat_token_limit=chat_token_limit,
                    verbose=verbose,
                    chat_mode=chat_mode,
                    system_prompt=system_prompt,
                )
            else:
                chat_engine = SimpleChatEngine.from_defaults(
                    llm=llm,
                    system_prompt=system_prompt
                )
        
        response = chat_engine.stream_chat(q)
        for token in response.response_gen:
            print(token, end="")
        return response.response, chat_engine
    
    
    class ConversationAgent_LlamaIndex:
        """LLamaIndex-based conversation agent with RAG support."""
        
        def __init__(
            self,
            llm,
            name: str,
            instructions: str,
            index=None,
            chat_token_limit: int = 2500,
            verbose: bool = False,
            chat_mode: str = "context",
        ):
            """Initialize LLamaIndex agent.
            
            Args:
                llm: LLM instance
                name: Agent name
                instructions: System instructions
                index: Optional vector index for RAG
                chat_token_limit: Chat memory limit
                verbose: Verbose output
                chat_mode: Chat mode
            """
            self._name = name
            self._instructions = instructions
            self._source_nodes = []
           
            if index is not None:
                self.chat_engine = get_chat_engine_from_index_LlamaIndex(
                    llm,
                    index,
                    chat_token_limit=chat_token_limit,
                    verbose=verbose,
                    chat_mode=chat_mode,
                    system_prompt=self._instructions,
                )
            else:
                self.chat_engine = SimpleChatEngine.from_defaults(
                    llm=llm,
                    system_prompt=self._instructions
                )
        
        @property
        def name(self) -> str:
            """Get agent name."""
            return self._name
        
        def get_conv(self) -> list:
            """Get conversation history."""
            return self.chat_engine.chat_history
        
        def get_source_nodes(self) -> list:
            """Get source nodes from RAG."""
            return self._source_nodes
        
        def reset_chat(self):
            """Reset chat history."""
            self.chat_engine.reset()
            
        def reply(self, question: str) -> tuple:
            """Generate reply.
            
            Args:
                question: User question
                
            Returns:
                Tuple of (response_text, full_response_object)
            """
            response = self.chat_engine.stream_chat(question)
            for token in response.response_gen:
                print(token, end="")
            
            self._source_nodes.append(response.source_nodes)
            return response.response, response


# Utility functions for history and summarization
def get_entire_conversation(
    q: str,
    conversation_turns: list,
    marker_ch: str = '### ',
    start_with_q: bool = False,
    question_gpt_name: str = 'Question:',
) -> str:
    """Format conversation as text.
    
    Args:
        q: Initial question
        conversation_turns: List of turn dicts with 'name' and 'text'
        marker_ch: Marker character for formatting
        start_with_q: Include question in output
        question_gpt_name: Name for question asker
        
    Returns:
        Formatted conversation text
    """
    txt = ''
    
    if start_with_q:
        txt += f"{marker_ch}The question discussed is: {q.strip()}\n\n"
    else:
        txt += f"{marker_ch}{question_gpt_name}: {q.strip()}\n\n"
    
    for turn in conversation_turns:
        txt += f"{marker_ch}{turn['name'].strip()}: {turn['text']}\n\n"
    
    return txt


def read_and_summarize(gpt, txt: str = 'This is a conversation.', q: str = '') -> tuple:
    """Summarize conversation text.
    
    Args:
        gpt: Generate function or model
        txt: Conversation text
        q: Original question
        
    Returns:
        Tuple of (summary, bullet_points, key_takeaway)
    """
    with system():
        lm = gpt + "Analyze text and provide accurate account from all sides."
    
    with user():        
        lm += f"""Carefully read this conversation: 

<<<{txt}>>>

Summarize and identify key points. Think step by step: 
""" 
        
    with assistant():        
        lm += gen('summary', max_tokens=1024)
    
    with user():        
        lm += f'List the salient insights as bullet points.'
        
    with assistant():        
        lm += gen('bullet', max_tokens=1024)
     
    with user():        
        lm += f'Identify the single most important takeaway and how it answers: <<<{q}>>>.'
        
    with assistant():        
        lm += gen('takeaway', max_tokens=1024)

    return lm['summary'], lm['bullet'], lm['takeaway']


if LLAMAINDEX_AVAILABLE:
    def read_and_summarize_LlamaIndex(
        llm,
        txt: str = 'This is a conversation.',
        q: str = '',
    ) -> tuple:
        """Summarize conversation using LLamaIndex.
        
        Args:
            llm: LLM instance
            txt: Conversation text
            q: Original question
            
        Returns:
            Tuple of (summary, bullet_points, key_takeaway)
        """
        query = f"""Carefully read this conversation: 

>>>{txt}<<<

Accurately summarize and identify key points.
""" 

        summary, chat_engine = get_answer_LlamaIndex(
            llm,
            q=query,
            system_prompt="Analyze text and provide accurate account.",
        )

        query = 'Now list the salient insights as bullet points.'
        bullet, chat_engine = get_answer_LlamaIndex(
            llm,
            q=query,
            system_prompt="Analyze text and provide accurate account.",
            chat_engine=chat_engine,
        )

        query = f'Identify the single most important takeaway and how it answers: <<<{q}>>>.'
        takeaway, chat_engine = get_answer_LlamaIndex(
            llm,
            q=query,
            system_prompt="Analyze text and provide accurate account.",
            chat_engine=chat_engine,
        )

        return summary, bullet, takeaway
