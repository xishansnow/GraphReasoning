"""
Triple Extractor Module
========================

Extract knowledge graph triples directly from text chunks using LLM.

This module provides a streamlined interface for extracting (subject, predicate, object) 
triples from text chunks without the full graph construction pipeline. Useful for:
- Batch triple extraction from large corpora
- Custom graph construction workflows
- Fine-grained control over triple extraction process

Key Features:
- Direct chunk-to-triples extraction
- Schema validation support
- Entity normalization
- Batch processing with progress tracking
- Flexible output formats (list, DataFrame, JSON)

Author: GraphConstruct Package
"""

import json
import uuid
from typing import List, Dict, Union, Callable, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Llms.prompt_templates import render_prompt


class TripleExtractor:
    """
    Extract knowledge graph triples from text chunks.
    
    This class provides methods to extract (subject, predicate, object) triples
    from text chunks using LLM-based extraction with optional schema validation.
    
    Attributes:
        generate (callable): LLM generation function
        schema (GraphSchema): Optional schema for validation
        normalize_entities (bool): Whether to normalize entity names
        validate_triples (bool): Whether to validate against schema
    """
    
    def __init__(self, 
                 generate: Callable,
                 schema: 'GraphSchema' = None,
                 normalize_entities: bool = True,
                 validate_triples: bool = False,
                 verbatim: bool = False):
        """
        Initialize triple extractor.
        
        Args:
            generate (callable): Function to generate text from LLM.
                Should accept system_prompt and prompt parameters.
            schema (GraphSchema): Schema for validation. If None and validate_triples=True,
                a default schema will be used.
            normalize_entities (bool): Whether to normalize entity names (lowercase, strip).
            validate_triples (bool): Whether to validate triples against schema.
            verbatim (bool): Whether to print detailed extraction logs.
        """
        self.generate = generate
        self.schema = schema
        self.normalize_entities = normalize_entities
        self.validate_triples = validate_triples
        self.verbatim = verbatim
        
        # Import schema and validation functions if needed
        if validate_triples and schema is None:
            from GraphConstruct.graph_generation import GraphSchema
            self.schema = GraphSchema()
    
    def extract_from_chunk(self, 
                          chunk: str,
                          chunk_id: str = None,
                          repeat_refine: int = 0,
                          metadata: dict = None) -> List[Dict]:
        """
        Extract triples from a single text chunk.
        
        Args:
            chunk (str): Text chunk to extract triples from.
            chunk_id (str): Unique identifier for the chunk. If None, auto-generated.
            repeat_refine (int): Number of refinement iterations (0 = no refinement).
            metadata (dict): Additional metadata to attach to each triple.
        
        Returns:
            List[Dict]: List of triple dictionaries with keys:
                - node_1: subject entity
                - node_2: object entity
                - edge: predicate/relation
                - chunk_id: source chunk identifier
                - (additional metadata keys if provided)
        """
        # Generate chunk ID if not provided
        if chunk_id is None:
            chunk_id = uuid.uuid4().hex
        
        # Prepare metadata
        meta = metadata or {}
        meta["chunk_id"] = chunk_id
        
        # Extract triples using LLM
        triples = self._llm_extract(chunk, meta, repeat_refine)
        
        if triples is None:
            return []
        
        # Apply post-processing
        if self.normalize_entities:
            triples = self._normalize_triples(triples)
        
        if self.validate_triples and self.schema:
            triples = self._validate_triples(triples)
        
        return triples
    
    def extract_from_chunks(self,
                           chunks: List[str],
                           chunk_ids: List[str] = None,
                           repeat_refine: int = 0,
                           show_progress: bool = True,
                           metadata: Union[dict, List[dict]] = None) -> List[Dict]:
        """
        Extract triples from multiple text chunks.
        
        Args:
            chunks (List[str]): List of text chunks.
            chunk_ids (List[str]): List of chunk identifiers. If None, auto-generated.
            repeat_refine (int): Number of refinement iterations per chunk.
            show_progress (bool): Whether to show progress bar.
            metadata (Union[dict, List[dict]]): Metadata to attach. 
                If dict, same metadata for all chunks.
                If list, per-chunk metadata (must match length of chunks).
        
        Returns:
            List[Dict]: Combined list of all extracted triples.
        """
        # Validate inputs
        if chunk_ids is None:
            chunk_ids = [uuid.uuid4().hex for _ in range(len(chunks))]
        
        if len(chunk_ids) != len(chunks):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks but {len(chunk_ids)} IDs")
        
        # Handle metadata
        if isinstance(metadata, dict):
            metadata_list = [metadata.copy() for _ in range(len(chunks))]
        elif isinstance(metadata, list):
            if len(metadata) != len(chunks):
                raise ValueError(f"Length mismatch: {len(chunks)} chunks but {len(metadata)} metadata entries")
            metadata_list = metadata
        else:
            metadata_list = [None] * len(chunks)
        
        # Extract triples from each chunk
        all_triples = []
        iterator = zip(chunks, chunk_ids, metadata_list)
        
        if show_progress:
            iterator = tqdm(list(iterator), desc="Extracting triples")
        
        for chunk, cid, meta in iterator:
            triples = self.extract_from_chunk(
                chunk=chunk,
                chunk_id=cid,
                repeat_refine=repeat_refine,
                metadata=meta
            )
            all_triples.extend(triples)
        
        return all_triples
    
    def extract_to_dataframe(self,
                            chunks: Union[str, List[str]],
                            **kwargs) -> pd.DataFrame:
        """
        Extract triples and return as DataFrame.
        
        Args:
            chunks (Union[str, List[str]]): Single chunk or list of chunks.
            **kwargs: Additional arguments passed to extract methods.
        
        Returns:
            pd.DataFrame: DataFrame with columns node_1, node_2, edge, chunk_id, etc.
        """
        # Handle single chunk
        if isinstance(chunks, str):
            triples = self.extract_from_chunk(chunks, **kwargs)
        else:
            triples = self.extract_from_chunks(chunks, **kwargs)
        
        # Convert to DataFrame
        if not triples:
            return pd.DataFrame(columns=["node_1", "node_2", "edge", "chunk_id"])
        
        df = pd.DataFrame(triples)
        
        # Clean up
        df.replace("", np.nan, inplace=True)
        df.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
        
        return df
    
    def _llm_extract(self, chunk: str, metadata: dict, repeat_refine: int) -> Optional[List[Dict]]:
        """
        Core LLM-based triple extraction logic.
        
        This method follows the same multi-step refinement process as graphPrompt
        in graph_generation.py:
        1. Initial extraction
        2. Format improvement
        3. Format fixing
        4. Optional refinement iterations
        
        Args:
            chunk (str): Text chunk to process.
            metadata (dict): Metadata to attach to triples.
            repeat_refine (int): Number of refinement iterations.
        
        Returns:
            Optional[List[Dict]]: List of triple dictionaries or None on error.
        """
        if self.verbatim:
            print(f"\n{'='*60}")
            print(f"Processing chunk: {metadata.get('chunk_id', 'unknown')}")
            print(f"{'='*60}")
        
        #############################################################
        # Step 1: Generate initial graph triplets using LLM
        #############################################################
        sys_prompt, user_prompt = render_prompt(name="graph_maker_initial", input=chunk)
        
        if not self.verbatim:
            print(".", end="")
        
        response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
        
        if self.verbatim:
            print(f"\n[Step 1] Initial extraction:\n{response[:200]}...")
        
        #############################################################
        # Step 2: Format response using LLM
        #############################################################
        sys_prompt, user_prompt = render_prompt(
            name="graph_format", 
            input=chunk, 
            ontology=response
        )
        
        response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
        
        if self.verbatim:
            print(f"\n[Step 2] After format improvement:\n{response[:200]}...")
        
        #############################################################
        # Step 3: Ensure proper format
        #############################################################
        sys_prompt, user_prompt = render_prompt(
            name="graph_fix_format", 
            ontology=response
        )
        
        response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
        response = response.replace('\\', '')
        
        if self.verbatim:
            print(f"\n[Step 3] After format fixing:\n{response[:200]}...")
        
        #############################################################
        # Step 4: Optional refinement iterations
        #############################################################
        if repeat_refine > 0:
            if self.verbatim:
                print(f"\n[Step 4] Running {repeat_refine} refinement iterations...")
            
            refine_iterator = range(repeat_refine)
            if self.verbatim:
                refine_iterator = tqdm(refine_iterator, desc="Refining")
            
            for rep in refine_iterator:
                # Add new triplets
                sys_prompt, user_prompt = render_prompt(
                    "graph_add_triplets", 
                    input=chunk, 
                    ontology=response
                )
                response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
                
                # Fix format after adding
                sys_prompt, user_prompt = render_prompt(
                    "graph_fix_format", 
                    ontology=response
                )
                response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
                response = response.replace('\\', '')
                
                # Refine nodes and edges
                sys_prompt, user_prompt = render_prompt(
                    "graph_refine", 
                    input=chunk, 
                    ontology=response
                )
                response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
                
                if self.verbatim:
                    print(f"\n[Refine {rep+1}/{repeat_refine}]:\n{response[:200]}...")
        
        #############################################################
        # Step 5: Final format fixing and parsing
        #############################################################
        sys_prompt, user_prompt = render_prompt("graph_fix_format", ontology=response)
        response = self.generate(system_prompt=sys_prompt, prompt=user_prompt)
        response = response.replace('\\', '')
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            response_cleaned = self._extract_json(response)
            result = json.loads(response_cleaned)
            
            # Attach metadata to each triple
            result = [dict(item, **metadata) for item in result]
            
            if self.verbatim:
                print(f"\n✅ Extracted {len(result)} triples")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON parsing error: {e}")
            if self.verbatim:
                print(f"Response: {response}")
            return None
        except Exception as e:
            print(f"\n❌ Extraction error: {e}")
            return None
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON array from text response.
        
        Handles cases where LLM returns JSON wrapped in markdown code blocks
        or with additional text.
        
        Args:
            text (str): Raw LLM response.
        
        Returns:
            str: Cleaned JSON string.
        """
        # Remove markdown code blocks
        text = text.replace("```json", "").replace("```", "")
        
        # Find JSON array boundaries
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx+1]
        
        return text
    
    def _normalize_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Normalize entity names in triples.
        
        Args:
            triples (List[Dict]): List of triple dictionaries.
        
        Returns:
            List[Dict]: Normalized triples with deduplicated entries.
        """
        from GraphConstruct.graph_generation import normalize_entity_names
        return normalize_entity_names(triples)
    
    def _validate_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Validate triples against schema.
        
        Args:
            triples (List[Dict]): List of triple dictionaries.
        
        Returns:
            List[Dict]: Filtered list of valid triples.
        """
        from GraphConstruct.graph_generation import validate_and_filter_triples
        return validate_and_filter_triples(
            triples, 
            schema=self.schema, 
            verbatim=self.verbatim
        )


# Convenience functions for quick usage
def extract_triples_from_chunk(chunk: str,
                               generate: Callable,
                               chunk_id: str = None,
                               schema: 'GraphSchema' = None,
                               validate: bool = False,
                               normalize: bool = True,
                               repeat_refine: int = 0,
                               verbatim: bool = False) -> List[Dict]:
    """
    Quick function to extract triples from a single chunk.
    
    Args:
        chunk (str): Text chunk to process.
        generate (callable): LLM generation function.
        chunk_id (str): Chunk identifier (auto-generated if None).
        schema (GraphSchema): Schema for validation.
        validate (bool): Whether to validate against schema.
        normalize (bool): Whether to normalize entity names.
        repeat_refine (int): Number of refinement iterations.
        verbatim (bool): Print detailed logs.
    
    Returns:
        List[Dict]: Extracted triples.
    
    Example:
        >>> from Llms.llm_providers import get_generator
        >>> generate = get_generator("openai", model="gpt-4")
        >>> chunk = "Albert Einstein developed the theory of relativity in 1915."
        >>> triples = extract_triples_from_chunk(chunk, generate)
        >>> print(triples)
        [{'node_1': 'albert einstein', 'node_2': 'theory of relativity', 
          'edge': 'developed', 'chunk_id': '...'}]
    """
    extractor = TripleExtractor(
        generate=generate,
        schema=schema,
        normalize_entities=normalize,
        validate_triples=validate,
        verbatim=verbatim
    )
    
    return extractor.extract_from_chunk(
        chunk=chunk,
        chunk_id=chunk_id,
        repeat_refine=repeat_refine
    )


def extract_triples_from_chunks(chunks: List[str],
                                generate: Callable,
                                chunk_ids: List[str] = None,
                                schema: 'GraphSchema' = None,
                                validate: bool = False,
                                normalize: bool = True,
                                repeat_refine: int = 0,
                                show_progress: bool = True,
                                verbatim: bool = False) -> List[Dict]:
    """
    Quick function to extract triples from multiple chunks.
    
    Args:
        chunks (List[str]): List of text chunks.
        generate (callable): LLM generation function.
        chunk_ids (List[str]): Chunk identifiers (auto-generated if None).
        schema (GraphSchema): Schema for validation.
        validate (bool): Whether to validate against schema.
        normalize (bool): Whether to normalize entity names.
        repeat_refine (int): Number of refinement iterations per chunk.
        show_progress (bool): Show progress bar.
        verbatim (bool): Print detailed logs.
    
    Returns:
        List[Dict]: Combined list of all extracted triples.
    
    Example:
        >>> chunks = [
        ...     "Einstein developed relativity.",
        ...     "Newton discovered gravity."
        ... ]
        >>> triples = extract_triples_from_chunks(chunks, generate)
        >>> len(triples)
        2
    """
    extractor = TripleExtractor(
        generate=generate,
        schema=schema,
        normalize_entities=normalize,
        validate_triples=validate,
        verbatim=verbatim
    )
    
    return extractor.extract_from_chunks(
        chunks=chunks,
        chunk_ids=chunk_ids,
        repeat_refine=repeat_refine,
        show_progress=show_progress
    )


def extract_triples_to_dataframe(chunks: Union[str, List[str]],
                                 generate: Callable,
                                 schema: 'GraphSchema' = None,
                                 validate: bool = False,
                                 normalize: bool = True,
                                 repeat_refine: int = 0,
                                 show_progress: bool = True,
                                 verbatim: bool = False) -> pd.DataFrame:
    """
    Extract triples and return as DataFrame.
    
    Args:
        chunks (Union[str, List[str]]): Single chunk or list of chunks.
        generate (callable): LLM generation function.
        schema (GraphSchema): Schema for validation.
        validate (bool): Whether to validate against schema.
        normalize (bool): Whether to normalize entity names.
        repeat_refine (int): Number of refinement iterations.
        show_progress (bool): Show progress bar (for multiple chunks).
        verbatim (bool): Print detailed logs.
    
    Returns:
        pd.DataFrame: DataFrame with columns node_1, node_2, edge, chunk_id.
    
    Example:
        >>> df = extract_triples_to_dataframe(
        ...     ["Einstein developed relativity.", "Newton discovered gravity."],
        ...     generate
        ... )
        >>> df[['node_1', 'edge', 'node_2']]
           node_1        edge              node_2
        0  einstein  developed  theory of relativity
        1  newton    discovered            gravity
    """
    extractor = TripleExtractor(
        generate=generate,
        schema=schema,
        normalize_entities=normalize,
        validate_triples=validate,
        verbatim=verbatim
    )
    
    return extractor.extract_to_dataframe(
        chunks=chunks,
        repeat_refine=repeat_refine,
        show_progress=show_progress
    )


def save_triples(triples: Union[List[Dict], pd.DataFrame],
                filepath: Union[str, Path],
                format: str = "csv") -> None:
    """
    Save extracted triples to file.
    
    Args:
        triples (Union[List[Dict], pd.DataFrame]): Triples to save.
        filepath (Union[str, Path]): Output file path.
        format (str): Output format - "csv", "json", "jsonl".
    
    Example:
        >>> triples = extract_triples_from_chunks(chunks, generate)
        >>> save_triples(triples, "output/triples.csv")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame if needed
    if isinstance(triples, list):
        df = pd.DataFrame(triples)
    else:
        df = triples
    
    # Save based on format
    if format == "csv":
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} triples to {filepath}")
    elif format == "json":
        df.to_json(filepath, orient="records", indent=2)
        print(f"✅ Saved {len(df)} triples to {filepath}")
    elif format == "jsonl":
        df.to_json(filepath, orient="records", lines=True)
        print(f"✅ Saved {len(df)} triples to {filepath}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'jsonl'")


def load_triples(filepath: Union[str, Path],
                format: str = None) -> pd.DataFrame:
    """
    Load triples from file.
    
    Args:
        filepath (Union[str, Path]): Input file path.
        format (str): Input format. If None, infer from extension.
    
    Returns:
        pd.DataFrame: Loaded triples.
    
    Example:
        >>> df = load_triples("output/triples.csv")
        >>> print(df.shape)
        (100, 4)
    """
    filepath = Path(filepath)
    
    # Infer format from extension
    if format is None:
        ext = filepath.suffix.lower()
        if ext == ".csv":
            format = "csv"
        elif ext == ".json":
            format = "json"
        elif ext == ".jsonl":
            format = "jsonl"
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}")
    
    # Load based on format
    if format == "csv":
        df = pd.read_csv(filepath)
    elif format == "json":
        df = pd.read_json(filepath, orient="records")
    elif format == "jsonl":
        df = pd.read_json(filepath, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✅ Loaded {len(df)} triples from {filepath}")
    return df
