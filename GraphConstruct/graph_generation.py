######################################################
# Imports
######################################################
# Import from GraphReasoning package (graph analysis tools)
import sys
from pathlib import Path
# Add parent directory to path to import from GraphReasoning
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphTools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

import copy

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import itertools
import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
import pandas as pd

import transformers
from transformers import logging

logging.set_verbosity_error()

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader

from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


####################################################################
# Graph generation functions
# Code based on: https://github.com/rahulnyk/knowledge_graph
#####################################################################

def extract (string, start='[', end=']'):
    """
    Extracts a substring from 'string' that starts with the first occurrence of 'start' and ends with the last occurrence of 'end'.
    """
    start_index = string.find(start)
    end_index = string.rfind(end)
     
    return string[start_index :end_index+1]

# Convert list of document chunks to dataframe
def documents2Dataframe(documents) -> pd.DataFrame:
    """Convert list of documents to dataframe.
    Args:
        documents (list): List of document chunks.
    Returns:
        pd.DataFrame: DataFrame with columns 'text' and 'chunk_id'.
    """
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
           # **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

# Convert list of concepts to dataframe
def concepts2Df(concepts_list) -> pd.DataFrame:
    """Convert list of concepts to dataframe.
    Args:
        concepts_list (list): List of concept dictionaries.
    Returns:
        pd.DataFrame: DataFrame with columns 'entity' and 'type'."""
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe

# Convert dataframe to list of graph triplets
def df2Graph(dataframe: pd.DataFrame, generate, repeat_refine=0, verbatim=False,
          
            ) -> list:
    """Convert dataframe of text chunks to list of graph triplets.
    Args:
        dataframe (pd.DataFrame): DataFrame with columns 'text' and 'chunk_id'.
        generate (function): Function to generate graph triplets from text.
    Returns:
        list: List of graph triplet dictionaries."""
  
    # Apply graphPrompt to each row in the dataframe
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, generate, {"chunk_id": row.chunk_id}, repeat_refine=repeat_refine,
                                verbatim=verbatim,#model
                               ), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


# Convert list of graph triplets to dataframe
def graph2Df(nodes_list) -> pd.DataFrame:
    """Convert list of graph triplets to dataframe.
    Args:
        nodes_list (list): List of graph triplet dictionaries.
    Returns:
        pd.DataFrame: DataFrame with columns 'node_1', 'node_2', and 'edge'."""
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe

import sys
from yachalk import chalk
sys.path.append("..")

import json
from Llms.prompt_templates import render_prompt


####################################################################
# Schema and Ontology Support
####################################################################

class GraphSchema:
    """
    Define a schema/ontology for knowledge graphs.
    Ensures consistency and validity of extracted triples.
    """
    
    def __init__(self, entity_types: dict = None, relation_types: dict = None):
        """
        Initialize schema with allowed entity and relation types.
        
        Args:
            entity_types (dict): Mapping of entity type names to allowed properties
                Example: {
                    "Person": {"properties": ["name", "age", "occupation"]},
                    "Organization": {"properties": ["name", "sector"]},
                    "Location": {"properties": ["name", "type"]}
                }
            relation_types (dict): Mapping of relation names to domain/range constraints
                Example: {
                    "works_for": {"domain": "Person", "range": "Organization"},
                    "located_in": {"domain": ["Organization", "Person"], "range": "Location"}
                }
        """
        # Default schema if none provided
        if entity_types is None:
            entity_types = {
                "Person": {"properties": ["name", "role"]},
                "Organization": {"properties": ["name", "type"]},
                "Location": {"properties": ["name", "type"]},
                "Concept": {"properties": ["name", "definition"]},
                "Event": {"properties": ["name", "date"]},
            }
        
        if relation_types is None:
            relation_types = {
                "works_for": {"domain": "Person", "range": "Organization"},
                "located_in": {"domain": ["Organization", "Person", "Event"], "range": "Location"},
                "part_of": {"domain": "Organization", "range": "Organization"},
                "participated_in": {"domain": "Person", "range": "Event"},
                "related_to": {"domain": ["Concept", "Person", "Organization"], "range": ["Concept", "Person", "Organization"]},
            }
        
        self.entity_types = entity_types
        self.relation_types = relation_types
    
    def get_entity_types(self) -> list:
        """Get list of allowed entity types."""
        return list(self.entity_types.keys())
    
    def get_relation_types(self) -> list:
        """Get list of allowed relation types."""
        return list(self.relation_types.keys())
    
    def validate_entity_type(self, entity_type: str) -> bool:
        """Check if entity type is in schema."""
        return entity_type in self.entity_types
    
    def validate_relation_type(self, relation_type: str) -> bool:
        """Check if relation type is in schema."""
        return relation_type in self.relation_types
    
    def validate_triple(self, subject: str, subject_type: str, predicate: str, 
                       object_: str, object_type: str) -> tuple:
        """
        Validate a triple against schema.
        
        Returns:
            (is_valid, error_message)
        """
        # Check subject type
        if not self.validate_entity_type(subject_type):
            return False, f"Invalid subject type: {subject_type}"
        
        # Check predicate type
        if not self.validate_relation_type(predicate):
            return False, f"Invalid predicate: {predicate}"
        
        # Check object type
        if not self.validate_entity_type(object_type):
            return False, f"Invalid object type: {object_type}"
        
        # Check domain/range constraints
        relation_info = self.relation_types[predicate]
        domain = relation_info.get("domain")
        range_ = relation_info.get("range")
        
        # Validate domain
        if isinstance(domain, str):
            domain = [domain]
        if subject_type not in domain:
            return False, f"Subject type {subject_type} not in domain of {predicate}"
        
        # Validate range
        if isinstance(range_, str):
            range_ = [range_]
        if object_type not in range_:
            return False, f"Object type {object_type} not in range of {predicate}"
        
        return True, "Valid"


def validate_and_filter_triples(triples: list, schema: GraphSchema = None, 
                                verbatim: bool = False) -> list:
    """
    Validate and filter triples against schema.
    
    Args:
        triples (list): List of triple dictionaries with keys:
            ["node_1", "node_2", "edge", "node_1_type", "node_2_type"]
        schema (GraphSchema): Schema to validate against
        verbatim (bool): Whether to print validation details
    
    Returns:
        list: Filtered list of valid triples
    """
    if schema is None:
        schema = GraphSchema()
    
    valid_triples = []
    invalid_count = 0
    
    for triple in triples:
        # Extract triple components
        subject = triple.get("node_1", "")
        object_ = triple.get("node_2", "")
        predicate = triple.get("edge", "")
        
        # Use provided types or default to "Concept"
        subject_type = triple.get("node_1_type", "Concept")
        object_type = triple.get("node_2_type", "Concept")
        
        # Validate triple
        is_valid, error_msg = schema.validate_triple(
            subject, subject_type, predicate, object_, object_type
        )
        
        if is_valid:
            valid_triples.append(triple)
        else:
            invalid_count += 1
            if verbatim:
                print(f"❌ Rejected: {subject} ({subject_type}) --{predicate}--> {object_} ({object_type})")
                print(f"   Reason: {error_msg}")
    
    if verbatim:
        print(f"\n✅ Valid triples: {len(valid_triples)}")
        print(f"❌ Invalid triples: {invalid_count}")
    
    return valid_triples


def normalize_entity_names(triples: list, normalize_fn=None) -> list:
    """
    Normalize entity names in triples to improve consistency.
    
    Args:
        triples (list): List of triple dictionaries
        normalize_fn (callable): Custom normalization function, defaults to lowercase + strip
    
    Returns:
        list: Triples with normalized entity names
    """
    if normalize_fn is None:
        normalize_fn = lambda x: str(x).strip().lower()
    
    normalized = []
    seen = {}  # Track normalization mappings
    
    for triple in triples:
        normalized_triple = triple.copy()
        
        # Normalize node names
        node_1 = normalize_fn(triple.get("node_1", ""))
        node_2 = normalize_fn(triple.get("node_2", ""))
        
        normalized_triple["node_1"] = node_1
        normalized_triple["node_2"] = node_2
        
        # Remove duplicates (same triple)
        triple_key = (node_1, triple.get("edge", ""), node_2)
        if triple_key not in seen:
            seen[triple_key] = True
            normalized.append(normalized_triple)
    
    return normalized


# Generate graph triplets from text using LLM and prompt templates
def graphPrompt(input: str, generate, metadata={}, #model="mistral-openorca:latest",
                repeat_refine=0,verbatim=False,
               )-> list:
    """Generate graph triplets from text using LLM.
    
    Uses centralized prompt templates for easy customization and maintenance.
    
    Args:
        input (str): Text chunk to generate graph from.
        generate (function): Function to generate graph triplets from text.
        metadata (dict): Additional metadata to include in results.
        repeat_refine (int): Number of refinement iterations.
        verbatim (bool): Whether to print debug output.
    
    Returns:
        list: List of graph triplet dictionaries."""
    
    #############################################################
    # Step 1: Generate initial graph triplets using LLM
    #############################################################
    
    sys_prompt, user_prompt = render_prompt(name="graph_maker_initial", input=input)
    
    print (".", end ="")    
    
    # Generate initial response
    response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
    
    # Print first response
    if verbatim:
        print ("---------------------\nFirst result: ", response)
   
    #############################################################
    # Step 2: Format response using LLM
    #############################################################
    sys_prompt, user_prompt = render_prompt(name="graph_format", input=input, ontology=response)
    
    # Generate improved response
    response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
    if verbatim:
        print ("---------------------\nAfter improve: ", response)
    
    #############################################################
    # Step 3: Ensure proper format
    #############################################################
    sys_prompt, user_prompt = render_prompt(name="graph_fix_format", ontology=response)
    
    # Generate formatted response
    response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
    response =   response.replace ('\\', '' )
    if verbatim:
        print ("---------------------\nAfter clean: ", response)
    
    #############################################################
    # Step 4: Repeat refinement if specified 
    # (repeat `repeat_refine` times)
    #############################################################
    if repeat_refine>0:
        for rep in tqdm(range (repeat_refine)):
            
            # Add new triplets
            sys_prompt, user_prompt = render_prompt("graph_add_triplets", input=input, ontology=response)
            response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
            if verbatim:
                print ("---------------------\nAfter adding triplets: ", response)
            
            # Fix format after adding
            sys_prompt, user_prompt = render_prompt("graph_fix_format", ontology=response)
            response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
            response =   response.replace ('\\', '' )
            
            # Refine nodes and edges
            sys_prompt, user_prompt = render_prompt("graph_refine", input=input, ontology=response)
            response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)            
            if verbatim:
                print (f"---------------------\nAfter refine {rep}/{repeat_refine}: ", response)

     
    sys_prompt, user_prompt = render_prompt("graph_fix_format", ontology=response)
    
    # Final generation of response
    response  =  generate(system_prompt=sys_prompt, prompt=user_prompt)
    response =   response.replace ('\\', '' )
    
    # Final parsing of response
    try:
        response=extract (response)
       
    except:
        print (end='')
    
    # Convert response to list of dictionaries
    try:
        result = json.loads(response)
        print (result)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

# Convert list of communities to dataframe with colors
def colors2Community(communities) -> pd.DataFrame:
    """Assign colors to communities.
    Args:
        communities (list): List of communities, each community is a list of nodes.
    Returns:
        pd.DataFrame: DataFrame with columns 'node', 'color', and 'group'"""
    
    # Generate color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

# Generate contextual proximity edges from graph dataframe
def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Generate contextual proximity edges from graph dataframe.
    Args:
        df (pd.DataFrame): DataFrame with columns 'node_1', 'node_2', 'edge', and 'chunk_id'.
    Returns:
        pd.DataFrame: DataFrame with columns 'node_1', 'node_2', 'edge', 'chunk_id', and 'count'."""
        
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2

# Generate graph from text using LLM and prompt templates    
def make_graph_from_text (txt, generate,
                          include_contextual_proximity=False,
                          graph_root='graph_root',
                          chunk_size=2500, chunk_overlap=0,
                          repeat_refine=0, verbatim=False,
                          data_dir='./data_output_KG/',
                          save_PDF=False,
                          save_HTML=True,
                          schema: GraphSchema = None,
                          validate_against_schema=False,
                          normalize_entities=True,
                         ):    
    """Generate graph from text using LLM with optional schema validation.
    
    Args:
        txt (str): Text to generate graph from.
        generate (function): Function to generate graph triplets from text.
        include_contextual_proximity (bool): Whether to include contextual proximity edges.
        graph_root (str): Root name for output files.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap between chunks.
        repeat_refine (int): Number of refinement iterations.
        verbatim (bool): Print verbose output.
        data_dir (str): Directory to save output files.
        save_PDF (bool): Whether to save PDF visualization.
        save_HTML (bool): Whether to save HTML visualization.
        schema (GraphSchema): Schema for validation. If None, default schema is used.
        validate_against_schema (bool): Whether to validate triples against schema.
        normalize_entities (bool): Whether to normalize entity names.
        
    Returns:
        tuple: (graph_HTML, graph_GraphML, G, net, output_pdf, validation_stats) 
    """
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
     
    outputdirectory = Path(f"./{data_dir}/")
    
    # Initialize schema if validation is enabled
    if validate_against_schema and schema is None:
        schema = GraphSchema()
    
 
    #############################################################
    # Step 1: Split text into chunks
    #############################################################    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )    
    
    pages = splitter.split_text(txt)
    print(f"✅ Number of chunks = {len(pages)}")
    
    # Display first chunk
    if verbatim:
        display(Markdown(pages[0]))
    
    # Convert documents to dataframe
    df = documents2Dataframe(pages)

    #############################################################
    # Step 2: Generate graph from chunks
    #############################################################
    regenerate = True
    
    if regenerate:
        # Generate graph from dataframe
        concepts_list = df2Graph(df, generate, repeat_refine=repeat_refine, verbatim=verbatim)
        
        #############################################################
        # Step 2a: Schema Validation and Normalization
        #############################################################
        validation_stats = {
            "total_extracted": len(concepts_list),
            "after_normalization": len(concepts_list),
            "after_validation": len(concepts_list),
        }
        
        # Normalize entity names for consistency
        if normalize_entities:
            concepts_list = normalize_entity_names(concepts_list)
            validation_stats["after_normalization"] = len(concepts_list)
            if verbatim:
                print(f"✅ After normalization: {len(concepts_list)} triples")
        
        # Validate against schema
        if validate_against_schema and schema:
            concepts_list = validate_and_filter_triples(concepts_list, schema, verbatim=verbatim)
            validation_stats["after_validation"] = len(concepts_list)
        
        if verbatim:
            print(f"\n📊 Validation Summary:")
            print(f"   Total extracted: {validation_stats['total_extracted']}")
            print(f"   After normalization: {validation_stats['after_normalization']}")
            print(f"   After validation: {validation_stats['after_validation']}")
        
        # Convert concepts list to dataframe
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        # Save graph and chunks dataframes to CSV
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False)
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph_clean.csv")
        df.to_csv(outputdirectory/f"{graph_root}_chunks_clean.csv")
        
        # Save validation stats
        with open(outputdirectory/f"{graph_root}_validation_stats.json", 'w') as f:
            json.dump(validation_stats, f, indent=2)
    else:
        dfg1 = pd.read_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|")
        validation_stats = {}
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print(f"✅ Shape of graph DataFrame: {dfg1.shape}")
        
    # Display first few rows of graph DataFrame
    dfg1.head()
    
    # Include contextual proximity edges if specified
    if include_contextual_proximity:
        # Generate contextual proximity edges
        dfg2 = contextual_proximity(dfg1)
        # Combine original graph edges with contextual proximity edges
        dfg = pd.concat([dfg1, dfg2], axis=0)
    else:
        dfg = dfg1
        
    # Group and aggregate edges
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )    
    
    # Display shape of final graph DataFrame    
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print ("Nodes shape: ", nodes.shape)
    
    ##############################################
    # Step 3: Create NetworkX graph
    ##############################################
    G = nx.Graph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
        node_list.append (node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
         
        chunk_id_list.append (row['chunk_id'] )

    try:
            
        df_nodes = pd.DataFrame({"nodes": node_list} )    
        df_nodes.to_csv(f'{data_dir}/{graph_root}_nodes.csv')
        df_nodes.to_json(f'{data_dir}/{graph_root}_nodes.json')
        
        df_edges = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge_list": title_list, "weight_list": weight_list } )    
        df_edges.to_csv(f'{data_dir}/{graph_root}_edges.csv')
        df_edges.to_json(f'{data_dir}/{graph_root}_edges.json')
        
    except:
        
        print ("Error saving CSV/JSON files.")
    
    #################################################
    # Step 4: Community detection using Girvan-Newman
    #################################################
    communities_generator = nx.community.girvan_newman(G)
    #top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    
    if verbatim:
        print("Number of Communities = ", len(communities))
        
    if verbatim:
        print("Communities: ", communities)
    
    colors = colors2Community(communities)
    if verbatim:
        print ("Colors: ", colors)
    
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
    
    ##############################################
    # Step 5: Visualize graph using PyVis
    # Network is core class from pyvis library
    ##############################################        
    net = Network(
             
            notebook=True,
         
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            
            filter_menu=False,
        )
        
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
   
    net.show_buttons()
    
    graph_HTML= f'{data_dir}/{graph_root}_grapHTML.html'
    graph_GraphML=  f'{data_dir}/{graph_root}_graphML.graphml'  #  f'{data_dir}/resulting_graph.graphml',
    nx.write_graphml(G, graph_GraphML)
    
    if save_HTML:
        net.show(graph_HTML)

    if save_PDF:
        output_pdf=f'{data_dir}/{graph_root}_PDF.pdf'
        pdfkit.from_file(graph_HTML, output_pdf)
    else:
        output_pdf=None
        
    # Compute graph statistics
    res_stat = graph_statistics_and_plots_for_large_graphs(
        G, data_dir=data_dir, include_centrality=False, make_graph_plot=False
    )
        
    print(f"✅ Graph statistics: {res_stat}")
    
    return graph_HTML, graph_GraphML, G, net, output_pdf, validation_stats

import time
from copy import deepcopy
import traceback

# Add new subgraph generated from text to an existing graph
def add_new_subgraph_from_text(txt,generate,node_embeddings,tokenizer, model,
                               original_graph_path_and_fname,
                               data_dir_output='./data_temp/', verbatim=True,
                               size_threshold=10,chunk_size=10000,
                               do_Louvain_on_new_graph=True,include_contextual_proximity=False,
                               repeat_refine=0,similarity_threshold=0.95, do_simplify_graph=True,#whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,graph_GraphML_to_add=None,
                              ):
    """Add new subgraph generated from text to an existing graph.
    Args:
        txt (str): Text to generate new subgraph from.
        generate (function): Function to generate graph triplets from text.
        node_embeddings (dict): Dictionary of node embeddings for original graph.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for embedding model.
        model (transformers.PreTrainedModel): Embedding model used.
        original_graph_path_and_fname (str): Path to the original graph GraphML file.
        data_dir_output (str): Directory to save output files.
        verbatim (bool): Whether to print verbose output.
        size_threshold (int): Minimum size of fragments to keep.
        chunk_size (int): Size of text chunks for graph generation.
        do_Louvain_on_new_graph (bool): Whether to perform Louvain community detection on the new graph.
        include_contextual_proximity (bool): Whether to include contextual proximity edges.
        repeat_refine (int): Number of times to refine graph generation.
        similarity_threshold (float): Similarity threshold for graph simplification.
        do_simplify_graph (bool): Whether to simplify the graph.
        return_only_giant_component (bool): Whether to return only the giant component of the graph.
        save_common_graph (bool): Whether to save the common nodes subgraph.
        G_to_add (networkx.Graph): Graph object to add (if provided).
        graph_GraphML_to_add (str): Path to GraphML file of graph to add (if provided).
    Returns:
        tuple: (graph_GraphML, G_new, G_loaded, G_original, node_embeddings, res) where graph_GraphML is the path to the GraphML file of the new graph,
               G_new is the new combined graph object,
               G_loaded is the loaded graph object to add,
               G_original is the original graph object,
               node_embeddings is the updated node embeddings dictionary,
               res is the graph statistics of the new graph."""


    display (Markdown(txt[:256]+"...."))
    graph_GraphML=None
     
    G_new=None
    res=None
    assert not (G_to_add is not None and graph_GraphML_to_add is not None), "G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added."
 
    try:
        start_time = time.time() 
        idx=0
        
        if verbatim:
            print ("Now create or load new graph...")

        ##############################################
        # Generate or load new graph to be added
        ##############################################
        # If no graph is provided or pre-generated, create a new one from the supplied text
        if graph_GraphML_to_add is None and G_to_add is None:
            print ("Make new graph from text...")
            # Generate new graph from text
            _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                      include_contextual_proximity=include_contextual_proximity,                                      
                                     data_dir=data_dir_output,
                                     graph_root=f'graph_new_{idx}',                                    
                                     chunk_size=chunk_size,   
                                     repeat_refine=repeat_refine, 
                                     verbatim=verbatim,                                       
                                  )
            if verbatim:
                print ("Generated new graph from text provided: ", graph_GraphML_to_add)

        else:
            if verbatim:
                print ("Instead of generating graph, loading it or using provided graph...(any txt data provided will be ignored...)")

            if graph_GraphML_to_add!=None:
                print ("Loading graph: ", graph_GraphML_to_add)
        
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print ("ALERT: Graph generation failed...for idx=",idx)
    
    print ("Now add node to existing graph...")
    
    #############################################
    # Load original graph and add new graph
    #############################################
    try:
        # Load original graph        
        G = nx.read_graphml(original_graph_path_and_fname)
        
        # Load new graph to be added
        if G_to_add!=None:
            # Use provided graph directly
            G_loaded=H = deepcopy(G_to_add)
            if verbatim:
                print ("Using provided graph to add (any txt data provided will be ignored...)")
        else:
            # Load graph from provided GraphML file
            if verbatim:
                print ("Loading graph to be added either newly generated or provided.")
            G_loaded = nx.read_graphml(graph_GraphML_to_add)
        
        # Analyze new graph to be added        
        res_newgraph=graph_statistics_and_plots_for_large_graphs(
            G_loaded, data_dir=data_dir_output,
            include_centrality=False,
            make_graph_plot=False,
            root='new_graph')
        print (res_newgraph)
        
        # Combine original graph with new graph
        G_new = nx.compose(G,G_loaded)
        
        if save_common_graph:
            # Identify common nodes and save subgraph if specified
            print ("Identify common nodes and save...")
            try:
                
                common_nodes = set(G.nodes()).intersection(set(G_loaded.nodes()))
    
                subgraph = G_new.subgraph(common_nodes)
                graph_GraphML=  f'{data_dir_output}/{graph_root}_common_nodes_before_simple.graphml' 
                nx.write_graphml(subgraph, graph_GraphML)
            except: 
                print ("Common nodes identification failed.")
            print ("Done!")
        
        if verbatim:
            print ("Now update node embeddings")
        
        
        # Update node embeddings for the new combined graph        
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model)        
        print ("Done update node embeddings.")
        
        # Simplify graph if specified
        if do_simplify_graph:
            if verbatim:
                print ("Now simplify graph.")
            G_new, node_embeddings =simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                    similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                    verbatim=verbatim,)
            if verbatim:
                print ("Done simplify graph.")
            
        if verbatim:
            print ("Done update graph")
        
        # Remove small fragments if specified
        if size_threshold >0:
            if verbatim:
                print ("Remove small fragments")            
            G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
        
        # Return only giant component if specified
        if return_only_giant_component:
            if verbatim:
                print ("Select only giant component...")   
            connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
            G_new = G_new.subgraph(connected_components[0]).copy()
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
            
        print (".")
        
        # Perform Louvain community detection if specified
        if do_Louvain_on_new_graph:
            G_new=graph_Louvain (G_new, 
                      graph_GraphML=None)
            if verbatim:
                print ("Don Louvain...")

        print (".")
         
        # Save new combined graph to GraphML
        graph_root=f'graph'
        graph_GraphML=  f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        print (".")
        nx.write_graphml(G_new, graph_GraphML)
        print ("Done...written: ", graph_GraphML)
        
        # Compute graph statistics for the new combined graph
        res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='assembled')
        
        print ("Graph statistics: ", res)

    except:
        print ("Error adding new graph.")
        print (end="")

    return graph_GraphML, G_new, G_loaded, G, node_embeddings, res


def main():
    """Main function to test graph generation functionality."""
    
    # Import provider factory for pluggable LLMs (avoids circular imports)
    from Llms.llm_providers import get_generate_fn
    
    # Example text for testing
    test_text = """
    Silk is a natural protein fiber produced by silkworms. It has exceptional mechanical properties
    due to its hierarchical structure. Beta-sheets are the primary structural motif that provides
    strength to silk fibers. The crystalline regions formed by beta-sheets are embedded in an
    amorphous matrix. Spider silk is even stronger than silkworm silk and can stretch significantly
    before breaking. The combination of strength and elasticity makes silk an ideal material for
    various applications in biomaterials and tissue engineering.
    """
    
    # Choose provider here. Examples:
    # provider = 'openai' / 'deepseek' / 'qwen' / 'llama_cpp' / 'transformers'
    provider = 'openai'
    import os
    provider_config = {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo'),
        'organization': os.getenv('OPENAI_ORG', ''),
        # For deepseek/qwen, also set: 'base_url': os.getenv('LLM_BASE_URL')
        # For llama_cpp: 'model_path': '/path/to/model.gguf'
    }

    generate = get_generate_fn(provider, provider_config)
    
    print("Testing graph generation from text...")
    
    try:
        # Test make_graph_from_text function
        graph_html, graph_graphml, G, net, output_pdf = make_graph_from_text(
            txt=test_text,
            generate=generate,
            include_contextual_proximity=True,
            graph_root='test_graph',
            chunk_size=500,
            chunk_overlap=0,
            repeat_refine=0,
            verbatim=True,
            data_dir='./test_output/',
            save_PDF=False,
            save_HTML=True
        )
        
        print(f"\nGraph generation successful!")
        print(f"HTML file: {graph_html}")
        print(f"GraphML file: {graph_graphml}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        
    except Exception as e:
        print(f"Error during graph generation: {str(e)}")
        traceback.print_exc()

    # -------------------------------------------------
    # Test adding a new subgraph to the generated graph
    # -------------------------------------------------
    try:
        print("\nTesting add_new_subgraph_from_text...")

        # Lightweight encoder for embeddings
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Start with empty embeddings; they will be populated
        node_embeddings = {}

        new_text = (
            "Spider silk contains glycine-rich and alanine-rich regions. "
            "Glycine increases flexibility, while alanine forms beta-sheet crystals that add strength."
        )

        graph_GraphML_new, G_new, G_loaded, G_orig, node_embeddings, res = add_new_subgraph_from_text(
            txt=new_text,
            generate=generate,
            node_embeddings=node_embeddings,
            tokenizer=tokenizer,
            model=model,
            original_graph_path_and_fname=graph_graphml,
            data_dir_output='./test_output/',
            verbatim=True,
            size_threshold=0,
            chunk_size=300,
            do_Louvain_on_new_graph=False,
            include_contextual_proximity=False,
            repeat_refine=0,
            similarity_threshold=0.9,
            do_simplify_graph=False,
            return_only_giant_component=False,
            save_common_graph=False,
        )

        print("\nSubgraph augmentation successful!")
        print(f"Augmented graph saved to: {graph_GraphML_new}")
        print(f"Original nodes: {G_orig.number_of_nodes()} -> New nodes: {G_new.number_of_nodes()}")
        print(f"Original edges: {G_orig.number_of_edges()} -> New edges: {G_new.number_of_edges()}")

    except Exception as e:
        print(f"Error during subgraph augmentation: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()