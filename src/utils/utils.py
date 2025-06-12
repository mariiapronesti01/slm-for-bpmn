import os
import random
import re
import ast

import numpy as np
import pandas as pd
import networkx as nx

from utils.BPMN_parser import getProcessInfo_fromXML


def obtain_NxGraph(edge_df: pd.DataFrame):
    """
    Function that given an edgedf, returns a directed graph with the nodes and their attributes, compatible with the networkx library.
    
    Input:
    - edge_df: a pandas dataframe with the following columns
        - sourceRef, sourceType, sourceName
        - targetRef, targetType, targetName
    
    Output:
    - G: a networkx directed graph with nodes and their attributes
    """
    
    # Create a new df storing only attributes of nodes
    sourceAtt = edge_df[['sourceRef', 'sourceType', 'sourceName']].rename(columns={'sourceRef': 'id', 'sourceType': 'type', 'sourceName': 'name'})
    targetAtt = edge_df[['targetRef', 'targetType', 'targetName']].rename(columns={'targetRef': 'id', 'targetType': 'type', 'targetName': 'name'})

    # Concatenate source_df and target_df to stack them vertically
    att_df = pd.concat([sourceAtt, targetAtt], ignore_index=True)
    att_df.drop_duplicates(inplace=True)
    att_df.set_index('id', inplace=True)

    # Add a column to store the general type of the node
    att_df['general_type'] = np.where(att_df['type'].str.contains('Event|event'), 'event', 
                                    np.where(att_df['type'].str.contains('Gateway|gateway'), 'gateway', 
                                    np.where(att_df['type'].str.contains('Task|task'), 'task',
                                    'None')))

    # Create a directed graph from the edge dataframe
    G = nx.from_pandas_edgelist(edge_df, source='sourceRef', target='targetRef', edge_attr=True, create_using=nx.DiGraph)
    # Add attributes to the nodes
    G.add_nodes_from((n, dict(d)) for n, d in att_df.iterrows())

    return G


def load_files_from_specific_folder(folder_path: str, type=None):
    """
    Function that loads all BPMN files from a specific folder.
    
    Input:
    - folder_path: the path to the folder containing the BPMN files 
    - type: extention of the files to be loaded (.bpmn, .mer, etc.)
    
    Output:
    - bpmn_files: a list of paths to the BPMN files
    """
    
    # List all files in the given folder
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            # Join root and filename to get the full file path
            files.append(os.path.join(root, filename))
        if type != None:
            files = [file for file in files if file.endswith(type)]
    return files


def find_shortest_path(G: nx.DiGraph):
    """
    Function that computes the shortest path length from any start node to any node in the graph and from any node to any end node.
    
    Input:
    - G: a networkx directed graph
    
    Output:
    - shortest_path_start: a dictionary with the nodes as keys and the shortest path length from any start node as values
    - shortest_path_end: a dictionary with the nodes as keys and the shortest path length to any end node as values
    """
    
    start_node = [node for node in G.nodes if G.in_degree(node) == 0]
    end_node = [node for node in G.nodes if G.out_degree(node) == 0]
    
    shortest_path_start = {}
    for node in G.nodes:
        # Find the minimum path length from any start node
        shortest_path_start[node] = min(
            [(nx.shortest_path_length(G, start, node)+1) for start in start_node if nx.has_path(G, start, node)],
            default=G.number_of_nodes()  # Changed infinity to number of nodes in the graph -- condition when there is no path from start to node
        )
    
    shortest_path_end = {}
    for node in G.nodes:
        # Find the minimum path length to any end node
        shortest_path_end[node] = min(
            [(nx.shortest_path_length(G, node, end)+1) for end in end_node if nx.has_path(G, node, end)], # add 1 to the path length to account for the node itself and avoid prolem with 0 length path 
            default=G.number_of_nodes()  # Changed infinity to number of nodes in the graph
        )
    
    return shortest_path_start, shortest_path_end


def read_file(file: str):
    """
    Function that reads a file and returns its content.
    
    Input:
    - file: a string containing the path to the file
    """
    with open(file, 'r') as f:
        return f.read()
    

def compute_labels_embeddings(model, G: nx.DiGraph):
    return {node : model.encode(G.nodes[node]['name'], convert_to_tensor=True) for node in G.nodes}

def compute_nodes_degree(G: nx.DiGraph):
    return {node: (G.in_degree(node), G.out_degree(node)) for node in G.nodes}

def compute_lanes_embeddings(model, lane_info: dict):
    if lane_info == {}:
        return lane_info
    else:
        for lane_id, lane in lane_info.items():
            lane_name = lane['name']
            lane_embedding = model.encode(lane_name, convert_to_tensor=True)
            lane_info[lane_id]['name_embedding'] = lane_embedding
            return lane_info
        
        
def build_process_info_dict(all_files: list, model, type: list["MER", "JSON", "XML"]):
    """
    Function that builds a dictionary containing the information extracted from the BPMN files.
    
    Input:
    - all_files: a list of paths to the BPMN/MER files
    - model: a sentence transformer model
    
    Output:
    - files_info: a dictionary containing the following information:
        - edge_df: a pandas dataframe with the edge information
        - lane_info: a dictionary containing the lane information
        - G: a networkx directed graph
        - start_shortest_path: a dictionary with the shortest path length from any start node
        - end_shortest_path: a dictionary with the shortest path length to any end node
        - embeddings: a dictionary containing the embeddings of the nodes
        - degree: a dictionary containing the in and out degree of the nodes
    """
    
    files_info = {}
    for file in all_files:
        #print("Processing file: ", file)
        
        if type == "MER":
            edge_df, lane_info = getProcessInfo_fromMER(file)
        elif type == "JSON":
            edge_df, lane_info = getProcessInfo_fromJSON(file)
        else:
            edge_df, lane_info = getProcessInfo_fromXML(file)
        
        graph = obtain_NxGraph(edge_df)
        shortest_path_start, shortest_path_end = find_shortest_path(graph)
        embeddings = compute_labels_embeddings(model, graph)
        degree = compute_nodes_degree(graph)
        
        #extract embeddings for lane names
        lane_info = compute_lanes_embeddings(model, lane_info)
            
        files_info[file] = {
            'file_name': file, # Assign to 'file_name' key
            'edge_df': edge_df,   # Assign to 'edge_df' key
            'lane_info': lane_info, # Assign to 'lane_info' key 
            'G' : graph, # Assign to 'graph' key
            'start_shortest_path' : shortest_path_start, # Assign to 'startShortestPath' key
            'end_shortest_path' : shortest_path_end, # Assign to 'endShortestPath' key
            'embeddings' : embeddings, # Assign to 'embeddings' key
            'degree' : degree # Assign to 'degree' key
        }
    return files_info


def getProcessInfo_fromJSON(file):
    """
    Reads a BPMN-like structure from 'file' and returns:
      1) A DataFrame describing each edge (source node -> target node).
      2) A dictionary of lane_info if lanes are present (empty if not).

    The JSON-like structure is expected to look like:
    {
        'start_nodes': [...],
        'nodes': [...],
        'end_nodes': [...],
        'lanes': [
            {
                'id': '<lane_id>',
                'name': '<lane_name>',
                'nodes': ['node_id_1', 'node_id_2', ...]
            },
            ...
        ]
    }
    """
    # Parse the file contents into a Python dict
    data = ast.literal_eval(read_file(file))
    
    edges = []
    lane_info = {}

    # Combine all node definitions into a single list
    # (start_nodes, nodes, end_nodes)
    all_nodes = data.get('start_nodes', []) \
                + data.get('nodes', []) \
                + data.get('end_nodes', [])
    # Create a dictionary to quickly look up a node by its ID
    nodes_by_id = {node['id']: node for node in all_nodes if 'id' in node}

    if 'lanes' in data and data['lanes']:
        # Format WITH lanes
        for lane in data['lanes']:
            lane_id = lane.get('id', '')
            lane_name = (lane.get('name') or '').strip()

            # Initialize lane info
            lane_info[lane_id] = {
                'name': lane_name,
                'nodes': []
            }

            # Each lane['nodes'] is now a list of node IDs
            for node_id in lane.get('nodes', []):
                if node_id in nodes_by_id:
                    node_obj = nodes_by_id[node_id]
                    # Record the node_id in this lane
                    lane_info[lane_id]['nodes'].append(node_id)

                    # Extract source info
                    source_type = node_obj.get('BPMNtype', '')
                    source_ref = node_obj.get('id', '')
                    source_name = (node_obj.get('name') or '').strip()

                    # Build edges for each 'outgoing' reference
                    for outgoing_id in node_obj.get('outgoing', []):
                        edge_id = f"{source_ref}_TO_{outgoing_id}"
                        target_obj = nodes_by_id.get(outgoing_id)
                        if target_obj:
                            target_type = target_obj.get('BPMNtype', '')
                            target_ref = target_obj.get('id', '')
                            target_name = (target_obj.get('name') or '').strip()
                            edges.append([
                                source_type, source_ref, source_name,
                                'sequenceFlow', edge_id,
                                target_type, target_ref, target_name
                            ])
    else:
        # Format WITHOUT lanes
        # Simply iterate over all nodes
        for node_obj in all_nodes:
            source_type = node_obj.get('BPMNtype', '')
            source_ref = node_obj.get('id', '')
            source_name = (node_obj.get('name') or '').strip()

            for outgoing_id in node_obj.get('outgoing', []):
                edge_id = f"{source_ref}_TO_{outgoing_id}"
                target_obj = nodes_by_id.get(outgoing_id)
                if target_obj:
                    target_type = target_obj.get('BPMNtype', '')
                    target_ref = target_obj.get('id', '')
                    target_name = (target_obj.get('name') or '').strip()
                    edges.append([
                        source_type, source_ref, source_name,
                        'sequenceFlow', edge_id,
                        target_type, target_ref, target_name
                    ])

    # Create a DataFrame with the gathered edge info
    df = pd.DataFrame(edges, columns=[
        "sourceType", "sourceRef", "sourceName",
        "edgeType",   "edgeID",
        "targetType", "targetRef", "targetName"
    ])

    return df, lane_info


def getProcessInfo_fromMER(file: str):     
    """
    Function that extracts the edgeDF and lane info from a MERMAID file.
    
    Input:
    - mer_file: path to the MERMAID file
    
    Output:
    - df: a pandas dataframe with the edge information
    - lane_info: a dictionary containing lane information
    """
    
    # Initialize a list to store the rows and a dictionary for lane info
    mer_file = read_file(file)
    rows = []
    lane_info = {}

    # Split the input string by newlines to get individual entries
    bpmn_list = mer_file.strip().split('\n')

    edge_pattern = r'(\w+_\d+)\((.*?)\)-->(\w+_\d+)\((.*?)\)'
    lane_pattern = r'lane_(\d+)\((.*?)\)'

    for entry in bpmn_list:
        edge_match = re.match(edge_pattern, entry)
        lane_match = re.match(lane_pattern, entry)
        
        if edge_match:
            source_id = edge_match.group(1)
            source_name = edge_match.group(2)
            target_id = edge_match.group(3)
            target_name = edge_match.group(4)
            edge_id = f"{source_id}_TO_{target_id}"

            
            source_type = (source_id.split('_')[0])
            target_type = (target_id.split('_')[0])
            edge_type ="sequenceFlow"
            
            rows.append([source_type, source_id, source_name, 
                         edge_type, edge_id,
                         target_type, target_id, target_name])
            
        elif lane_match:
            lane_id = 'lane_'+lane_match.group(1)
            lane_name = lane_match.group(2)
            lane_nodes = []
            
            # Collect node identifiers in the lane
            for node_entry in bpmn_list[bpmn_list.index(entry) + 1:]:
                if node_entry.strip() == 'end':
                    break  # Stop if we reach the end of the lane definition
                lane_nodes.append(node_entry.strip())

            # Store each lane's information in a dictionary, keyed by lane_id
            lane_info[lane_id] = {
                'name': lane_name,
                'nodes': lane_nodes
            }

    # Create a DataFrame from the rows
    columns = ['sourceType', 'sourceRef', 'sourceName', 
               'edgeType', 'edgeID',
               'targetType', 'targetRef', 'targetName']
    
    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates(subset=['sourceRef', 'targetRef'], keep='first')
    return df, lane_info


def get_filename_without_extension(filepath):
    """ 
    Function that extracts the base name of a file without the extension.

    Input:
    - filepath: a string containing the path to the file
    
    Output:
    - file_name: a string containing the base name of the
    """
    base_name = os.path.basename(filepath)
    file_name = os.path.splitext(base_name)[0]
    return file_name


def sort_by_prefix(file_list, source:str):
    """
    Function that sorts a list of file names by a numeric prefix.
    
    Input:
    - file_list: a list of file names
    
    Output:
    - sorted_list: a list of file names sorted by the numeric prefix
    """
    # Sort the list by extracting the numeric part of the prefix for each item
    sorted_list = sorted(file_list, key=lambda x: int(x.split(source)[0]))
    return sorted_list

