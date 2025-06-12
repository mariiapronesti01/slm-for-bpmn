import numpy as np
import networkx as nx
import os
import pandas as pd

from sentence_transformers import util

def get_TypeSimilarityMatrix(G1: nx.DiGraph, G2: nx.DiGraph, degree_G1: dict, degree_G2: dict):
    """""
    Compute the similarity between the types of two nodes of two processes.
    
    Input:
    - G1: a networkx directed graph representing the first process
    - G2: a networkx directed graph representing the second process
    - degree_G1: a dictionary containing the in and out degree of the nodes of the first process
    - degree_G2: a dictionary containing the in and out degree of the nodes of the second process
    
    Output:
    - type_similarity: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity between the types of the nodes of the two processes
    """""
    type_similarity = np.zeros((len(G1.nodes), len(G2.nodes)))

    for i, node_i in enumerate(G1.nodes):
        node_i_type = G1.nodes[node_i]['type']
        node_i_general_type = G1.nodes[node_i]['general_type']
        in_degree_i, out_degree_i = degree_G1[node_i]

        for j, node_j in enumerate(G2.nodes):
            # Compute type similarity
            if node_i_type == G2.nodes[node_j]['type']:
                type_similarity[i, j] = 1
            elif node_i_general_type == G2.nodes[node_j]['general_type'] and node_i_general_type == 'gateway':
                type_similarity[i, j] = 0.3
                # Additional gateway checks
                in_degree_j, out_degree_j = degree_G2[node_j]
                if in_degree_i == in_degree_j and out_degree_i == out_degree_j:
                    type_similarity[i, j] += 0.20
            elif node_i_general_type == G2.nodes[node_j]['general_type']:
                type_similarity[i, j] = 0.5
                
    return type_similarity


def get_LabelSimilarityMatrix(embedding1: dict, embedding2: dict):
    """
    Compute the similarity between the labels of two nodes of two processes.
    
    Input:
    - embedding1: a dictionary containing the embeddings of the nodes of the first process
    - embedding2: a dictionary containing the embeddings of the nodes of the second process
    
    Output:
    - label_similarity_matrix: a numpy array of shape ((len(embedding1), len(embedding2))) containing the cosine similarity between the labels of the nodes of the two processes
    """
    label_similarity_matrix = np.zeros((len(embedding1), len(embedding2)))

    for i, node_i in enumerate(embedding1):
        embedding_i = embedding1[node_i]
        for j, node_j in enumerate(embedding2):
            embedding_j = embedding2[node_j]
            label_similarity_matrix[i, j] = util.pytorch_cos_sim(embedding_i, embedding_j).item()
    return label_similarity_matrix


def get_NeighbourSimilarityMatrix(G1: nx.DiGraph, G2: nx.DiGraph, type_similarity_matrix: np.array, start_shortest_path_distance: np.array, end_shortest_path_distance: np.array):
    """
    
    Compute the similarity between the neighbours of each node of two processes.
    
    Input:
    - G1: a networkx directed graph representing the first process
    - G2: a networkx directed graph representing the second process
    - type_similarity_matrix: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity between the types of the nodes of the two processes
    - start_shortest_path_distance: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the shortest path distance between the start nodes of the two processes
    - end_shortest_path_distance: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the shortest path distance between the end nodes of the two processes    
    
    Output:
    - neighbour_similarity_matrix: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity score between the neighbours of the nodes of the two processes
    """
    # create mapping between node and index
    node_to_index_G1 = {node: i for i, node in enumerate(G1.nodes)} 
    node_to_index_G2 = {node: i for i, node in enumerate(G2.nodes)}

    # Compute the similarity between the neighbours of each node
    neighbour_similarity_matrix = np.zeros((len(G1.nodes), len(G2.nodes)))

    for i, node_i in enumerate(G1.nodes):
        for j, node_j in enumerate(G2.nodes):
            # Get the neighbours of the current nodes
            neighbours_i = list(G1.successors(node_i)) + list(G1.predecessors(node_i))
            neighbours_j = list(G2.successors(node_j)) + list(G2.predecessors(node_j))

            type_similarity = 0
            start_similarity = 0
            end_similarity = 0
            for neighbour_i in neighbours_i:
                for neighbour_j in neighbours_j:
                    type_similarity += type_similarity_matrix[node_to_index_G1[neighbour_i], node_to_index_G2[neighbour_j]]
                    start_similarity += start_shortest_path_distance[node_to_index_G1[neighbour_i], node_to_index_G2[neighbour_j]]
                    end_similarity += end_shortest_path_distance[node_to_index_G1[neighbour_i], node_to_index_G2[neighbour_j]]
            neighbour_similarity_matrix[i, j] = (type_similarity + start_similarity + end_similarity) / (3*len(neighbours_i)*len(neighbours_j))
    return neighbour_similarity_matrix


def get_ShortestPathDistanceMatrix(shortest_path_G1: dict, shortest_path_G2: dict):
    """
    Compute the similarity between the shortest path of each node of two processes.
    
    Input:
    - shortest_path_G1: a dictionary with the nodes as keys and the shortest path length from any start/end node as values for the first process
    - shortest_path_G2: a dictionary with the nodes as keys and the shortest path length from any start/end node as values for the second process
    
    Output:
    - shortest_path_distance: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity score between the shortest path of the nodes of the two processes
    """
    
    # Convert the shortest path dicts to numpy arrays for faster operations
    sp_G1_values = np.array(list(shortest_path_G1.values()))
    sp_G2_values = np.array(list(shortest_path_G2.values()))

    # Create a grid of differences and maximum values
    diff = np.abs(sp_G1_values[:, np.newaxis] - sp_G2_values)
    max_val = np.maximum(sp_G1_values[:, np.newaxis], sp_G2_values)

    shortest_path_distance = 1 - (diff / max_val)
    return shortest_path_distance


def get_NodeEmbeddingDict(lane):
    """
    Function that, given a dictionary containing the lane information and having the lane id as key, 
    returns a new dictionary in which the key is the node id and the value is the embedding of the name of the corresponding lane.
    """
    nodes_embeddings = {}
    for entry in lane.values():
        name_embedding = entry['name_embedding']
        for node in entry['nodes']:
            nodes_embeddings[node] = name_embedding
    return nodes_embeddings


def get_LaneSimilarityMatrix(G1, G2, lane1, lane2):
    """
    Function that computes the similarity between the lanes of two processes, by comparing the name of the lane to which each nodes in each graph belongs.
    """
    node_emb_1 = get_NodeEmbeddingDict(lane1)
    node_emb_2 = get_NodeEmbeddingDict(lane2)
    
    # Ensure the order of nodes from G1 and G2
    nodes_1 = list(G1.nodes())
    nodes_2 = list(G2.nodes())
    
    lane_similarity_matrix = np.zeros((len(nodes_1), len(nodes_2)))
    
    for i, node1 in enumerate(nodes_1):
        for j, node2 in enumerate(nodes_2):
            if node1 in node_emb_1 and node2 in node_emb_2:
                lane_similarity_matrix[i][j] = util.pytorch_cos_sim(node_emb_1[node1], node_emb_2[node2]).item()
            else:
                lane_similarity_matrix[i][j] = 0  # Assign default similarity value for missing nodes
    return lane_similarity_matrix


def get_2ProcessesSimilarity(info_process1: dict, info_process2: dict, type_weight, start_weight, end_weight, neighbor_weight, lane_weight, label_weight):
    """
    Compute the similarity score between two processes.
    
    Input:
    - info_process1: a dictionary containing the information of the first process
    - info_process2: a dictionary containing the information of the second process
    - type_weight: weight for the type similarity
    - start_weight: weight for the start shortest path distance
    - end_weight: weight for the end shortest path distance
    - neighbor_weight: weight for the neighbour similarity
    - lane_weight: weight for the lane similarity
    - label_weight: weight for the label similarity
    
    Output:
    - similarity_score: a float value representing the similarity score between the two processes
    """

    # Calculate the different similarity metrics
    type_similarity = get_TypeSimilarityMatrix(info_process1['G'], info_process2['G'], info_process1['degree'], info_process2['degree'])
    label_similarity = get_LabelSimilarityMatrix(info_process1['embeddings'], info_process2['embeddings'])
    start_shortest_path_distance = get_ShortestPathDistanceMatrix(info_process1['start_shortest_path'], info_process2['start_shortest_path'])
    end_shortest_path_distance = get_ShortestPathDistanceMatrix(info_process1['end_shortest_path'], info_process2['end_shortest_path'])
    neighbor_similarity = get_NeighbourSimilarityMatrix(info_process1['G'], info_process2['G'], type_similarity, start_shortest_path_distance, end_shortest_path_distance)
    
    if info_process1['lane_info'] and info_process2['lane_info']:
        lane_similarity_matrix = get_LaneSimilarityMatrix(info_process1['G'], info_process2['G'], info_process1['lane_info'], info_process2['lane_info'])
    elif info_process1['lane_info'] or info_process2['lane_info']:  
        lane_similarity_matrix = np.zeros((info_process1['G'].number_of_nodes(), info_process2['G'].number_of_nodes()))
    else:
        lane_similarity_matrix = np.ones((info_process1['G'].number_of_nodes(), info_process2['G'].number_of_nodes()))
    
    #Combine the similarities with the given weights
    structural_similarity_matrix = (
        label_weight * label_similarity + 
        type_weight * type_similarity + 
        start_weight * start_shortest_path_distance + 
        end_weight * end_shortest_path_distance +
        neighbor_weight * neighbor_similarity+
        lane_weight * lane_similarity_matrix
    )

    # STRUCTURAL SIMILARITY SCORE
    structural_max_row_mean = np.max(structural_similarity_matrix, axis=1).mean()
    structural_max_col_mean = np.max(structural_similarity_matrix, axis=0).mean()
    structural_similarity_score = (structural_max_row_mean + structural_max_col_mean) / 2
    
    return structural_similarity_score


def compute_StructuralSimilarity(generated_model_info, original_model_info, Wtype=0.20, Wstart=0.10, Wend=0.10, Wneighbor=0.10, Wlane=0.30, Wlabel=0.20):
    """
    Compare models and compute similarities (both structural & label) for each matching pair of processes.
    Then compute statistics (mean, std, max, min) over those values.
    
    Input:
    - generated_model_info: dictionary containing the information of the generated models
    - original_model_info: dictionary containing the information of the original models
    - Wtype: weight for the type similarity
    - Wstart: weight for the start shortest path distance
    - Wend: weight for the end shortest path distance
    - Wneighbor: weight for the neighbour similarity
    - Wlane: weight for the lane similarity
    - Wlabel: weight for the label similarity
    """
    
    process_similarities = {}

    # Compare models and compute similarities
    for gen_key in generated_model_info:
        for orig_key in original_model_info:
            # Match condition (adjust as needed)
            if os.path.splitext(os.path.basename(gen_key))[0] == os.path.basename(os.path.dirname(orig_key)):

                structural_similarity  = get_2ProcessesSimilarity(
                    generated_model_info[gen_key],
                    original_model_info[orig_key],
                    type_weight=Wtype,
                    start_weight=Wstart,
                    end_weight=Wend,
                    neighbor_weight=Wneighbor,
                    lane_weight=Wlane,
                    label_weight=Wlabel
                )
                
                # Identify a process_name for output
                process_name = os.path.basename(gen_key)
                
                # If this process_name hasn't appeared before, initialize it
                if process_name not in process_similarities:
                    process_similarities[process_name] = {
                        'structural': []
                    }
                
                # Append both structural and label similarities
                process_similarities[process_name]['structural'].append(structural_similarity)
    
    # Create a DataFrame that stores mean, std, max, and min for 
    process_data = []
    for p_name, sim_dict in process_similarities.items():
        structural_list = sim_dict['structural']
        
        process_data.append({
            "Process": p_name,
            "Mean_Struct": np.mean(structural_list),
            "Std_Struct":  np.std(structural_list),
            "Max_Struct":  np.max(structural_list),
            "Min_Struct":  np.min(structural_list)
        })
    
    df = pd.DataFrame(process_data)
    return df

