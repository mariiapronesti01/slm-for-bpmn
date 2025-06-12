import numpy as np
from utils.structural_correctness import BPMNChecker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict


def extract_bpmn_names(bpmn_dict):
    """
    Extracts lane names and task names from a BPMN dictionary structure.

    Parameters:
    bpmn_dict (dict): A dictionary containing the BPMN process definition

    Returns:
    tuple: (lane_names, task_names) where both are lists of strings
    """
    # Extract lane names
    lane_names = []
    if 'lanes' in bpmn_dict:
        lane_names = [lane['name'] for lane in bpmn_dict['lanes']]

    # Extract task names from all nodes
    supported_task = ['task', 'userTask', 'serviceTask', 'sendTask', 'receiveTask', 'manualTask', 'businessRuleTask', 'scriptTask']
    task_names = []
    task_types = []
    gateways = [ ]
    if 'nodes' in bpmn_dict:
        for node in bpmn_dict['nodes']:
            if node['BPMNtype'] in supported_task:
                task_names.append(node['name'])
                task_types.append(node['BPMNtype'])
            elif node['BPMNtype'] == 'exclusiveGateway' or node['BPMNtype'] == 'parallelGateway':
                gateways.append(node['BPMNtype'])

    return lane_names, task_names, task_types, gateways


def component_similarity_metric(ground_truth, predictions):
    """
    Calculate similarity between lists of actor names using sentence transformer embeddings,
    cosine similarity, and F1 score.

    Parameters:
    ground_truth (list): List of ground truth actor names
    predictions (list): List of predicted actor names

    Returns:
    dict: Dictionary containing matching pairs and evaluation metrics
    """
    # If either list is empty, handle appropriately
    if len(ground_truth) == 0 and len(predictions) == 0:
        f1=1
        return f1

    if len(ground_truth) == 0:
        f1=0
        return f1

    if len(predictions) == 0:
        f1=0
        return f1

    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model

    # Generate embeddings
    gt_embeddings = model.encode(ground_truth)
    pred_embeddings = model.encode(predictions)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(pred_embeddings, gt_embeddings)

    # Match each prediction to the most similar ground truth name
    matched_pairs = []
    used_gt_indices = set()

    # Sort predictions by their maximum similarity score (match most confident first)
    pred_max_scores = np.max(similarity_matrix, axis=1)
    sorted_pred_indices = np.argsort(-pred_max_scores)

    for pred_idx in sorted_pred_indices:
        # Get best matching ground truth index
        gt_similarities = similarity_matrix[pred_idx]

        # Sort ground truth indices by similarity
        sorted_gt_indices = np.argsort(-gt_similarities)

        # Find best available match
        best_gt_idx = None
        for gt_idx in sorted_gt_indices:
            if gt_idx not in used_gt_indices:
                similarity_score = gt_similarities[gt_idx]
                # Only match if similarity is above threshold
                if similarity_score >= 0.6:  # Threshold for semantic similarity
                    best_gt_idx = gt_idx
                    used_gt_indices.add(gt_idx)
                    break

        if best_gt_idx is not None:
            matched_pairs.append({
                "pred": predictions[pred_idx],
                "pred_clean": predictions[pred_idx],
                "gt": ground_truth[best_gt_idx],
                "gt_clean": ground_truth[best_gt_idx],
                "similarity": float(gt_similarities[best_gt_idx])
            })

    # Calculate metrics
    true_positives = len(matched_pairs)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def check_safeness(bpmn_json: Dict) -> bool:
    """
    Check if the process is safe (no flow can have multiple tokens).

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if the process is safe, False otherwise
    """
    checker = BPMNChecker(bpmn_json)

    # If no start events or infinite loop, return False
    if len(checker.start_events) < 1 or not checker.check_infinite_loop_dfs():
        return False

    # Simulate token flow from each start event
    for start_event in checker.start_events:
        _, _, token_counts = checker.simulate_token_flow(start_event)

        # Check if any flow has multiple tokens
        if any(count > 1 for count in token_counts.values()):
            return False

    return True

def check_are_there_gateway(bpmn_json: Dict) -> bool:
    """
    Check if there are any gateways in the BPMN model.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if gateways exist, False otherwise
    """
    checker = BPMNChecker(bpmn_json)
    gateway_types = {'parallelGateway', 'exclusiveGateway', 'inclusiveGateway'}
    return any(node['BPMNtype'] in gateway_types for node in checker.nodes.values())

def check_gateway_correctness(bpmn_json: Dict) -> bool:
    """
    Check if gateways are properly formed.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if gateways are correct, False otherwise
    """
    checker = BPMNChecker(bpmn_json)
    is_correct, _ = checker.check_gateway_correctness()
    return is_correct

def check_proper_synchronization(bpmn_json: Dict) -> bool:
    """
    Check if parallel paths are properly synchronized.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if synchronization is proper, False otherwise
    """
    checker = BPMNChecker(bpmn_json)
    is_synchronized, _ = checker.check_synchronization()
    return is_synchronized

def check_no_infinite_loop(bpmn_json: Dict) -> bool:
    """
    Check if there is NO cycle (infinite loop) in the BPMN model.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if no infinite loop exists, False otherwise
    """
    checker = BPMNChecker(bpmn_json)
    return checker.check_infinite_loop_dfs()

def check_option_to_complete(bpmn_json: Dict) -> bool:
    """
    Check if the process can reach an end event from start events.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if process can reach an end event, False otherwise
    """
    checker = BPMNChecker(bpmn_json)

    # Check for start events and no infinite loop
    if len(checker.start_events) < 1:
        return False

    if not checker.check_infinite_loop_dfs():
        return False

    # Simulate token flow from each start event
    for start_event in checker.start_events:
        can_complete, _, _ = checker.simulate_token_flow(start_event)
        if can_complete:
            return True

    return False

def check_exist_end_event(bpmn_json: Dict) -> bool:
    """
    Check if there is at least one end event in the BPMN model.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if end events exist, False otherwise
    """
    checker = BPMNChecker(bpmn_json)
    return len(checker.end_events) > 0

def check_no_dead_activities(bpmn_json: Dict) -> bool:
    """
    Check if all nodes are reachable from start events.

    Args:
        bpmn_json (Dict): BPMN model JSON representation

    Returns:
        bool: True if no dead activities exist, False otherwise
    """
    checker = BPMNChecker(bpmn_json)

    # If no start events or infinite loop, return False
    if len(checker.start_events) < 1 or not checker.check_infinite_loop_dfs():
        return False

    # Collect all reachable nodes
    reachable_nodes = set()
    for start_event in checker.start_events:
        _, reached, _ = checker.simulate_token_flow(start_event)
        reachable_nodes.update(reached)

    # Check if all nodes are reachable
    all_nodes = set(checker.nodes.keys())
    return len(all_nodes - reachable_nodes) == 0