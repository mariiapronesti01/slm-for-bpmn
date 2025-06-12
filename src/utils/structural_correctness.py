from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import ast
import numpy as np
import sys

from utils.utils import read_file

class BPMNChecker:
    def __init__(self, bpmn_json: Dict):
        self.bpmn = bpmn_json
        self.nodes = {}
        self.start_events = []
        self.end_events = []
        self.initialize_nodes()
        
    def initialize_nodes(self):
        """Extract all nodes from the BPMN JSON structure."""
        for node in self.bpmn.get('start_nodes', []):
            self.add_node(node)
        for node in self.bpmn.get('nodes', []):
            self.add_node(node)
        for node in self.bpmn.get('end_nodes', []):
            self.add_node(node)
    
    def add_node(self, node: Dict):
        """Add a single node to the internal node collection."""
        self.nodes[node['id']] = node
        if node['BPMNtype'] == 'startEvent':
            self.start_events.append(node['id'])
        elif node['BPMNtype'] == 'endEvent':
            self.end_events.append(node['id'])

    def check_gateway_correctness(self) -> Tuple[bool, List[str]]:
        """
        Check if gateways are properly formed.
        Returns: (is_correct, list of issues)
        """
        issues = []
        gateway_types = {'parallelGateway', 'exclusiveGateway', 'inclusiveGateway'}
        
        for node_id, node in self.nodes.items():
            if node['BPMNtype'] in gateway_types:
                incoming = node.get('incoming', [])
                outgoing = node.get('outgoing', [])
                
                # Check for gateways with single flow
                if len(outgoing) == 1 and len(incoming) == 1:
                    issues.append(
                        f"Gateway {node_id} ({node['BPMNtype']}) has single incoming and outgoing flow"
                    )
                
                # Check for gateways with no outgoing flows
                if len(outgoing) == 0:
                    issues.append(
                        f"Gateway {node_id} ({node['BPMNtype']}) has no outgoing flows"
                    )
                
                # Check for gateways with no incoming flows
                if len(incoming) == 0:
                    issues.append(
                        f"Gateway {node_id} ({node['BPMNtype']}) has no incoming flows"
                    )
        
        return len(issues) == 0, issues

    def check_synchronization(self) -> Tuple[bool, List[str]]:
        """
        Check if parallel paths are properly synchronized.
        Returns: (is_properly_synchronized, list of issues)
        """
        issues = []
        
        # Find all parallel gateways that split the flow
        parallel_splits = {
            node_id: node for node_id, node in self.nodes.items()
            if node['BPMNtype'] == 'parallelGateway' and len(node['outgoing']) > 1
        }
        
        for split_id, split_node in parallel_splits.items():
            # For each parallel split, track paths until they converge
            paths = self.trace_parallel_paths(split_id)
            
            # Check convergence points
            for convergence_point in self.find_convergence_points(paths):
                node = self.nodes[convergence_point]
                
                # Check if convergence point is correct gateway type
                if node['BPMNtype'] != 'parallelGateway':
                    issues.append(
                        f"Parallel paths from {split_id} converge at {convergence_point} "
                        f"which is a {node['BPMNtype']} instead of a parallel gateway"
                    )
                
                # Check if all parallel paths reach this convergence point
                paths_to_convergence = sum(
                    1 for path in paths if convergence_point in path
                )
                if paths_to_convergence != len(split_node['outgoing']):
                    issues.append(
                        f"Not all parallel paths from {split_id} reach convergence point {convergence_point}"
                    )
        
        return len(issues) == 0, issues

    def trace_parallel_paths(self, start_id: str) -> List[Set[str]]:
        """Trace all paths from a parallel gateway split until they converge or end."""
        paths = []
        start_node = self.nodes[start_id]
        
        for outgoing in start_node['outgoing']:
            path = set()
            queue = deque([outgoing])
            
            while queue:
                current_id = queue.popleft()
                if current_id not in path:
                    path.add(current_id)
                    current_node = self.nodes[current_id]
                    
                    # Stop if we hit another parallel gateway
                    if current_node['BPMNtype'] != 'parallelGateway':
                        queue.extend(current_node['outgoing'])
            
            paths.append(path)
        
        return paths

    def find_convergence_points(self, paths: List[Set[str]]) -> Set[str]:
        """Find nodes where parallel paths converge."""
        node_count = defaultdict(int)
        for path in paths:
            for node in path:
                node_count[node] += 1
        
        return {
            node for node, count in node_count.items()
            if count > 1 and self.nodes[node]['BPMNtype'] != 'endEvent'
        }
        
    def check_infinite_loop_dfs(self) -> bool:
        """
        Check if there is any cycle (infinite loop) in the BPMN model 
        using DFS-based cycle detection (white-gray-black).
        
        Returns:
            True if there is NO cycle (i.e., the process is acyclic).
            False if a cycle (infinite loop) exists in the graph.
        """
        # 0 = White (unvisited), 1 = Gray (visiting), 2 = Black (fully visited)
        color = { node_id: 0 for node_id in self.nodes }

        def dfs_detect_cycle(node_id: str) -> bool:
            """
            Returns:
                True if a cycle is found,
                False otherwise.
            """
            # If we see a node that is Gray, we found a back edge -> cycle
            if color[node_id] == 1:
                return True  # cycle detected
            if color[node_id] == 2:
                return False  # already fully explored, no cycle from here

            # Mark the current node as Gray (in progress)
            color[node_id] = 1

            # Explore all outgoing edges
            for nxt in self.nodes[node_id].get('outgoing', []):
                if nxt not in self.nodes:
                    # If 'nxt' is an end event or unknown node, skip
                    # or handle as needed
                    continue

                if dfs_detect_cycle(nxt):
                    return True  # cycle found in a descendant

            # Mark the current node as Black (fully explored)
            color[node_id] = 2
            return False

        nodes_to_check = list(self.nodes.keys())

        for node_id in nodes_to_check:
            if color[node_id] == 0:  # White => not visited yet
                if dfs_detect_cycle(node_id):
                    return False  # cycle found -> infinite loop

        return True  # no cycles found


    def simulate_token_flow(self, start_node: str, max_depth: int = 1000) -> Dict[str, int]:
        """
        Simulate token flow from a start node to estimate max token counts per flow.
        
        Args:
            start_node (str): The ID of the starting node.
            max_depth (int): Maximum path depth to prevent infinite loops.

        Returns:
            Dict[str, int]: A dictionary mapping flow IDs to maximum token counts observed.
        """
        from collections import defaultdict

        max_tokens = defaultdict(int)
        visited_states = set()
        queue = deque([(start_node, 0, {start_node: 1})])  # (current_node, depth, token_state)

        while queue:
            current_node_id, depth, token_state = queue.popleft()

            if depth > max_depth:
                continue  # avoid infinite path exploration

            # Create a hashable representation of state to prevent revisiting
            state_signature = (current_node_id, frozenset(token_state.items()))
            if state_signature in visited_states:
                continue
            visited_states.add(state_signature)

            current_node = self.nodes[current_node_id]
            
            # Update max token count for current node
            for node_id, count in token_state.items():
                max_tokens[node_id] = max(max_tokens[node_id], count)

            outgoing_flows = current_node.get('outgoing', [])

            if current_node['BPMNtype'] == 'parallelGateway':
                # Split: duplicate tokens across all branches
                new_token_state = token_state.copy()
                for target in outgoing_flows:
                    new_tokens = new_token_state.copy()
                    new_tokens[target] = new_tokens.get(current_node_id, 1)
                    queue.append((target, depth + 1, new_tokens))

            elif current_node['BPMNtype'] in ['exclusiveGateway', 'inclusiveGateway']:
                # Route: select one or multiple paths (simulate all options)
                for target in outgoing_flows:
                    new_tokens = token_state.copy()
                    new_tokens[target] = new_tokens.get(current_node_id, 1)
                    queue.append((target, depth + 1, new_tokens))

            else:
                # Standard flow: continue with same tokens
                for target in outgoing_flows:
                    new_tokens = token_state.copy()
                    new_tokens[target] = new_tokens.get(current_node_id, 1)
                    queue.append((target, depth + 1, new_tokens))

        return dict(max_tokens)

    
    def check_integrated_properties(self) -> Dict[str, Dict]:

        queue = deque()
        reachable_nodes = set()
        problematic_end_events = set()
        visited_global = set()

        # Initialize BFS from each start event
        for start in self.start_events:
            queue.append((start, set(), defaultdict(int)))

        while queue:
            node_id, visited_path, end_event_counter = queue.popleft()

            if node_id not in self.nodes:
                continue

            state_key = (node_id, tuple(sorted(end_event_counter.items())))
            if state_key in visited_global:
                continue
            visited_global.add(state_key)

            node = self.nodes[node_id]
            visited_path = visited_path | {node_id}
            reachable_nodes.update(visited_path)

            # Track end events
            if node['BPMNtype'] == 'endEvent':
                end_event_counter[node_id] += 1
                if end_event_counter[node_id] > 1:
                    problematic_end_events.add(node_id)
                # Don't continue traversal after an end event (path terminates here)
                continue

            # Continue traversal
            for neighbor in node.get('outgoing', []):
                queue.append((neighbor, visited_path.copy(), end_event_counter.copy()))

        # Dead activities = nodes never reached
        dead_nodes = set(self.nodes.keys()) - reachable_nodes       
        
        can_complete = len(self.end_events) > 0 and any(e in reachable_nodes for e in self.end_events)
        proper_completion = len(problematic_end_events) == 0
        
        return can_complete, proper_completion, dead_nodes
    
    
    def check_SC(self) -> Dict[str, Dict]:
        """Check safeness and soundness properties of the BPMN model."""
        results = {
            'correct_synchronization': {'satisfied': True, 'details': []},
            'option_to_complete': {'satisfied': True, 'details': []},
            'no_dead_activities': {'satisfied': True, 'details': []},
            'safeness': {'satisfied': True, 'details': []},
            'proper_completion': {'satisfied': True, 'details': []},
        }
            
        # Check synchronization
        is_synchronized, sync_issues = self.check_synchronization()
        if not is_synchronized:
            results['correct_synchronization']['satisfied'] = False
            results['correct_synchronization']['details'].extend(sync_issues)
          
        can_complete, proper_completion, dead_nodes = self.check_integrated_properties()
        
        # Check if all process instance can reach an end event
        if len(self.start_events) <  1  or not can_complete:
            results['option_to_complete']['satisfied'] = False
            results['option_to_complete']['details'].append("Option to complete is not satisfied")
            
        # Check if there are any dead activities
        if len(dead_nodes) > 0:
            results['no_dead_activities']['satisfied'] = False
            results['no_dead_activities']['details'].append(
                f"Unreachable nodes: {', '.join(dead_nodes)}"
            )
            
        # Check safeness using token simulation
        for start_event in self.start_events:
            token_counts = self.simulate_token_flow(start_event)
            unsafe_flows = [flow for flow, count in token_counts.items() if count > 1]
            if unsafe_flows:
                results['safeness']['satisfied'] = False
                results['safeness']['details'].extend(
                    f"Flow {flow} can have multiple tokens" for flow in unsafe_flows
                )
                
        # Check proper completion
        if not proper_completion:
            results['proper_completion']['satisfied'] = False
            results['proper_completion']['details'].append(
                "Some end events are executed more than once in terminating paths."
            )

        return results



def mainSC(bpmn_json: Dict) -> Dict[str, Dict]:
    """Main function to check BPMN soundness and safeness."""
    checker = BPMNChecker(bpmn_json)
    results = checker.check_SC()
    total_checks = len(results)
    failed_checks = sum(1 for check in results.values() if not check['satisfied'])
    score = ((total_checks - failed_checks) / total_checks)*100
    return results, score


def compute_StructuralCorrectness(file_path):
    soundness_scores = []
    results = []
    for file in file_path:
        print("Processing file: ", file)
        try:
            json_str = read_file(file)  # raw string
            json_trial = ast.literal_eval(json_str) 
            result, score = mainSC(json_trial)
            soundness_scores.append(score)
            results.append(result)
        except Exception as e:
            print("Error in file: ", file, "skipping...")
            print(e)
            soundness_scores.append(0)
            
    mean_score = np.mean(soundness_scores) if soundness_scores else 0
    sd_score = np.std(soundness_scores) if soundness_scores else 0
    
    # restituisci mean, sd e result e print su compute_metrics
    return mean_score, sd_score, results
