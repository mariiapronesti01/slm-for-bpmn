from utils.BPMN_parser import parse_bpmn
from utils.utils import get_filename_without_extension
import pandas as pd
import re


def extract_info(file_paths):
    """
    Extracts information from the BPMN files. For each file, the function counts the number of 
    - elements in the bpmn
    - tasks
    - catch events 
    - edges
    - exclusive gateways
    - parallel gateways 
    - start events
    - end events
    """
    # Create an empty list to store the results
    results = []
    lanes = []
    
    for file_path in file_paths:
        try: 
            # Parse the BPMN file
            nodes, _, lane_info = parse_bpmn(file_path)
            lanes.append(lane_info)
            
            # Count total number of nodes for each example
            not_allowed_nodes = ["laneSet", "dataObject","dataObjectReference", "dataStore", "dataStoreReference", "association", "textAnnotation", "sequenceFlow", "extensionElements" ]
            allowed_nodes = [node for node in nodes if not any([word in node for word in not_allowed_nodes])]
            total_elements = len(allowed_nodes)
            
            # Count the number of tasks and catch events
            general_task_count = sum(bool(re.search('task', word)) for word in nodes)
            specific_task_count = sum(bool(re.search(r'(serviceTask|userTask|sendTask|receiveTask|scriptTask|manualTask|businessRuleTask)', word)) for word in nodes)
            
            catch_event_count = sum(bool(re.search('catch', word, re.IGNORECASE)) for word in nodes)
            
            # Count also number od edges
            edges = sum(bool(re.search('sequenceFlow', word)) for word in nodes)
            
            # exclusive gateways
            exclusive_gateways = sum(bool(re.search('exclusiveGateway', word)) for word in nodes)
            parallel_gateways = sum(bool(re.search('parallelGateway', word)) for word in nodes)
            
            # count number of start/end events
            start_events = sum(bool(re.search('startEvent', word)) for word in nodes)
            end_events = sum(bool(re.search('endEvent', word)) for word in nodes)
        except:
            continue
        
        # Append the results to the list
        results.append({'file_path': file_path, 
                        'total_elements': total_elements,
                        'general_task_count': general_task_count, 
                        'specific_task_count': specific_task_count,
                        'catch_event_count': catch_event_count, 
                        'edges_count': edges, 
                        'exclusive_gateways_count': exclusive_gateways, 
                        'parallel_gateways_count': parallel_gateways,
                        'start_events': start_events, 
                        'end_events': end_events})
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)
    
    return df, lanes


## NEEDED to extract statiscs from MaD dataset
def extract_info_from_edges(file_paths):
    """
    Extracts information from the BPMN represented as edges from a list of file paths. For each file, the function counts the number of
    - elements in the BPMN
    - tasks
    - catch events 
    - edges
    - exclusive gateways
    - parallel gateways 
    - start events
    - end events
    """
    # Create an empty list to store the results
    results = []
    lanes = []  # Assuming lane information is not present in this format

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                edge_lines = file.readlines()

            # Initialize counters
            total_elements = set()
            general_task_count = 0
            specific_task_count = 0
            catch_event_count = 0
            edges = 0
            exclusive_gateways = 0
            parallel_gateways = 0
            start_events = 0
            end_events = 0

            for line in edge_lines:
                edges += 1

                # Extract source and target elements
                source, target = line.split('-->')
                source = source.strip()
                target = target.strip()

                # Add to total elements (using a set to avoid duplicates)
                total_elements.update([source, target])

                # Count specific elements
                if re.search(r'startEvent', source, re.IGNORECASE):
                    start_events += 1
                if re.search(r'endEvent', target, re.IGNORECASE):
                    end_events += 1

                if re.search(r'parallelGateway', source, re.IGNORECASE) or re.search(r'parallelGateway', target, re.IGNORECASE):
                    parallel_gateways += 1
                if re.search(r'exclusiveGateway', source, re.IGNORECASE) or re.search(r'exclusiveGateway', target, re.IGNORECASE):
                    exclusive_gateways += 1

                if re.search(r'task', source, re.IGNORECASE) or re.search(r'task', target, re.IGNORECASE):
                    general_task_count += 1
                    if re.search(r'(serviceTask|userTask|sendTask|receiveTask|scriptTask|manualTask|businessRuleTask)', source, re.IGNORECASE) or re.search(r'(serviceTask|userTask|sendTask|receiveTask|scriptTask|manualTask|businessRuleTask)', target, re.IGNORECASE):
                        specific_task_count += 1

            # Append the results to the list
            results.append({
                'file_path': file_path,
                'total_elements': len(total_elements),
                'general_task_count': general_task_count,
                'specific_task_count': specific_task_count,
                'catch_event_count': catch_event_count,
                'edges_count': edges,
                'exclusive_gateways_count': exclusive_gateways,
                'parallel_gateways_count': parallel_gateways,
                'start_events': start_events,
                'end_events': end_events
            })

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)

    return df, lanes


def print_statistics(df, lane_info):
    """
    Returns mean and sd for each column in the DataFrame.
    """
    print("Average number of Nodes: ", round(df['total_elements'].mean(), 2), "   sd: ", round(df['total_elements'].std(), 2))
    print("Average number of General tasks: ", round(df['general_task_count'].mean(), 2), "   sd: ", round(df['general_task_count'].std(), 2))
    print("Average number of Specific tasks: ", round(df['specific_task_count'].mean(), 2), "   sd: ", round(df['specific_task_count'].std(), 2))
    print("Average number of catch events: ", round(df['catch_event_count'].mean(), 2), "   sd: ", round(df['catch_event_count'].std(), 2))
    print("Average number of edges: ", round(df['edges_count'].mean(), 2), "   sd: ", round(df['edges_count'].std(), 2))
    print("Average number of exclusive gateways: ", round(df['exclusive_gateways_count'].mean(), 2), "   sd: ", round(df['exclusive_gateways_count'].std(), 2))
    print("Average number of parallel gateways: ", round(df['parallel_gateways_count'].mean(), 2), "   sd: ", round(df['parallel_gateways_count'].std(), 2))
    print("Average number of start events: ", round(df['start_events'].mean(), 2), "   sd: ", round(df['start_events'].std(), 2))
    print("Average number of end events: ", round(df['end_events'].mean(), 2), "   sd: ", round(df['end_events'].std(), 2))
    print("--------------")
    
    print("Median number of Nodes: ", round(df['total_elements'].median(), 2), "  max: ", round(df['total_elements'].max(), 2), "  min: ", round(df['total_elements'].min(), 2))
    print("Median number of General tasks: ", round(df['general_task_count'].median(), 2), "  max: ", round(df['general_task_count'].max(), 2), "  min: ", round(df['general_task_count'].min(), 2))
    print("Median number of Specific tasks: ", round(df['specific_task_count'].median(), 2), "  max: ", round(df['specific_task_count'].max(), 2), "  min: ", round(df['specific_task_count'].min(), 2))
    print("Median number of catch events: ", round(df['catch_event_count'].median(), 2), "  max: ", round(df['catch_event_count'].max(), 2), "  min: ", round(df['catch_event_count'].min(), 2))
    print("Median number of edges: ", round(df['edges_count'].median(), 2), "  max: ", round(df['edges_count'].max(), 2), "  min: ", round(df['edges_count'].min(), 2))
    print("Median number of exclusive gateways: ", round(df['exclusive_gateways_count'].median(), 2), "  max: ", round(df['exclusive_gateways_count'].max(), 2), "  min: ", round(df['exclusive_gateways_count'].min(), 2))
    print("Median number of parallel gateways: ", round(df['parallel_gateways_count'].median(), 2), "  max: ", round(df['parallel_gateways_count'].max(), 2), "  min: ", round(df['parallel_gateways_count'].min(), 2))
    print("Median number of start events: ", round(df['start_events'].median(), 2), "  max: ", round(df['start_events'].max(), 2), "  min: ", round(df['start_events'].min(), 2))
    print("Median number of end events: ", round(df['end_events'].median(), 2), "  max: ", round(df['end_events'].max(), 2), "  min: ", round(df['end_events'].min(), 2))
    print("--------------")
    #extract non empty lanes
    non_empty_lanes = []
    if lane_info: 
        for lane in lane_info:
            if lane:
                non_empty_lanes.append(lane)

    #get percenatge of non empty lanes
    len(non_empty_lanes)
    print("Percentage of examples containing Lanes: ", round(len(non_empty_lanes)/len(lane_info)*100, 2), "%")


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

def filter_matching_files(textual_description_list, bpmn_list):
    """
    Filters files in `bpmn_list` where the filenames match those in `textual_description_list`.
    Returns two filtered lists: textual_description_list and bpmn_list.
    """
    # Extract filenames from textual_description_list
    textual_filenames = {get_filename_without_extension(path) for path in textual_description_list}

    # Filter bpmn_list keeping original paths
    new_bpmn_list = [path for path in bpmn_list if get_filename_without_extension(path) in textual_filenames]
    
    # Now extract filenames again from new_bpmn_list
    new_bpmn_file_names = {get_filename_without_extension(path) for path in new_bpmn_list}
    
    # Filter textual_description_list accordingly
    new_textual_description_list = [path for path in textual_description_list if get_filename_without_extension(path) in new_bpmn_file_names]

    return new_textual_description_list, new_bpmn_list

