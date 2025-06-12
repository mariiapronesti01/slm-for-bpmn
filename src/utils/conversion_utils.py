from utils.utils import getProcessInfo_fromXML, getProcessInfo_fromJSON, getProcessInfo_fromMER
from typing import List, Optional
from pydantic import BaseModel, Field
import re
import random
import os

class Node(BaseModel):
    """Class representing individual nodes extracted from a textual description of a BPMN diagram"""
    id: Optional[str] = Field(description="A unique identifier for the nodes")                       # if want to allow Optional, use specify None in the field
    name: Optional[str] = Field(description="The name of the node (3-5 words)")
    BPMNtype: Optional[str] = Field(description="A valid node type according to the BPMN standard")  # Explicitly Optional
    incoming: Optional[List[str]] = Field(default_factory=list, description="A list of ids of incoming nodes")
    outgoing: Optional[List[str]] = Field(default_factory=list, description="A list of ids of outgoing nodes")

class Lane(BaseModel):
    """Class representing a BPMN lane that contains a set of nodes."""
    id: Optional[str] = Field(None, description="A unique identifier for the lane")
    name: Optional[str] = Field(None, description="The name of the lane")
    nodes: List[str] = Field(default_factory=list, description="A list of nodes belonging to this lane")

class BPMN(BaseModel):
    """Class representing a collection of nodes extracted from the textual description of a BPMN diagram"""
    start_nodes: List[Node] = Field(description="A list of start nodes of the BPMN diagram")
    nodes: List[Node] = Field(description="A list of nodes extracted from the textual description of a BPMN diagram")
    end_nodes: List[Node] = Field(description="A list of the end nodes of the BPMN diagram")
    lanes: List[Lane] = Field(default_factory=list, description="A list of lanes in the BPMN diagram")


def convert_toJSON(df, lane_info):
    nodes = {}
    start_nodes = []
    end_nodes = []
    lanes = []
    id_mapping = {} 
    lane_id_mapping = {}
    used_numbers = set() 
    
    for _, row in df.iterrows():
        source_id, source_name, source_type = row['sourceRef'], row['sourceName'], row['sourceType']
        target_id, target_name, target_type = row['targetRef'], row['targetName'], row['targetType']
        
        # Generate unique IDs for source and target nodes
        for node_id, node_type in [(source_id, source_type), (target_id, target_type)]:
            if node_id not in id_mapping:
                while True:
                    random_number = random.randint(100, 999)
                    if random_number not in used_numbers:
                        used_numbers.add(random_number)
                        break
                new_id = f"{node_type}_{random_number}"
                id_mapping[node_id] = new_id
                
        # Initialize source node if not already in nodes
        if source_id not in nodes:
            nodes[source_id] = {
                'id': id_mapping[source_id],
                'name': source_name,
                'BPMNtype': source_type,
                'incoming': [],
                'outgoing': []
            }
            if source_type == 'startEvent':
                start_nodes.append(nodes[source_id])
            elif source_type == 'endEvent':
                end_nodes.append(nodes[source_id])
        
        # Initialize target node if not already in nodes
        if target_id not in nodes:
            nodes[target_id] = {
                'id': id_mapping[target_id],
                'name': target_name,
                'BPMNtype': target_type,
                'incoming': [],
                'outgoing': []
            }
            if target_type == 'startEvent':
                start_nodes.append(nodes[target_id])
            elif target_type == 'endEvent':
                end_nodes.append(nodes[target_id])
        
        # Update connections
        nodes[source_id]['outgoing'].append(id_mapping[target_id])
        nodes[target_id]['incoming'].append(id_mapping[source_id])
    
    # If lanes are provided, structure them
    if lane_info:
        for lane_id, lane_data in lane_info.items():
            if lane_id not in lane_id_mapping:
                while True:
                    random_number = random.randint(100, 999)
                    if random_number not in used_numbers:
                        used_numbers.add(random_number)
                        break
                new_lane_id = f"lane_{random_number}"
                lane_id_mapping[lane_id] = new_lane_id

            lane_nodes = [id_mapping[node_id] for node_id in lane_data['nodes'] if node_id in id_mapping]
            lanes.append({
                'id': lane_id_mapping[lane_id],
                'name': lane_data['name'],
                'nodes': lane_nodes
            })
    else:
        lanes = []    
    # Extract non-start/non-end nodes
    start_and_end_ids = {node['id'] for node in start_nodes + end_nodes}
    nodes_list = [node for node_id, node in nodes.items() if node['id'] not in start_and_end_ids]
    
    return {
        'start_nodes': start_nodes,
        'nodes': nodes_list,
        'end_nodes': end_nodes,
        'lanes': lanes
    }
    

# change df in edge_df, chiama la funzione toJSON
def toJSON(file, output_folder, input_type=None):
    try: 
        if input_type == "MER":
            edge_df, lane_info = getProcessInfo_fromMER(file)
        else: 
            edge_df, lane_info = getProcessInfo_fromXML(file)
        converted_df = convert_toJSON(edge_df, lane_info)
        # check if valid BPMN
        try:
            bpmn_instance = BPMN(**converted_df)
        except: 
            print("Validation failed for file: ", file)
        # write to file with the same name
        output = os.path.join(output_folder, re.sub(r'\.(bpmn|mer)$', '.json', os.path.basename(file)))
        
        # change in output_filename
        
        with open(output, 'w') as f:
            f.write(str(converted_df))
        print(f"Converted {file} to JSON format.")
    except:
        print("Error in file: ", file)


def toMER(file: str, output_folder: str, input_type=None):
    """
    Function that writes a pseudo MERMAID file for a given BPMN file.
    """
    if input_type == "JSON":
        df, lane_info = getProcessInfo_fromJSON(file)
    else:
        df, lane_info = getProcessInfo_fromXML(file)
    
    df.rename(columns={'id': 'sequenceRef', 'node_type': 'sequenceType'}, inplace=True)
    
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the output file path within the specified folder
    output = os.path.join(output_folder, os.path.basename(file).replace('.bpmn', '.mer'))
    
    id_mapping = {}  # mapping from original IDs to new IDs
    used_numbers = set()  # set to store used numbers for new IDs
    
    with open(output, 'w') as f:  # Open the file for writing once
        # First, create mappings for sourceRef and targetRef
        for _, row in df.iterrows():
            for ref_type in ['sourceRef', 'targetRef', 'sequenceRef']:
                ref = row[ref_type]
                if ref not in id_mapping:
                    # Generate a unique 6-digit numeric ID
                    while True:
                        random_number = random.randint(100, 999)
                        if random_number not in used_numbers:
                            used_numbers.add(random_number)
                            break
                    new_id = f"{row[ref_type.replace('Ref', 'Type')]}_{random_number}"
                    id_mapping[ref] = new_id
        
        # Write connections between nodes using new IDs
        for _, row in df.iterrows():
            source_new_id = id_mapping[row['sourceRef']]
            target_new_id = id_mapping[row['targetRef']]
                    
            f.write(
                f"{source_new_id}({row['sourceName']})"
                f"-->"                                                  #f"-->|{sequence_new_id}({row['name']})|"
                f"{target_new_id}({row['targetName']})\n"
            )
        
        if lane_info:
            # Write subgraphs for each lane
            for _, lane_data in lane_info.items():
                # Write the subgraph label for the current lane
                random_number = random.randint(100, 999)
                f.write(f"lane_{random_number}({lane_data['name']})\n")
    
                # Write all nodes associated with the current lane using new IDs
                current_lane_nodes = set(lane_data['nodes'])
                for node in current_lane_nodes:
                    new_node_id = id_mapping.get(node, node)  # Use new ID if available
                    f.write(f"  {new_node_id}\n")
    
                # Close the subgraph for the current lane
                f.write("end\n\n")