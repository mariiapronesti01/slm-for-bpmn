import xml.etree.ElementTree as ET
import pandas as pd
import re

def flatten(xss):
    return [x for xs in xss for x in xs]

def extract_process_info(processes):
    node_type = []
    attributes = []

    for child in processes:
            # Append the node type after removing any namespace from the tag
            node_type.append(re.sub(r'{.*}', '', child.tag))
            
            # Make a copy of child.attrib so we don't modify the original
            child_attrib = child.attrib.copy()
            
            # If 'name' attribute is missing or None, set it to 'None'
            if child_attrib.get('name') is None or child_attrib.get('name') == "":
                child_attrib['name'] = 'Unnamed'
            else:
                child_attrib['name'] = re.sub(r'[^A-Za-z0-9\s><]', '', child_attrib['name']).replace('\n', ' ')
            
            # Append the modified attributes dictionary to the list
            attributes.append(child_attrib)
    return node_type, attributes

def extract_lane(process, ns):
    lane_info = {}
    process_name = re.sub(r'[^A-Za-z0-9\s]', '', process.attrib.get('name', 'Unnamed Process')).replace('\n', ' ') # Get process name or provide a default

    for lane in process.findall(f'{ns}laneSet/{ns}lane'):
        lane_id = lane.attrib['id']
        lane_name = re.sub(r'[^A-Za-z0-9\s]', '', lane.attrib.get('name', process_name)).replace('\n', ' ')  # Use process_name if lane name is missing
        lane_nodes = [flow_node_ref.text for flow_node_ref in lane.findall(f'{ns}flowNodeRef')]
        
        # Store each lane's information in a dictionary, keyed by lane_id
        lane_info[lane_id] = {
            'name': lane_name,
            'nodes': lane_nodes
        }
    
    return lane_info


def parse_bpmn(file: str, verbose: bool = False, ns='{http://www.omg.org/spec/BPMN/20100524/MODEL}'):

    """
    Function to parse the BPMN file and extract the process(es) elements

    Parameters:
    - ns: namespace of the BPMN file
    - file: path to the BPMN file

    Returns:
    - node_type: list of node types
    - attributes: list of dictionaries containing the attributes of the process elements
    - message_flow: DataFrame containing the message flows
    - lane_info: dictionary containing the lane information
    - participants: dictionary containing the participants information

    """

    # Parse the XML file
    tree = ET.parse(file)
    root = tree.getroot()

    # Extract the process elements
    processes = root.findall(f'{ns}process')

    all_node_type = []
    all_attributes = []
    lane_info = {}
            
    for process in processes:
                node_type, attributes = extract_process_info(process)
                all_node_type.append(node_type)
                all_attributes.append(attributes)
                lane_info.update(extract_lane(process, ns))

    node_type = flatten(all_node_type)
    attributes = flatten(all_attributes)
    if verbose:
        print(f'Detected {len(processes)} processes in the BPMN file')

    return node_type, attributes, lane_info


def get_edge_df(attributes: list, node_type: list):

    """
    Function to create a DataFrame from the attributes of the process elements

    Parameters:
    - attributes: list of dictionaries containing the attributes of the process elements
    - node_type: list of node types
    - complete: boolean to return the complete DataFrame or only the sequenceFlow elements

    Returns:
    - df: DataFrame containing the attributes of the process elements

    """
    
    df = pd.DataFrame(attributes)
    df['node_type'] = node_type

    # map id to node type and name
    id_to_node_type = df.set_index('id')['node_type'].to_dict()
    id_to_name = df.set_index('id')['name'].to_dict()

    # add source type and target type columns
    df['sourceType']=df['sourceRef'].map(id_to_node_type)
    df['targetType']=df['targetRef'].map(id_to_node_type)

    # add source name and target name columns
    df['sourceName']=df['sourceRef'].map(id_to_name)
    df['targetName']=df['targetRef'].map(id_to_name)

    return df[(df['node_type'] == 'sequenceFlow')]


def getProcessInfo_fromXML(file:str, ns='{http://www.omg.org/spec/BPMN/20100524/MODEL}'):

    """
    Function to extract the edges from the BPMN file and return a DataFrame

    Parameters:
    - ns: namespace of the BPMN file
    - file: path to the BPMN file
    - brackets: boolean to add brackets to the source and target names

    Returns:
    - df: DataFrame containing the edges of the process elements

    """
    node_type, attributes, lane_info = parse_bpmn(file, ns=ns)
    
    df = get_edge_df(attributes, node_type)
    df = df.drop_duplicates(subset=['sourceRef', 'targetRef'], keep='first')
    return df, lane_info