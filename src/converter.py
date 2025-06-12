from utils.utils import load_files_from_specific_folder
from utils.conversion_utils import toMER, toJSON
import os
import argparse



# get --fromTO parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BPMN files to pseudo MERMAID format.")
    parser.add_argument("--input_folder", type=str, help="Input folder containing BPMN files.")
    parser.add_argument("--output_folder", type=str, help="Output folder for the pseudo MERMAID files.")
    parser.add_argument("--input_type", type=str, choices=["XML", "MER" ,"JSON"], default="XML", help="Type of input BPMN files (XML, MER, JSON).")
    parser.add_argument("--output_type", type=str, choices=["XML", "MER", "JSON"], default="JSON", help="Type of BPMN file (XML, MER, JSON).")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    input_type = args.input_type
    output_type = args.output_type
    
    print("Input folder: ", input_folder)
    print("Output folder: ", output_folder)
    
    # check if input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"The path {input_folder} does not exist")
    
    # check output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if input_type == "XML":
        input_path = load_files_from_specific_folder(input_folder, ".bpmn")
    elif input_type == "MER":
        input_path = load_files_from_specific_folder(input_folder, ".mer")
    else:
        input_path = load_files_from_specific_folder(input_folder, ".json")
        
    if output_type == "MER" and input_type == "XML":
        for filename in input_path:
                toMER(filename, output_folder)
                print(f"Converted {filename} to pseudo MERMAID format.")
    elif output_type == "MER" and input_type == "JSON":
        for filename in input_path:
                toMER(filename, output_folder, type="JSON")
                print(f"Converted {filename} to MER format.")
                
    elif output_type == "JSON" and input_type == "XML":
        for filename in input_path:
                toJSON(filename, output_folder)
    elif output_type == "JSON" and input_type == "MER":
        for filename in input_path:
                toJSON(filename, output_folder, input_type="MER")
    else:
        print("Invalid input type or output type. Please check the arguments.")