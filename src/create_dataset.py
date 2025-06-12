from utils.utils import load_files_from_specific_folder, get_filename_without_extension, read_file
from utils.data_filtering import filter_matching_files, extract_bpmn_names
import json
import random
import os
import ast
import argparse
import numpy as np

def create_dataset(textual_description, bpmn, instruction, format_file, output_dir):
    # Initialize data list
    data = []

    for i in range(len(textual_description)):
        try: 
            # Check if filenames (without extensions) match
            if get_filename_without_extension(textual_description[i]) == get_filename_without_extension(bpmn[i]):        
                #print(f"Processing {textual_description[i]} and {bpmn[i]}")
                # Read the content of the textual description and MER files
                with open(textual_description[i], "r") as text_file:
                    textual_description_content = text_file.read()
                with open(bpmn[i], "r") as bpmn_file:
                    bpmn_content = bpmn_file.read()

            # Append the structured dictionary to the data list
            data.append({
                    "instruction": instruction,
                    "input": f"Please generate the {format_file} file corresponding to the following textual description: {textual_description_content}",
                    "output": bpmn_content

                })
            
        except Exception as e:
            #print(f"Error processing files {textual_description[i]} and {bpmn[i]}: {e}")
            continue

    # Shuffle the data
    random.shuffle(data)

    # Split data
    total = len(data)
    train_split = int(total * 0.90)
    val_split = int(total * 0.10)

    train_data = data[:train_split]
    val_data = data[train_split:train_split + val_split]

    # Write the split data to JSON files
    train_dir = os.path.join(output_dir, format_file, f"train{format_file}.json")
    with open(train_dir, "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    val_dir = os.path.join(output_dir, format_file, f"val{format_file}.json")
    with open(val_dir, "w") as val_file:
        json.dump(val_data, val_file, indent=4)
        

def create_GRPOdataset(textual_description, bpmn_file, instruction, output_dir):
    # Initialize data list
    data = []

    for i in range(len(textual_description)):
        try:
            # Check if filenames (without extensions) match
            if get_filename_without_extension(textual_description[i]) == get_filename_without_extension(bpmn_file[i]):        
                #print(f"Processing {textual_description[i]} and {bpmn_file[i]}")
                # Read the content of the textual description and MER files
                with open(textual_description[i], "r") as text_file:
                    textual_description_content = text_file.read()
                    bpmn_content = ast.literal_eval(read_file(bpmn_file[i]))
                    lane_names, task_names, task_types, gateways = extract_bpmn_names(bpmn_content)

            # Append the structured dictionary to the data list
            if len(gateways) > 0:
                data.append({
                    "instruction": instruction,
                    "input": f"Here's the textual description you have to analyze: {textual_description_content}",
                    "true_task" : task_names,
                    "true_actors" : lane_names,
                })
        except Exception as e:
            #print(f"Error processing {textual_description[i]} and {bpmn_file[i]}: {e}")
            continue

    # Shuffle the data
    random.shuffle(data)

    # Split data into train (80%), val (10%), and test (10%)
    total = len(data)
    train_split = int(total * 1)

    train_data = data[:train_split]

    # Write the split data to JSON files
    output_dir = os.path.join(output_dir, "GRPO", f"trainGRPO.json")
    with open(output_dir, "w") as train_file:
        json.dump(train_data, train_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build datasets")
    parser.add_argument("--textual_description_path", type=str, required=True, help="Path to the folder containing the textual descriptions")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the folder containing the files in JSON format")
    parser.add_argument("--mer_path", type=str, required=True, help="Path to the folder containing the files in MER format")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    

    args = parser.parse_args()
    
    # Ensure the input directories exist
    if not os.path.exists(args.textual_description_path):
        raise FileNotFoundError(f"Textual description path {args.textual_description_path} does not exist.")
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON path {args.json_path} does not exist.")
    if not os.path.exists(args.mer_path):
        raise FileNotFoundError(f"MER path {args.mer_path} does not exist.")
    
    
    # Ensure the output directory exists (if not create it) 
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "MER"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "JSON"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "GRPO"), exist_ok=True)
    
    # Load the textual descriptions and BPMN files
    textual_description_path = load_files_from_specific_folder(args.textual_description_path, ".txt")
    mer_file_path = load_files_from_specific_folder(args.mer_path, ".mer")
    json_files_path = load_files_from_specific_folder(args.json_path, ".json")
    
    mer_instruction= read_file("..//prompts//MER//SFT_prompt.txt")
    json_instruction= read_file("..//prompts//JSON//SFT_prompt.txt")
    grpo_instruction = read_file("..//prompts//JSON//GRPO_prompt.txt")
    
    # Filter the files to ensure they match
    mer_descriptions, mer_files = filter_matching_files(textual_description_path, mer_file_path)
    print(f"Number of MER files: {len(mer_files)}")
    print(f"Number of corresponding textual descriptions: {len(mer_descriptions)}")

    json_descriptions, json_files = filter_matching_files(textual_description_path, json_files_path)
    print(f"Number of JSON files: {len(json_files)}")
    print(f"Number of corresponding textual descriptions: {len(json_descriptions)}")
    
    # Create MER dataset
    create_dataset(mer_descriptions, mer_files, mer_instruction, "MER", args.output_dir) 
    # Create JSON dataset
    create_dataset(json_descriptions, json_files, json_instruction, "JSON", args.output_dir)
    # Create GRPO dataset
    create_GRPOdataset(json_descriptions, json_files, grpo_instruction, args.output_dir)
    
    