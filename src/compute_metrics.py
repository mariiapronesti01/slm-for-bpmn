import argparse
from utils.utils import load_files_from_specific_folder, build_process_info_dict

import os

from utils.structural_similarity import compute_StructuralSimilarity
from utils.structural_correctness import compute_StructuralCorrectness

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def calculate_compute_metrics(ground_truth_folder_path, generated_folder_path):
    # Load the files from the folders
    print("Loading groudtruth files from the folders")
    ground_truth_files = load_files_from_specific_folder(ground_truth_folder_path, ".bpmn")
    print("GROUND FILES: ",len(ground_truth_files))

    print("Loading generated files from the folders")
    generated_files = load_files_from_specific_folder(generated_folder_path, ".json")
    print("GENERATED FILES: ",len(generated_files))

    # Build the process info dict for the ground truth files
    print("Building the process info. It may take a while...")
    original_process_info_dict = build_process_info_dict(ground_truth_files, model, type=None)
    # Build the process info dict for the generated files
    generated_process_info_dict = build_process_info_dict(generated_files, model, type="JSON")
    print("Generated process info dict: ", generated_process_info_dict)
    # Compute the metrics
    print("Structural Similarity")
    structural_similarity_df = compute_StructuralSimilarity(generated_process_info_dict, original_process_info_dict)
    print("Average Structural similarity:", structural_similarity_df["Max_Struct"].mean())
    print("Std of Structural similarity: ", structural_similarity_df["Max_Struct"].std())
    print("\nStructural Correctness")
    mean_score, sd_score, _ = compute_StructuralCorrectness(generated_files)
    print("Average Structural Correctness: ", mean_score)
    print("Std of Structural Correctness: ", sd_score)
    return structural_similarity_df["Max_Struct"].mean(), structural_similarity_df["Max_Struct"].std(), mean_score, sd_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for code files")
    parser.add_argument("--ground_truth_folder_path", type=str, required=True, help="Path to the folder containing the groundtruth files")
    
    parser.add_argument("--generated_folder_path", type=str, required=True, help="Path to the folder containing the generated files in JSON format")

    args = parser.parse_args()
    
    ground_truth_folder_path = args.ground_truth_folder_path
    generated_folder_path = args.generated_folder_path
    
    print("groundtruth folder: ", ground_truth_folder_path)
    print("generated folder: ", generated_folder_path)

    # Check if the paths exists
    if not os.path.exists(ground_truth_folder_path):
        raise ValueError(f"The path {ground_truth_folder_path} does not exist")
    if not os.path.exists(generated_folder_path):
        raise ValueError(f"The path {generated_folder_path} does not exist")
    
    calculate_compute_metrics(ground_truth_folder_path, generated_folder_path)






