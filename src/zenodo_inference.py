from utils.conversion_utils import BPMN
from utils.utils import load_files_from_specific_folder, read_file

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import time
import os
import ast
from pydantic import ValidationError
import argparse


def generate_bpmn(instruction, description_path, output_dir, model, tokenizer):
    total_retries = 0
    processed_files = 0

    # We'll measure the total time to process each file
    file_times = []

    for file in description_path:
        description = read_file(file)

        # Start timing for this file
        start_file_time = time.time()

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f": Please generate the JSON file for the following textual description: + {description}."}
        ]

        retries = 0
        while retries < 3:
            print(f"Processing file: {file}")
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
                padding=True
            ).to("cuda")

            try:
                print("Generating BPMN...")
                outputs = model.generate(inputs, max_new_tokens=3048, num_return_sequences=1)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract assistant response
                output = text.split("assistant", 1)[-1].strip()

                # Parse as Python dictionary
                try:
                    parsed_output = ast.literal_eval(output)
                    if not isinstance(parsed_output, dict):
                        raise ValueError("Parsed output is not a dictionary")
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing output as dictionary: {e}")
                    retries += 1
                    continue  # Retry the generation

                print("Validating BPMN...")
                bpmn_instance = BPMN(**parsed_output)
                print("Validation successful!")
                break  # Exit loop if validation is successful

            except (ValidationError, KeyError, TypeError) as e:
                print(f"Error during BPMN validation: {e}")
                retries += 1

        # Keep track of how many total retries happened across all files
        total_retries += retries
        processed_files += 1

        # Save output to the specified folder
        output_file_path = os.path.join(
            output_dir,
            os.path.basename(file).replace(".txt", ".bpmn")
        )
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(str(output))  # Save as a valid Python dictionary format

        # End timing for this file
        end_file_time = time.time()
        # Store the total time for processing this single file
        file_times.append(end_file_time - start_file_time)

    # Calculate averages
    if processed_files > 0:
        average_retries = total_retries / processed_files
        average_file_time = sum(file_times) / processed_files
    else:
        average_retries = 0
        average_file_time = 0

    print(f"Average number of retries over all files: {average_retries:.2f}")
    print(f"Average processing time per file (seconds): {average_file_time:.4f}")


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Generate BPMN from textual descriptions using a pre-trained model.")
        parser.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model name or path")
        parser.add_argument("--instruction_path", type=str, default="./prompt/JSON/SFT_prompt.txt", help="Path to the system prompt file")
        parser.add_argument("--description_path", type=str, help="Path to the folder containing the description files")
        parser.add_argument("--output_dir", type=str, default="./output/JSON/SFT/", help="Directory to save the generated BPMN files")
        
        args = parser.parse_args()
        model_path = args.model_path
        instruction_path = args.instruction_path
        description_path = load_files_from_specific_folder(args.description_path)
        output_dir = args.output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        # Check if description path exists
        if not os.path.exists(args.description_path):
            raise FileNotFoundError(f"Description path {args.description_path} does not exist.")
        
        # Load the model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 6024,
            load_in_4bit = False,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1")

        FastLanguageModel.for_inference(model)
        
        # Read the instruction from the file
        instruction = read_file(instruction_path)
        
        generate_bpmn(instruction, description_path, output_dir, model, tokenizer)
