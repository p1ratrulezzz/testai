import torch
from transformers import pipeline
from dotenv import load_dotenv
import os
import glob

# --- 1. Initialization ---
# Load environment variables from .env file
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the text generation pipeline
print("Loading model... This may take some time.")
pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    token=token,
)
print("Model loaded successfully!")

# --- 2. Settings ---
# Generation arguments
generation_args = {
    "max_new_tokens": 2048,
    "return_full_text": False,
    "temperature": 0.4,
    "do_sample": True,
}

# --- 3. Batch Code Review with Chunking ---
CODE_DIR = "code"
OUTPUT_FILE = "result.txt"
CHUNK_SIZE = 150  # Number of lines to process per chunk

# Find all files in the specified directory recursively
all_files = glob.glob(os.path.join(CODE_DIR, '**/*'), recursive=True)

# Filter out directories, keeping only files
files_to_review = [f for f in all_files if os.path.isfile(f)]

if not files_to_review:
    print(f"No files found in the '{CODE_DIR}' directory. Exiting.")
    exit()

print(f"Found {len(files_to_review)} files to review.")

# Open the output file in write mode to clear previous results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("--- Code Review Results ---\n\n")

# Process each file
for file_path in files_to_review:
    print(f"Reviewing {file_path}...")
    try:
        # Attempt to read the file as text, skip if it's likely binary
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Process the file in chunks
        for i in range(0, len(lines), CHUNK_SIZE):
            chunk_lines = lines[i:i + CHUNK_SIZE]
            start_line = i + 1
            end_line = i + len(chunk_lines)

            print(f"  - Reviewing lines {start_line}-{end_line}...")

            # Prepend line numbers to each line in the chunk for context
            chunk_content = ""
            for line_num, line in enumerate(chunk_lines, start=start_line):
                chunk_content += f"{line_num}: {line}"

            # Create the specific prompt for the model
            messages = [
                {
                    "role": "user",
                    "content": (
                        "You are a meticulous code reviewer. Your task is to review the following code snippet "
                        f"from the file '{file_path}'. The code is prefixed with its original line numbers. "
                        "Provide your feedback ONLY in the specific format: 'Line [line_number]: [comment]'. "
                        "Only provide feedback on the code with serious issues or where fix is needed."
                        "Each comment must be on a new line. Do not add any introduction or summary.\n\n"
                        f"```\n{chunk_content}\n```"
                    )
                }
            ]

            # Get the review from the model
            output = pipe(messages, **generation_args)
            review_text = output[0]['generated_text'].strip()

            # Append the result to the output file
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"--- Review for: {file_path} (Lines {start_line}-{end_line}) ---\n")
                f.write(review_text)
                f.write("\n\n" + "="*50 + "\n\n")

        print(f"Finished review for {file_path}. Result appended to {OUTPUT_FILE}.")

    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Review for: {file_path} ---\n")
            f.write("Skipped: This appears to be a binary file and cannot be reviewed as text.\n")
            f.write("\n\n" + "="*50 + "\n\n")
    except Exception as e:
        print(f"Could not process file {file_path}. Error: {e}")
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Review for: {file_path} ---\n")
            f.write(f"Failed to review this file due to an error: {e}\n")
            f.write("\n\n" + "="*50 + "\n\n")

print("\nBatch code review complete.")
print(f"All results have been saved to {OUTPUT_FILE}.")
