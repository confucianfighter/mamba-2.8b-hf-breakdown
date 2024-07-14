import os
import pyperclip

def number_lines(text):
    lines = text.split('\n')
    numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)

def concatenate_files(folder_path, file_types, instructions_file='chatgpt_instructions.txt', recursive=True):
    output_text = []
    separator = "\n" + "-" * 50 + "\n"

    # Add instructions file content first
    instructions_path = os.path.join(folder_path, instructions_file)
    if os.path.exists(instructions_path):
        with open(instructions_path, 'r', encoding='utf-8') as f:
            instructions_content = f.read()
            numbered_instructions = number_lines(instructions_content)
            output_text.append(separator)
            output_text.append(f"{instructions_path}\n{separator}")
            output_text.append(numbered_instructions)

    # Traverse the folder recursively or non-recursively
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            include_path = input(f"Do you want to include the path: {root}? (yes/no): ").strip().lower() == 'yes'
            if not include_path:
                continue

            # Exclude __pycache__ and other library directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'env', 'venv', 'node_modules']]

            for file in files:
                if any(file.endswith(file_type) for file_type in file_types):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        numbered_file_content = number_lines(file_content)
                        output_text.append(separator)
                        output_text.append(f"{file_path}\n{separator}")
                        output_text.append(numbered_file_content)
    else:
        for file in os.listdir(folder_path):
            if any(file.endswith(file_type) for file_type in file_types):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    numbered_file_content = number_lines(file_content)
                    output_text.append(separator)
                    output_text.append(f"{file_path}\n{separator}")
                    output_text.append(numbered_file_content)

    return "".join(output_text)

if __name__ == "__main__":
    # Prompt user for recursion
    recursive = input("Do you want to search directories recursively? (yes/no): ").strip().lower() == 'yes'

    # Prompt user for directory
    include_directory = input("Do you want to specify a directory? (yes/no): ").strip().lower() == 'yes'
    if include_directory:
        folder_path = input("Please enter the directory path: ").strip()
        if not os.path.isdir(folder_path):
            print("Invalid directory path. Exiting.")
            exit(1)
    else:
        folder_path = os.getcwd()

    # Prompt user for file types to include
    file_types = []
    while True:
        file_type = input("Enter a file type to include (e.g., .py) or 'done' to finish: ").strip()
        if file_type.lower() == 'done':
            break
        file_types.append(file_type)

    concatenated_text = concatenate_files(folder_path, file_types, recursive=recursive)
    output_file = "concatenated_output.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        # overwrite with the concatenated text
        f.write(concatenated_text)
        

    # Copy the concatenated text to the clipboard
    pyperclip.copy(concatenated_text)

    print(f"All specified files and instructions have been concatenated into {output_file}")
    print("The concatenated output has been copied to the clipboard.")
