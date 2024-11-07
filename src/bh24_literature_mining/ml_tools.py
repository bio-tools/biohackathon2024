import os
import csv
import pathlib
from tqdm import tqdm
import nltk
from nltk.tokenize import wordpunct_tokenize

# Ensure NLTK is installed and the tokenizer is available
nltk.download('punkt')

def find_sub_span(token_span, entity_span):
    if token_span[0] < entity_span[1] and token_span[1] > entity_span[0]:
        return max(token_span[0], entity_span[0]), min(token_span[1], entity_span[1])
    return None

def convert_to_iob(texts, ner_tags_list):
    results = []

    for text, ner_tags in zip(texts, ner_tags_list):
        # Tokenize using NLTK's wordpunct_tokenizer
        tokens = wordpunct_tokenize(text)
        token_spans = []
        current_idx = 0

        # Calculate token spans based on the original text
        for token in tokens:
            start_idx = text.find(token, current_idx)
            end_idx = start_idx + len(token)
            token_spans.append((start_idx, end_idx))
            current_idx = end_idx

        iob_tags = ['O'] * len(tokens)

        for start, end, entity, entity_type in sorted(ner_tags, key=lambda x: x[0]):
            entity_flag = False  # Flag to indicate if we are inside an entity
            for i, token_span in enumerate(token_spans):
                if find_sub_span(token_span, (start, end)):
                    if not entity_flag:  # If it's the start of an entity
                        iob_tags[i] = 'B-' + entity_type
                        entity_flag = True
                    elif iob_tags[i] == 'O':  # Continue tagging inside of the entity
                        iob_tags[i] = 'I-' + entity_type
                else:
                    entity_flag = False  # Reset flag when we're no longer in an entity

        results.append(list(zip(tokens, iob_tags)))
    return results

def convert_to_IOB_format_from_df(dataframe, output_folder, filename, batch_size=500):
    # Prepare data for batch processing
    data = [(row['sentence'], row['ner_ines']) for index, row in dataframe.iterrows()]

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    result_path = os.path.join(output_folder, filename)

    with open(result_path, 'w', newline='\n') as f1:
        train_writer = csv.writer(f1, delimiter='\t', lineterminator='\n')

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i:i+batch_size]
            sentences, ner_tags_batch = zip(*batch)

            # Convert to IOB format
            batch_results = convert_to_iob(sentences, ner_tags_batch)

            for tagged_tokens in batch_results:
                for each_token in tagged_tokens:
                    train_writer.writerow(list(each_token))
                train_writer.writerow('')


def load_iob_file(file_path):
    """
    Function to load an IOB file into a pandas DataFrame.

    Parameters:
    file_path (str): Path to the IOB file.

    Returns:
    pd.DataFrame: DataFrame containing tokens and their IOB tags.
    """
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Only process non-empty lines
                parts = line.split("\t")
                if len(parts) == 2:
                    token, tag = parts
                    data.append((token, tag))
                # else:
                # print(f"Warning: Skipping invalid line {line_num} in file {file_path}: '{line}'")
            else:
                data.append(("", ""))  # Add empty line as a sentence boundary
    return pd.DataFrame(data, columns=["token", "tag"])


def check_data_integrity(df):
    """
    Function to check the integrity of NER data.
    Ensures:
    1. Each 'B-ent' is followed by its respective 'I-ent' (if applicable).
    2. No 'I-ent' appears without a preceding 'B-ent'.

    Parameters:
    df (pd.DataFrame): DataFrame containing the NER columns with tokens and entities.

    Returns:
    bool: True if the data integrity is intact, False if there are issues.
    issues (list): List of issues found in the dataset.
    """
    issues = []
    found_b_ent = False

    for i, entity in enumerate(df['tag']):
        if entity.startswith('B-'):
            found_b_ent = True
        elif entity.startswith('I-'):
            if not found_b_ent:
                issues.append(f"Row: {i}, Hanging I-ent without preceding B-ent.")
            # Check if the current I-ent is the correct continuation of a B-ent
            elif not entity[2:] == df['tag'].iloc[i - 1][2:]:
                issues.append(f"Row: {i}, Mismatch between B-ent and I-ent.")
        # Reset B-ent flag if it's an 'O', empty string, or sentence boundary
        if entity == 'O' or entity == '' or (df['token'].iloc[i] == "" and entity == ""):
            found_b_ent = False

    return (True, []) if not issues else (False, issues)


def check_integrity_of_files(train_file_paths, dev_file_paths, test_file_paths):
    """
    Check the integrity of NER train, development, and test sets for each dataset.

    Parameters:
    train_file_paths (list): List of file paths for training datasets.
    dev_file_paths (list): List of file paths for development datasets.
    test_file_paths (list): List of file paths for test datasets.

    Returns:
    None: Prints the integrity check results for each set.
    """
    for i, (train_file, dev_file, test_file) in enumerate(zip(train_file_paths, dev_file_paths, test_file_paths)):
        print(f"\nChecking Dataset {i + 1}:")

        # Load train set and check integrity
        train_df = load_iob_file(train_file)
        is_train_valid, train_issues = check_data_integrity(train_df)
        if is_train_valid:
            print(f"Train set {i + 1} is valid.")
        else:
            print(f"Train set {i + 1} has issues:\n", "\n".join(train_issues))

        # Load dev set and check integrity
        dev_df = load_iob_file(dev_file)
        is_dev_valid, dev_issues = check_data_integrity(dev_df)
        if is_dev_valid:
            print(f"Dev set {i + 1} is valid.")
        else:
            print(f"Dev set {i + 1} has issues:\n", "\n".join(dev_issues))

        # Load test set and check integrity
        test_df = load_iob_file(test_file)
        is_test_valid, test_issues = check_data_integrity(test_df)
        if is_test_valid:
            print(f"Test set {i + 1} is valid.")
        else:
            print(f"Test set {i + 1} has issues:\n", "\n".join(test_issues))


