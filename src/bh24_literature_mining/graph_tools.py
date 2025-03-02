import json 

def filter_edges_names_ids(adjacency_matrix):
    """
    Filters the adjacency matrix by removing edges between nodes that share 
    the same prefix (tool name before '_') or suffix (publication number after '_').

    Parameters:
    - adjacency_matrix (pd.DataFrame): The adjacency matrix (square DataFrame).

    Returns:
    - pd.DataFrame: The filtered adjacency matrix with edges removed.
    """
    adjacency_matrix = adjacency_matrix.copy()  # Work on a copy

    # Extract both the name and numeric identifier
    def extract_parts(name):
        parts = name.rsplit("_", 1)  # Split at the last underscore
        name_part, num_id = parts  # Always assume there's a valid suffix
        return name_part.lower(), num_id  # Convert name to lowercase

    # Create a mapping of node names to their (lowercased) name and numeric identifier
    node_parts = {node: extract_parts(node) for node in adjacency_matrix.index}

    # Group nodes by prefix and suffix
    name_groups = {}
    id_groups = {}

    for node, (name_part, num_id) in node_parts.items():
        name_groups.setdefault(name_part, []).append(node)  # Group by prefix
        id_groups.setdefault(num_id, []).append(node)  # Group by suffix (ALWAYS valid)

    # Merge all groups
    merged_groups = list(name_groups.values()) + list(id_groups.values())

    # --- Remove Edges Between Nodes in the Same Group ---
    for group in merged_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                adjacency_matrix.loc[group[i], group[j]] = 0  # Remove edge
                adjacency_matrix.loc[group[j], group[i]] = 0  # Ensure symmetry

    return adjacency_matrix  # Return filtered matrix


def filter_adjacency_by_threshold(adjacency_matrix, threshold=20):
    """
    Filters an adjacency matrix by setting edges below a threshold to 0.

    Parameters:
    - adjacency_matrix (pd.DataFrame): The adjacency matrix to be filtered.
    - threshold (int or float): The minimum weight an edge must have to be kept.

    Returns:
    - pd.DataFrame: The filtered adjacency matrix with edges below the threshold removed.
    """
    return adjacency_matrix.where(adjacency_matrix >= threshold, 0)

