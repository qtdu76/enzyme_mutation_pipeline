def indices_to_pymol(selection_string):
    # Convert to sorted list of unique integers
    indices = sorted(set(int(i.strip()) for i in selection_string.split(",") if i.strip()))
    
    # Group into ranges
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))

    # Format PyMOL-style selection
    formatted = []
    for start, end in ranges:
        if start == end:
            formatted.append(f"{start}")
        else:
            formatted.append(f"{start}-{end}")
    return "+".join(formatted)


# Example usage
raw_indices = "86, 87, 159, 160, 161, 162, 185, 205, 206, 207, 208, 236, 237"
pymol_selection = indices_to_pymol(raw_indices)
print(f"resi {pymol_selection}")
