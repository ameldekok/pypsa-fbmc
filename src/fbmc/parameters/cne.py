

import pandas as pd


def determine_cnes(max_absolute_flow, line_capacity, line_usage_threshold) -> list:
    """
    Determine Critical Network Elements (CNEs) based on line usage.
    This function identifies the lines in the network that are considered 
    Critical Network Elements (CNEs) by comparing their maximum absolute power 
    flow to a predefined threshold. Lines with a maximum usage above the 
    threshold are classified as CNEs.
    Args:
        max_absolute_flow (pd.Series): The maximum absolute power flow for each line.
        line_capacity (pd.Series): The capacity of each line.
    Returns:
        list: A list of line indices that are considered Critical Network Elements (CNEs).
    """
    assert (0 < line_usage_threshold <= 1), 'Threshold is out of acceptable bounds: (0,1]'
    assert (max_absolute_flow.index == line_capacity.index).all(), 'Indices of max_absolute_flow and line_capacity do not match.'
    assert (max_absolute_flow > 0).all(), 'Max absolute flow contains non-positive values.'
    assert (line_capacity > 0).all(), 'Line capacity contains non-positive values.'

    # TODO: Use data from a whole year to determine the critical network elements.
    # TODO: Include non-lines in the CNEs. NOTE: Wouter said this is not necessary for now.

    # Calculate the mean line usage
    max_line_usage = max_absolute_flow / line_capacity

    # Get the lines that are above the threshold
    cne_lines = max_line_usage[max_line_usage > line_usage_threshold].index.tolist()

    assert len(cne_lines) != 0, f'There are no Critical Network Elements for threshold {line_usage_threshold}.'
    return cne_lines


def filter_on_cne(ptdf_parameter: pd.DataFrame, cne_lines: list) -> pd.DataFrame:
    """
    Filters the PTDF parameter DataFrame to include only the specified critical network elements (CNEs).
    Args:
        ptdf_parameter (pd.DataFrame): A DataFrame with a multi-index, where the second level of the index 
                                        represents the critical network elements.
        cne_lines (list): A list of critical network elements to filter on.
    Returns:
        pd.DataFrame: A DataFrame filtered to include only the specified critical network elements, 
                        with the original multi-index restored and index names removed.
    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'level_0': ['A', 'A', 'B', 'B'],
        ...     'level_1': ['CNE1', 'CNE2', 'CNE1', 'CNE3'],
        ...     'value': [10, 20, 30, 40]
        ... }
        >>> df = pd.DataFrame(data).set_index(['level_0', 'level_1'])
        >>> cne_lines = ['CNE1', 'CNE3']
        >>> filter_on_cne(df, cne_lines)
                value
        A CNE1     10
        B CNE1     30
        B CNE3     40
    """
    
    # Get the critical network elements (level_1) equal to the cne_lines
    cne_filtered_parameter = ptdf_parameter[ptdf_parameter.index.isin(cne_lines)]

    return cne_filtered_parameter
