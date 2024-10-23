from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    lst = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        result.extend(reversed_group)
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
        
    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}
     def _flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():
            # Create a new key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten the sub-dictionary
                _flatten(value, new_key)
            elif isinstance(value, list):
                # Handle lists by enumerating over the elements
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        _flatten(item, f"{new_key}[{index}]")
                    else:
                        flattened[f"{new_key}[{index}]"] = item
            else:
                # Base case: add to flattened dictionary
                flattened[new_key] = value

    _flatten(nested_dict)
    return flattened

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        # If we've reached the end of the array, add the current permutation
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            # Swap the current element with the start element
            nums[start], nums[i] = nums[i], nums[start]
            # Recur on the next index
            backtrack(start + 1)
            # Backtrack: restore the original order
            nums[start], nums[i] = nums[i], nums[start]

    nums.sort()  # Sort to handle duplicates
    result = []
    backtrack(0)
    return result
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
     patterns = [
        r'\b([0-2][0-9]|(3)[0-1])-(0[1-9]|1[0-2])-(\d{4})\b',  # dd-mm-yyyy
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[1-2][0-9]|3[0-1])/\d{4}\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[1-2][0-9]|3[0-1])\b'  # yyyy.mm.dd
    ]
    combined_pattern = '|'.join(patterns)
    matches = re.findall(combined_pattern, text)
    valid_dates = [''.join(match) for match in matches if any(match)]
    return valid_dates
    pass

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0]  # First point has no previous point
    for i in range(1, len(df)):
        distance = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(distance)
    df['distance'] = distances
    
        

    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

     transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            transformed_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]
    
    return transformed_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Create a multi-index from (id, id_2)
    grouped = df.groupby(['id', 'id_2'])
    
    # Initialize a series to hold results
    results = pd.Series(index=grouped.groups.keys(), dtype=bool)

    for (id_value, id_2_value), group in grouped:
        # Check if all 7 days are present
        days_covered = group['start'].dt.day_name().unique()
        all_days_present = len(days_covered) == 7
        
        # Check if all days cover a full 24-hour period
        full_24_hour_cover = all(group['end'].dt.date == group['start'].dt.date)
        daily_cover = group.groupby(group['start'].dt.date).apply(
            lambda x: (x['end'] - x['start']).dt.total_seconds().max() >= 86400  # 86400 seconds in 24 hours
        ).all()
        
        results[(id_value, id_2_value)] = not (all_days_present and daily_cover)

    return results
