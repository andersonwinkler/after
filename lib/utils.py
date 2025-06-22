#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 20:27:37 2025

@author: winkler
"""

import numpy as np
import time

# =============================================================================
def progress_bar(current, total, start_time=None, bar_length=40, prefix='Progress:', 
                    update_frequency=0.01, min_update_interval=10):
    """
    Display a progress bar with time estimation and smart update frequency.
    
    Args:
        current (int): Current iteration (0-indexed)
        total (int): Total iterations
        start_time (float): Time when the process started (from time.time())
        bar_length (int): Length of the progress bar in characters
        prefix (str): Text to display before the progress bar
        update_frequency (float): Update frequency as a fraction of total (0.01 = every 1%)
        min_update_interval (int): Minimum number of iterations between updates
    
    Returns:
        bool: True if progress was updated, False otherwise
    """
    # Initialize start time if not provided
    if start_time is None:
        start_time = time.time()
    
    # Determine update interval (max of percentage-based and minimum interval)
    update_interval = max(1, min(min_update_interval, int(total * update_frequency)))
    
    # Check if we should update the progress bar
    should_update = (current % update_interval == 0) or (current == total - 1)
    
    if not should_update:
        return False
    
    # Calculate percentage and create the progress bar
    percent = 100.0 * (current + 1) / total
    filled_length = int(bar_length * percent / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Calculate time information
    elapsed_time = time.time() - start_time
    if current > 0:  # Avoid division by zero
        iterations_per_sec = current / elapsed_time
        remaining_iterations = total - current - 1
        eta_seconds = remaining_iterations / iterations_per_sec if iterations_per_sec > 0 else 0
        
        # Format time strings
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        time_info = f"| {elapsed_str} elapsed | ETA: {eta_str}"
    else:
        time_info = ""
    
    # Create the progress message
    message = f"\r{prefix} [{bar}] {percent:.1f}% ({current+1}/{total}) {time_info}"
    
    # Print the progress bar
    print(message, end='', flush=True)
    
    # Add a newline when complete
    if current == total - 1:
        print()
    return True

# =============================================================================
def sortrows(A, columns=None, ascending=True): # Note: AI-generated
    """
    Sort matrix rows like MATLAB's sortrows function
    
    Parameters:
    A : ndarray - 2D array to sort
    columns : int, list, or None - column(s) to sort by (0-indexed)
    ascending : bool or list - sort direction(s)
    
    Returns:
    ndarray - sorted array
    """
    if A.ndim != 2:
        raise ValueError("Input must be 2D array")
    
    if columns is None:
        # Sort by all columns, left to right
        columns = list(range(A.shape[1]))
    elif isinstance(columns, int):
        columns = [columns]
    
    # Handle ascending parameter
    if isinstance(ascending, bool):
        ascending = [ascending] * len(columns)
    
    # Prepare sort keys (reverse order for lexsort)
    sort_keys = []
    for i, col in enumerate(reversed(columns)):
        key = A[:, col]
        if not ascending[len(columns) - 1 - i]:
            key = -key  # Reverse sort by negating
        sort_keys.append(key)
    
    # Sort and return
    sort_indices = np.lexsort(sort_keys)
    return A[sort_indices], sort_indices