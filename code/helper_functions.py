import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import find_objects

def plot_patches(original_patch_id, extended_patch_id, original_labels, extended_labels):
    """
    Plots the original and extended patches with given patch IDs.
    
    :param original_patch_id: The ID of the patch in the original raster.
    :param extended_patch_id: The ID of the patch in the extended raster.
    :param original_labels: The array of patch labels for the original raster.
    :param extended_labels: The array of patch labels for the extended raster.
    """
    
    # Get slices for the original and extended patch IDs
    original_slice = find_objects(original_labels==original_patch_id)[0]
    extended_slice = find_objects(extended_labels==extended_patch_id)[0]
    
    # Get the patch data
    original_patch = (original_labels[original_slice] == original_patch_id).astype(int)
    extended_patch = (extended_labels[extended_slice] == extended_patch_id).astype(int)
    
    # Set up a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the original patch
    ax1.imshow(original_patch, cmap='viridis')
    ax1.set_title(f'Original Patch {original_patch_id}')
    ax1.axis('off')
    
    # Plot the extended patch
    ax2.imshow(extended_patch, cmap='viridis')
    ax2.set_title(f'Extended Patch {extended_patch_id}')
    ax2.axis('off')
    
    # Display the plots
    plt.tight_layout()
    plt.show()