import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def plot_raster_with_colormap(input_folder, output_folder, cmap='viridis'):
    """
    Plots all .tif files in a folder with a color scale and saves the plots.

    Parameters:
    - input_folder (str): Path to the folder containing .tif files.
    - output_folder (str): Path to save the plot images.
    - cmap (str): Colormap to use for visualization. Default is 'viridis'.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all .tif files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tif'):
            raster_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
            
            # Open the raster file
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)  # Read the first band
                
                # Replace nodata values with NaN for better visualization
                raster_data = np.where(raster_data == src.nodata, np.nan, raster_data)
            
            # Setting up the plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(raster_data, cmap=cmap)
            ax.set_title(f'Raster: {file_name}')
            ax.axis('off')  # Turn off axis for better visualization
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Save the plot
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close the figure to free memory
            print(f"Plot saved to {output_path}")

# Example Usage
input_folder = "/Users/peter/Documents/b-intact-math-model/output"
output_folder = "/Users/peter/Documents/b-intact-math-model/output2"
plot_raster_with_colormap(input_folder, output_folder, cmap='viridis')