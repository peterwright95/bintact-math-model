import geopandas as gpd
import rasterio
from rasterio.mask import mask
from scipy.ndimage import zoom
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from Constants import BIntact_recode, Frag_recode, MSALU_Ref, ESA_Class
from shapely.geometry import MultiLineString, box
import scipy.ndimage as ndimage
import os
import matplotlib.pyplot as plt

class PreProcessor:

    def __init__(self, land_use_raster_path, roads_raster_or_shape_path, evaluate_buffered = True, project_area_path = None) -> None:
        
        self.land_use_raster_path = land_use_raster_path
        self.roads_raster_or_shape_path = roads_raster_or_shape_path
        self.project_area_path = project_area_path
        self.evaluate_buffered = evaluate_buffered

    def pre_process(self,):
        self.roads_raster_or_shape_path = self.roads_raster_or_shape_path
        self.project_area_path = self.project_area_path

        # extract land use and roads rasters
        if self.project_area_path:
            (
                self.land_use_array, 
                self.land_use_transform, 
                self.land_use_meta
            ) = self.clip_input_raster_with_project_area(self.land_use_raster_path, self.project_area_path)

            # if roads raster finishes with .tif, then it is a raster, so same thing as above
            if self.roads_raster_or_shape_path.endswith('.tif'):
                (
                    self.roads_array, 
                    self.roads_transform, 
                    self.roads_meta
                ) = self.clip_input_raster_with_project_area(self.roads_raster_or_shape_path, self.project_area_path)
                self.roads_array, self.roads_transform, self.roads_meta = self.resample_roads_raster_to_match_lu(self.roads_array, self.land_use_meta, self.roads_meta)

            elif self.roads_raster_or_shape_path.endswith('.shp'):
                (
                    self.roads_array, 
                    self.roads_array_buffered, 
                    self.roads_transform, 
                    self.roads_meta, 
                    self.roadsMultiLineString, 
                    self.roadsMultiLineString_buffered
                )= self.roads_from_shp_to_tif(self.roads_raster_or_shape_path, self.land_use_raster_path, self.evaluate_buffered)

        else:
            (
                self.land_use_array, 
                self.land_use_transform, 
                self.land_use_meta
            ) = self.extract_raster_info(self.land_use_raster_path)
            if self.roads_raster_or_shape_path.endswith('.tif'):
                (
                    self.roads_array, 
                    self.roads_transform, 
                    self.roads_meta
                ) = self.extract_raster_info(self.roads_raster_or_shape_path)
                self.roads_array, self.roads_transform, self.roads_meta = self.resample_roads_raster_to_match_lu(self.roads_array, self.land_use_meta, self.roads_meta)
            elif self.roads_raster_or_shape_path.endswith('.shp'):
                (   
                    self.roads_array, 
                    self.roads_array_buffered, 
                    self.roads_transform, 
                    self.roads_meta, 
                    self.roadsMultiLineString, 
                    self.roadsMultiLineString_buffered
                 ) = self.roads_from_shp_to_tif(self.roads_raster_or_shape_path, self.land_use_raster_path, self.evaluate_buffered)

        # Recode ESA land use to B-Intact land use
        self.bintanct_lu = self.recode_raster_values(self.land_use_array, BIntact_recode)
        self.mask_for_area = np.isin(self.land_use_array, ESA_Class, invert=True)

        # Recode ESA land use to Fragmentation land classes
        self.fragmentation_lu = self.recode_raster_values(self.land_use_array, Frag_recode)
        self.fragmentation_i = self.introduce_na_values_in_fragmentation_layer(self.roads_array, self.fragmentation_lu, na_value=np.nan)
        self.infra_fragmentation_lu = self.create_infra_fragmentation_lu()

        # Get cell area from land use metadata
        self.cell_area = self.get_cell_area_from_land_use_meta(self.land_use_meta)
        self.cell_area_per_hectare = self.cell_area / 10000

    @staticmethod
    def clip_input_raster_with_project_area(raster_path, projectarea_path):
        """
        Clips and extracts rasters based on a project area shapefile, ensuring matching extents and setting CRS to LAEA projection (EPSG:6936). Needs to be updated

        Parameters:
        - raster_path (str): Path to the land use raster file.
        - projectarea_path (str): Path to the shapefile used for clipping.
        
        Returns:
        - A dictionary containing:
            - 'raster_array': NumPy array of the clipped raster.
            - 'raster_transform': Affine transform for the clipped raster.
            - 'raster_meta': Metadata for the clipped raster, updated to reflect changes.
        """
        # Load and reproject the project area shapefile to match raster CRS
        shapes = gpd.read_file(projectarea_path).to_crs("EPSG:6933")

        def clip_raster_to_array(raster_path):
            """Clip raster to the project area and return array, transform, and metadata."""
            with rasterio.open(raster_path) as src:
                clipped_array, clipped_transform = mask(src, shapes.geometry, crop=True, all_touched=True)
                meta = src.meta.copy()
                meta.update({"transform": clipped_transform, "height": clipped_array.shape[1], "width": clipped_array.shape[2], "crs": "EPSG:6936"})
                return clipped_array, clipped_transform, meta

        raster_array, raster_transform, raster_meta = clip_raster_to_array(raster_path)

        return raster_array.squeeze(), raster_transform, raster_meta
    
    @staticmethod
    def extract_raster_info(raster_path):
        with rasterio.open(raster_path) as src:
            raster_array = src.read(1)
            raster_transform = src.transform
            raster_meta = src.meta
        
        return raster_array, raster_transform, raster_meta

    @staticmethod
    def roads_from_shp_to_tif(roads_path, land_use_path, evaluate_buffered = True):
            
        # Function to iterate over the raster cells and calculate road lengths
        def calculate_road_lengths_in_raster_cells(roads, raster_meta):
            road_lengths = np.zeros((raster_meta['height'], raster_meta['width']), dtype=np.float32)
            total_cells = raster_meta['height'] * raster_meta['width']
            cells_done = 0
            
            for row in range(raster_meta['height']):
                for col in range(raster_meta['width']):
                    # Calculate the boundaries of the cell
                    x_min, y_min = rasterio.transform.xy(raster_meta['transform'], row, col, offset='ul')
                    x_max, y_max = rasterio.transform.xy(raster_meta['transform'], row+1, col+1, offset='ul')
                    
                    # Create a rectangle for the cell
                    cell_rect = box(x_min, y_min, x_max, y_max)
                    
                    # Intersect the roads with the cell rectangle and calculate the length
                    roadLengthInCell = roads.intersection(cell_rect).length
                    road_lengths[row, col] = roadLengthInCell
                    
                    cells_done += 1
                    progress_percentage = (cells_done / total_cells) * 100
                    bar_length = 20  # Modify this to change the progress bar length
                    filled_length = int(round(bar_length * cells_done / float(total_cells)))
                    bar = '#' * filled_length + '-' * (bar_length - filled_length)
                    
                    print(f"\rProgress: [{bar}] {progress_percentage:.2f}% Complete", end='')
            print("\n")

            
            return road_lengths

        with rasterio.open(land_use_path) as src:

            Roads = gpd.read_file(roads_path)

            # BUFFERING THE ROADS
            # Specify the buffer distance in m's
            buffer_distance = 1000
            # Create a new column in the GeoDataFrame with the buffered geometries
            Roads['buffered_geometry'] = Roads['geometry'].buffer(buffer_distance)
            # Working with the buffered geometries
            Roads_buffered = Roads.set_geometry('buffered_geometry')

            psroads_gdf = Roads[Roads['GP_RTP'] <= 3]
            psroads_buffered_gdf = Roads_buffered[Roads_buffered['GP_RTP'] <= 3]

            # check that the CRS of the roads is the same as the land use
            if Roads.crs != src.crs:
                raise ValueError("The CRS of the roads shapefile does not match the CRS of the land use raster.")

            # Calculate the road lengths in the raster cells
            roadsMultiLineString = psroads_gdf.geometry.unary_union
            roadsMultiLineString_buffered = psroads_buffered_gdf.geometry.unary_union
            # plot the roads

            road_lengths = calculate_road_lengths_in_raster_cells(roadsMultiLineString, src.meta)

            if evaluate_buffered:
                road_lengths_buffered = calculate_road_lengths_in_raster_cells(roadsMultiLineString_buffered, src.meta)
            else:
                road_lengths_buffered = None
        # close the raster
        src.close()
        return road_lengths, road_lengths_buffered, src.transform, src.meta, roadsMultiLineString, roadsMultiLineString_buffered

    @staticmethod 
    def resample_roads_raster_to_match_lu(roads_array, land_use_meta, roads_meta):
        """
        Resamples the roads raster to match the resolution and extent of the Land Use raster using the "nearest" method.
        Adjusts (or rather distributes) the values in the resampled raster to ensure the total sum matches the original roads raster for data integrity.

        Parameters:
        - roads_array (np.ndarray): The original roads raster array.
        - land_use_meta (dict): Metadata of the LU raster, containing 'height', 'width', and 'transform'.
        - roads_meta (dict): Metadata of the roads raster before resampling.
        
        Returns:
        - np.ndarray: The resampled roads raster array.
        - dict: Updated metadata for the resampled roads raster.
        
        Raises:
        - ValueError: If the sum of the original roads raster does not match the sum of the resampled roads raster after adjustment.
        """
        lu_rows, lu_cols = land_use_meta['height'], land_use_meta['width']
        road_rows, road_cols = roads_meta['height'], roads_meta['width']

        scaling_factor_rows = lu_rows / road_rows
        scaling_factor_cols = lu_cols / road_cols

        resampled_roads_raster = zoom(roads_array, zoom=(scaling_factor_rows, scaling_factor_cols), order=0)

        adjusted_resampled_roads_raster = resampled_roads_raster * (np.sum(roads_array) / np.sum(resampled_roads_raster))

        if not np.isclose(np.sum(roads_array), np.sum(adjusted_resampled_roads_raster)):
            raise ValueError("The sum of the original roads raster does not match the sum of the resampled roads raster.")

        # Update roads metadata to reflect the resampling
        updated_roads_meta = roads_meta.copy()
        updated_roads_meta.update({"transform": land_use_meta['transform'], "height": lu_rows, "width": lu_cols})

        return adjusted_resampled_roads_raster.read(1), adjusted_resampled_roads_raster.transform, updated_roads_meta
    
    @staticmethod
    def recode_raster_values(raster_data, recode_map):
        """
        Recodes pixel values in a raster based on a specified mapping.

        Parameters:
        - raster_data: The array containing raster data.
        - recode_map (dict): A dictionary mapping original values to their new values.

        Returns:
        - recoded_data: The array with recoded raster data.
        """
        recoded_data = raster_data.copy()
        # transform it into float to accommodate NA values (np.nan)
        recoded_data = recoded_data.astype(float)
        for original_value, new_value in recode_map.items():
            recoded_data[raster_data == original_value] = new_value

        return recoded_data

    @staticmethod
    def introduce_na_values_in_fragmentation_layer(adjusted_resampled_roads_raster, Fragmentation_LU, na_value=np.nan):
        """
        Modifies the fragmentation land use layer (Fragmentation_LU) to introduce NA values where roads exist.

        Parameters:
        - adjusted_resampled_roads_raster (np.ndarray): The resampled and adjusted roads raster array.
        - Fragmentation_LU (np.ndarray): The land use layer related to fragmentation, as a numpy array.
        - na_value (float, optional): The value to use for NA. Defaults to np.nan.

        Returns:
        - np.ndarray: The modified fragmentation layer with NA values where roads exist.
        """
        # Ensure Fragmentation_LU is a float array to accommodate NA values (np.nan)
        Frag_I = Fragmentation_LU.astype(float)
        
        # Create a boolean mask where adjusted roads raster has values greater than 0
        mask = adjusted_resampled_roads_raster > 0
        
        # Set elements in Frag_I to NA value using the mask
        Frag_I[mask] = na_value
        
        return Frag_I

    @staticmethod
    def get_cell_area_from_land_use_meta(land_use_meta):
        """
        Calculate the cell area from land use raster metadata.
        
        Parameters:
        - meta (dict): Metadata dictionary of a raster.
        
        Returns:
        - float: The area of a single cell/pixel in the raster's units squared.
        """
        # Extract pixel dimensions from the affine transform
        pixel_width = land_use_meta['transform'][0]
        pixel_height = -land_use_meta['transform'][4]  # Negating to ensure positive value, assuming top-left origin
        
        # Calculate cell area
        cell_area = pixel_width * pixel_height
        
        return cell_area
    
    def create_infra_fragmentation_lu (self,):

        na_mask = self.roads_array > 0
        # set values in Frag_array to NA where na_mask is True and assign to new array
        Infra_frag_lu = np.where(na_mask, np.nan, self.fragmentation_lu)
        # turn Infra_frag_lu into a float array
        Infra_frag_lu = Infra_frag_lu.astype(np.float32)
        # apply mask_for_all to Infra_frag_lu and assign na values
        Infra_frag_lu[self.mask_for_area] = np.nan

        return Infra_frag_lu

    @staticmethod
    def write_raster(folder_name, name, raster, meta):

        meta = meta.copy() 
        meta['dtype'] = 'float32'  # Ensure dtype matches your numpy array's dtype
        meta['count'] = 1  # Number of bands; adjust if your data structure is different
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)
        
        # Define the full path for the output raster
        output_path = os.path.join(folder_name, f"{name}.tif")
        
        # Write the raster to the specified path
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(raster, 1)  # Write the numpy array to the first band

    @staticmethod
    def plot_raster(raster, title, cmap='viridis', cmap_title='Class'):
        plt.figure(figsize=(10, 10))
        plt.imshow(raster, cmap=cmap)
        plt.title(title, fontsize=20)
        plt.colorbar(label=cmap_title)
        plt.show()

class Landscape:
    """
    A class to represent the project area landscape and perform operations such as patch labeling and area calculations.
    A patch is a homogenous group of a land use. i.e., adjacent and touching pixels of the same land use are grouped as a patch. 

    Attributes:
    - landscape_arr (np.ndarray): The landscape array where each cell represents a land cover class.
    - cell_area (float): The area represented by each cell in the landscape array.
    """

    def __init__(self, landscape_arr, cell_area):
        """
        Initializes the Landscape with a given array and cell area.

        Parameters:
        - landscape_arr (np.ndarray): The landscape array.
        - cell_area (float): The area represented by each cell.
        """
        self.landscape_arr = landscape_arr
        self.cell_area = cell_area
        self.artificial_area_mask = landscape_arr == 0

    def class_label(self, class_val):
        """
        Labels connected components (pixels) within the landscape array for a specific class value.

        Parameters:
        - class_val (int): The class value to label in the landscape array.

        Returns:
        - np.ndarray: An array with labeled components for the specified class.
        """
        bool_array = (self.landscape_arr == class_val)
        labeled_array, num_features = ndimage.label(
            bool_array,
            structure=ndimage.generate_binary_structure(2, 2)
        )
        return labeled_array

    def compute_patch_areas(self, label_arr):
        """
        Computes the areas of labeled patches in the landscape.

        Parameters:
        - label_arr (np.ndarray): An array with labeled patches.

        Returns:
        - dict: A dictionary with patch IDs as keys and their respective areas as values.
        """
        # Calculate the bincount but ignore the zero label
        patch_counts = np.bincount(label_arr.ravel())[1:]  # Skip count for background (0)
        
        # Multiply by the cell area to get actual areas
        patch_areas = patch_counts * self.cell_area
        
        # Create a dictionary of patch_id to patch_area
        patch_area_dict = {patch_id: area for patch_id, area in enumerate(patch_areas, start=1)}
        
        return patch_area_dict


