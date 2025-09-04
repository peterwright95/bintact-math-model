from pre_processor import PreProcessor, Landscape
from Constants import BIntact_recode, Frag_recode, MSALU_Ref, MSA_I_Min, hectares_conversion_factor, patch_area_thresholds, MSA_F_formula_components, Artificial_LU_values_HE
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from collections import Counter, deque, defaultdict
from helper_functions import plot_patches
import pickle
import os



class MSA_Calculator():

    def __init__(self,land_use_raster_path, 
                 roads_raster_path, 
                 extended_area_infra_frag_lu_path, 
                 extended_area_roads_path, 
                 project_area_path = None) -> None:

        # self.pre_processor = PreProcessor(land_use_raster_path, roads_raster_path)
        # self.pre_processor.pre_process()

        # open the pre_processor object from a file with a pickle
        with open('/Users/peter/Documents/b-intact-math-model/results/pre_processor.pkl', 'rb') as f:
            self.pre_processor = pickle.load(f)
        
        # save the pre_processor object to a file with a pickle, no method to save the object to a file
        # create the results folder if it doesn't exist
        os.makedirs('results', exist_ok=True)
        with open('results/pre_processor.pkl', 'wb') as f:
            pickle.dump(self.pre_processor, f)
    
        
        self.landscape = Landscape(self.pre_processor.land_use_array, self.pre_processor.cell_area)

        # NECESSARY PREPROSESSES DATA FOR MSA_F CALCULATION
        (self.extended_area_infra_frag_lu_array, 
        self.extended_area_infra_frag_transform, 
        self.extended_area_infra_frag_meta) = self.pre_processor.extract_raster_info(extended_area_infra_frag_lu_path)
        (self.extended_area_roads_array, 
         self.extended_area_roads_transform, 
         self.extended_area_roads_meta) = self.pre_processor.extract_raster_info(extended_area_roads_path)
        self.extended_area_cell_area = self.pre_processor.get_cell_area_from_land_use_meta(self.extended_area_infra_frag_meta)

        # check that cell area is the same for both rasters with a tolerance of 0.1
        if abs(self.pre_processor.cell_area - self.extended_area_cell_area) > 0.1:
            raise ValueError('The cell area for the extended area and the original area are different')
        
        # Instantiate the MSA values
        self.MSA_LU = None
        self.MSA_I = None
        self.MSA_F = None
        self.MSA_HE = None
        self.MSA = None

    def apply_general_area_mask(self, MSA_raster):
        """
        Apply the area mask to the MSA raster.

        Parameters:
        - MSA_raster: numpy array of MSA values.

        Returns:
        - MSA_raster: numpy array of MSA values with the area mask applied.
        """
        MSA_raster[self.pre_processor.mask_for_area] = np.nan
        return MSA_raster
    
    def calculate_MSA_LU(self):

        # Recode land use raster values
        self.pre_processor.write_raster('output', 'before_msa_lu', self.pre_processor.bintanct_lu, self.pre_processor.land_use_meta)
        MSA_LU = self.pre_processor.recode_raster_values(self.pre_processor.bintanct_lu, MSALU_Ref)

        return MSA_LU

    def calculate_MSA_Infrastructure(self,):
        """
        Calculates MSA values based on land use fragmentation, road presence, and pixel area,
        using constants for minimum MSA Infrastructure value and extracting pixel size from  land_use_meta.

        Parameters:
        - Fragmentation_LU (np.ndarray): The land use layer related to fragmentation.
        - adjusted_resampled_roads_raster (np.ndarray): The adjusted roads raster array.
        - land_use_meta (dict): Metadata for the land use raster, including pixel size and CRS.

        Returns:
        - np.ndarray: An array with MSA values calculated based on specified conditions.
        """

        MSA_I = np.zeros(self.pre_processor.fragmentation_lu.shape, dtype=np.float32)

        mask = (self.pre_processor.fragmentation_lu == 1) & (self.pre_processor.roads_array_buffered > 0)
        
        self.pre_processor.write_raster('output', 'roads_array_buffered', self.pre_processor.roads_array_buffered, self.pre_processor.land_use_meta)

        MSA_I[mask] = 0.78
        MSA_I[(self.pre_processor.fragmentation_lu == 0) | ((self.pre_processor.fragmentation_lu == 1) & (self.pre_processor.roads_array_buffered <= 0))] = 1

        # apply mask_for_all to MSA_I_rast and assign na values
        MSA_I = self.apply_general_area_mask(MSA_I)

        return MSA_I
    
    def calculate_MSA_F(self, class_val):
        """
        Calculate the Mean Species Abundance (MSA) Fragmentation.

        Parameters:
        - landscape_arr: A numpy array representing the landscape.
        - cell_area: float, area of a single cell in the landscape_arr.
        - class_val: int, the class value for which to calculate MSA_F.

        Returns:
        - MSA_F: A numpy array, same shape as landscape_arr, with MSA Fragmentation values.
        """

        def assign_roads_to_patches(class_1_labels_original, roads, patch_areas, radius, pre_processor):
            
            def find_borders(class_labels):
                borders = []
                rows, cols = class_labels.shape

                for i in range(rows):
                    for j in range(cols):
                        # if i and j are not nan
                        if np.isnan(class_labels[i, j]):
                            continue

                        else:
                            # Collect neighbors that are within the bounds of the matrix
                            neighbors = []
                            if i - 1 >= 0: neighbors.append(class_labels[i - 1, j])
                            if i + 1 < rows: neighbors.append(class_labels[i + 1, j])
                            if j - 1 >= 0: neighbors.append(class_labels[i, j - 1])
                            if j + 1 < cols: neighbors.append(class_labels[i, j + 1])

                            # Check if the current pixel value is different from at least one of its neighbors
                            if set(neighbors) != {class_labels[i, j]}:
                                borders.append((i, j))
                        

                return borders
            
            def find_unassigned_neighbours(i, j, roads, class_1_labels):

                neighbours = []
                # verify that i - 1 is within the bounds of the array
                if i - 1 >= 0:
                    if roads[i - 1, j] == 1 and np.isnan(class_1_labels[i - 1, j]):
                        neighbours.append((i - 1, j))
                
                # verify that i + 1 is within the bounds of the array
                if i + 1 < roads.shape[0]:
                    if roads[i + 1, j] == 1 and np.isnan(class_1_labels[i + 1, j]):
                        neighbours.append((i + 1, j))

                # verify that j - 1 is within the bounds of the array
                if j - 1 >= 0:
                    if roads[i, j - 1] == 1 and np.isnan(class_1_labels[i, j - 1]):
                        neighbours.append((i, j - 1))

                # verify that j + 1 is within the bounds of the array
                if j + 1 < roads.shape[1]:
                    if roads[i, j + 1] == 1 and np.isnan(class_1_labels[i, j + 1]):
                        neighbours.append((i, j + 1))
                        
                return neighbours
            
            def get_most_frequent_patch_in_area(i, j, matrix, width_of_analysis, patch_areas):
                rows, cols = matrix.shape
                # Calculate start and end indices ensuring they stay within matrix bounds
                start_i = max(0, i - width_of_analysis)
                end_i = min(rows, i + width_of_analysis + 1)
                start_j = max(0, j - width_of_analysis)
                end_j = min(cols, j + width_of_analysis + 1)
                
                # Get the matrix of width 'width_of_analysis' around (i, j)
                area_around = matrix[start_i:end_i, start_j:end_j]
                
                # Remove nans
                area_around_no_nans = area_around[~np.isnan(area_around)]
                
                # Flatten the matrix and count frequencies
                flattened_area = area_around_no_nans.flatten()
                counter = Counter(flattened_area)
                
                # Get the most common patch, excluding 0s
                if len(counter) == 0:
                    # return exception if the area has no non-zero elements
                    raise ValueError('The area has no non-zero elements, something weird happened')
                else:
                    # check if there are patches with the same frequency
                    most_common_patch = counter.most_common(1)[0][0]
                    most_common_patch_frequency = counter.most_common(1)[0][1]

                    # check if there are patches with the same frequency
                    patches_with_same_frequency = [patch for patch, frequency in counter.items() if frequency == most_common_patch_frequency]
                    # if there are patches with the same frequency, return the one with the largest area    
                    if len(patches_with_same_frequency) > 1:
                        patch_areas_dict = {patch: patch_areas[patch] for patch in patches_with_same_frequency}
                        most_common_patch = max(patch_areas_dict, key=patch_areas_dict.get)

                    return most_common_patch


            '''
            Requires:
            - class_1_labels_original: a 2D FLOAT numpy array with the original class 1 labels
            - roads: a 2D numpy array with the roads
            - patch_areas: a list with the areas of each patch
            - radius: the distance from the pixel to consider when calculating the most frequent patch
            '''   
            borders = find_borders(class_1_labels_original)
            queue = deque(borders)

            # copy class 1 labels
            class_1_labels = class_1_labels_original.copy()
            while queue:
                x, y = queue.popleft()
                unassigned_neighbours = find_unassigned_neighbours(x, y, roads, class_1_labels)

                if not unassigned_neighbours:
                    continue
                for i, j in unassigned_neighbours:
                    most_frequent_patch = get_most_frequent_patch_in_area(i, j, class_1_labels, radius, patch_areas)
                    if not np.isnan(most_frequent_patch):
                        class_1_labels[i, j] = most_frequent_patch
                        # append to queue if value is not in queu already
                        if (i, j) not in queue:
                            queue.append((i, j))
            
            return class_1_labels
        
        def identify_boundary_patches(original_labels):
            boundary_patches = set()
            rows, cols = original_labels.shape
            for i in range(rows):
                for j in range(cols):
                    if i == 0 or i == rows-1 or j == 0 or j == cols-1:
                        if not np.isnan(original_labels[i, j]):
                            boundary_patches.add(original_labels[i, j])
            return boundary_patches

        def find_corresponding_patch_id(original_patch_id, original_labels, extended_labels, original_transform, extended_transform):

            def pixel_to_coord(transform, row, col):
                a, b, c, d, e, f, _, _, _ = transform
                # Compute spatial coordinates from pixel coordinates
                x = a * col + b * row + c
                y = d * col + e * row + f  # Note: 'd' typically 0 in north-up images, used for rotation
                return (x, y)

            def coord_to_pixel(transform, x, y):
                a, b, c, d, e, f, _, _, _ = transform
                if a == 0 or e == 0:  # Corrected to check 'e' instead of 'd' for scale in Y direction
                    print("Error: Scale factor 'a' or 'e' is zero.")
                    return None, None

                # Correcting the formula to account for potential negative scale in 'e'
                col = (x - c) / a
                row = (y - f) / e  # 'd' is not used because it's typically 0, and 'e' is used for Y scale

                return (int(row), int(col))
            
            original_coords = np.column_stack(np.where(original_labels == original_patch_id))
            extended_patch_id_counts = defaultdict(int)

            number_of_pixels = len(original_coords)
            
            for (row, col) in original_coords:
                # Transform to spatial coordinates
                (x, y) = pixel_to_coord(original_transform, row, col)
                
                # Transform back to extended raster pixel coordinates
                (row_ext, col_ext) = coord_to_pixel(extended_transform, x, y)
                
                # Check bounds and count occurrences
                if 0 <= row_ext < extended_labels.shape[0] and 0 <= col_ext < extended_labels.shape[1]:
                    extended_patch_id = extended_labels[row_ext, col_ext]
                    if not np.isnan(extended_patch_id):
                        extended_patch_id_counts[extended_patch_id] += 1
            
            # Get the most common extended patch id
            if extended_patch_id_counts:
                corresponding_patch = max(extended_patch_id_counts, key=extended_patch_id_counts.get)
                confidence = extended_patch_id_counts[corresponding_patch] / number_of_pixels
                return corresponding_patch, confidence  
            else:
                return None, None

        landscape = Landscape(self.pre_processor.infra_fragmentation_lu, self.pre_processor.cell_area_per_hectare)

        class_1_labels = landscape.class_label(class_val)
        self.pre_processor.write_raster('output', 'patches', class_1_labels, self.pre_processor.land_use_meta)
        
        class_1_patch_areas = landscape.compute_patch_areas(class_1_labels)
        class_1_labels = class_1_labels.astype(np.float32)

        artificial_mask = (class_1_labels == 0)
        class_1_labels[class_1_labels == 0] = np.nan

        roads_array = self.pre_processor.roads_array
        roads_array[roads_array > 0] = 1

        # CHECK IF SOME OF THE PATCHES REACH THE BOUNDARY OF THE RASTER
        boundary_patches = identify_boundary_patches(class_1_labels)
        if boundary_patches:
            landscape_extended = Landscape(self.extended_area_infra_frag_lu_array, self.pre_processor.cell_area_per_hectare)
            class_1_labels_extended = landscape_extended.class_label(class_val)
            self.pre_processor.write_raster('output', 'patches_extended', class_1_labels_extended, self.pre_processor.land_use_meta)
            class_1_patch_areas_extended = landscape_extended.compute_patch_areas(class_1_labels_extended)
            class_1_labels_extended = class_1_labels_extended.astype(np.float32)
            class_1_labels_extended[class_1_labels_extended == 0] = np.nan

            corresponding_patch_ids = {}
            maximum_corresponding_patch_id = max(class_1_labels_extended.flatten())
            for patch_id in class_1_patch_areas:
                corresponding_patch, confidence_patch = find_corresponding_patch_id(patch_id, class_1_labels, class_1_labels_extended, self.pre_processor.land_use_transform, self.extended_area_infra_frag_transform)
                corresponding_patch_ids[patch_id] = (corresponding_patch, confidence_patch)

            class_1_patch_areas_reassigned = {}
            counter = 1
            for original_patch_id, (extended_patch_id, confidence) in corresponding_patch_ids.items():
                if original_patch_id in boundary_patches:
                    if extended_patch_id != None:
                        # change the values in the original patch to the corresponding patch in the extended raster
                        class_1_labels[class_1_labels == original_patch_id] = extended_patch_id
                        # filter the class_1_patch_areas_extended to only include patches in corresponding_patch_ids and reassign class_1_patch_areas
                        class_1_patch_areas_reassigned[extended_patch_id] = class_1_patch_areas_extended[extended_patch_id]
                        print(f'Original patch id: {original_patch_id} Matched to extended patch id: {extended_patch_id}, confidence: {confidence}')
                    else:
                        raise ValueError(f'Original patch id: {original_patch_id} not matched to any extended patch id, is boundary patch, should not happen')
                
                else:
                    new_patch_id = maximum_corresponding_patch_id + counter
                    class_1_labels[class_1_labels == original_patch_id] = new_patch_id
                    class_1_patch_areas_reassigned[new_patch_id] = class_1_patch_areas[original_patch_id]
                    counter += 1       
                    # print(f'Original patch id: {original_patch_id} not boundary patch, kept as was')


            class_1_patch_areas = class_1_patch_areas_reassigned   

        self.pre_processor.write_raster('output', 'patches_after_boundary_assignment', class_1_labels, self.pre_processor.land_use_meta)
        radius = 1 # distance from the pixel to consider when calculating the most frequent patch
        class_1_labels_new = assign_roads_to_patches(class_1_labels, roads_array, class_1_patch_areas, radius, self.pre_processor)
        self.pre_processor.write_raster('output', 'patches_reassigned', class_1_labels_new, self.pre_processor.land_use_meta)
        
        # TODO: We are reassigning the roads in the scope to the patches, but we are not updating the patch areas (so the added area of the roads is not considered)
        labels = np.unique(class_1_labels_new)
        labels = labels[~np.isnan(labels)]

        # CHECK THAT ALL VALUES WHERE ROADS ARE PRESENT ARE ASSIGNED TO A PATCH
        mask = (roads_array > 0) & (np.isnan(class_1_labels_new))
        if np.any(mask):
            raise ValueError('Some pixels where roads are present are not assigned to a patch')    
        
        # create an array defaulted to nan with the same shape as class_1_labels_new

        patch_area_values_in_hectares = np.full(class_1_labels_new.shape, np.nan)

        for label in labels:
            mask = class_1_labels_new == label
            patch_area_values_in_hectares[mask] = class_1_patch_areas[label]     
        
        self.pre_processor.write_raster('output', 'patch_areas', patch_area_values_in_hectares, self.pre_processor.land_use_meta)
        conditions = [
            (patch_area_values_in_hectares == 0), # the patch areais always 0 for artificial LU categories 
            (patch_area_values_in_hectares >= 0) & (patch_area_values_in_hectares < 100),
            (patch_area_values_in_hectares >= 100) & (patch_area_values_in_hectares < 1000),
            (patch_area_values_in_hectares >= 1000) & (patch_area_values_in_hectares < 10000),
            (patch_area_values_in_hectares >= 10000) & (patch_area_values_in_hectares < 100000),
            (patch_area_values_in_hectares >= 100000) & (patch_area_values_in_hectares < 1000000),
            (patch_area_values_in_hectares >= 1000000)
        ]

        # Define the corresponding values for each condition
        values = [
            1,
            patch_area_values_in_hectares * 0.0035,
            patch_area_values_in_hectares * 0.000111111 + 0.3389,
            patch_area_values_in_hectares * 2.22222E-05 + 0.4278,
            patch_area_values_in_hectares * 2.77778E-06 + 0.6222,
            patch_area_values_in_hectares * 8.88889E-08 + 0.8911,
            1
        ]
        # Use np.select to apply conditions and values
        MSA_F = np.select(conditions, values, default=1)
        MSA_F[self.pre_processor.mask_for_area] = np.nan

        return MSA_F    
    
    def calculate_MSA_human_encroachment(self, land_use_array):
        # Create a mask for filtering cropland/urbanland
        mask = np.isin(self.pre_processor.bintanct_lu, Artificial_LU_values_HE)
        self.pre_processor.write_raster('output', 'msa_he_mask', mask, self.pre_processor.land_use_meta)

        total_area_ha = self.pre_processor.cell_area_per_hectare * self.pre_processor.land_use_array.size
        total_area_mask_ha = np.sum(mask) * self.pre_processor.cell_area_per_hectare

        cropland_share = total_area_mask_ha / total_area_ha

        # Assuming croplandshare is already calculated and is a float
        MSAHE_float = 0.85 if cropland_share >= 0.015 else 1 - (cropland_share / 0.015) * (1 - 0.85)
        MSA_HE = np.full(self.pre_processor.land_use_array.shape, MSAHE_float)

        # Update MSA_HE values where LU is in filter_values to 1
        MSA_HE[mask] = 1
        
        return MSA_HE
    
    def calculate_final_MSA(self,):
        """
        Calculates the final Mean Species Abundance (MSA) by combining the MSA from land use, fragmentation,infrastructure, and human encroachment.

        Parameters:
        - MSA_LU: numpy array of MSA values for land use.
        - MSA_I: numpy array of MSA values for infrastructure.
        - MSA_F: numpy array of MSA values for fragmentation.
        - MSA_HE: numpy array of MSA values for human encroachment.

        Returns:
        - MSA_Aggregate: The final aggregate MSA value for the area.
        """
        # Calculate intermediate MSA product for each pixel
        MSA_Pixel = self.MSA_LU * self.MSA_I * self.MSA_F * self.MSA_HE
        
        total_pixel_notNA = np.sum(~np.isnan(MSA_Pixel))
        total_area_final = total_pixel_notNA * self.pre_processor.cell_area_per_hectare

        sum_non_na = np.nansum(MSA_Pixel)
        #sum_non_na = np.sum(MSA_Pixel)

        # to calculate the MSA Aggregate i.e. the Mean MSA for the area, the approach that we used is make an average 
        # between all the values of the MSA at pixel level. 
        # Therefore the sum of all the values in the pixel is divided by the total number of pixels and not the total area
        MSA_Aggregate = sum_non_na / total_pixel_notNA
        
        return MSA_Pixel, MSA_Aggregate
    
    def calculate_MSA_rasters(self):

        # Calculate Mean Species Abundance (MSA) from land use
        self.MSA_LU = self.calculate_MSA_LU()

        # Calculate Mean Species Abundance (MSA) from infrastructure
        self.MSA_I = self.calculate_MSA_Infrastructure()
        
        # Calculate Mean Species Abundance (MSA) from fragmentation
        self.MSA_F = self.calculate_MSA_F(1)

        # Calculate Mean Species Abundance (MSA) from Human Encroachment
        self.MSA_HE = self.calculate_MSA_human_encroachment(self.pre_processor.land_use_array)

        # Calculate Mean Species Abundance for the project area
        self.MSA_Pixel, self.MSA = self.calculate_final_MSA()

        self.pre_processor.write_raster('results/', 'MSA_LU', self.MSA_LU, self.pre_processor.land_use_meta)
        self.pre_processor.write_raster('results/', 'MSA_I', self.MSA_I, self.pre_processor.land_use_meta)
        self.pre_processor.write_raster('results/', 'MSA_F', self.MSA_F, self.pre_processor.land_use_meta)
        self.pre_processor.write_raster('results/', 'MSA_HE', self.MSA_HE, self.pre_processor.land_use_meta)
        self.pre_processor.write_raster('results/', 'MSA_Pixel', self.MSA_Pixel, self.pre_processor.land_use_meta)
        self.pre_processor.write_raster('results/', 'Land_Use', self.pre_processor.bintanct_lu, self.pre_processor.land_use_meta)

    
def compare_rasters_and_show_difference(MSI_I_rast, raster_file_path):
    # Read the second raster from the file path
    with rasterio.open(raster_file_path) as src:
        MSI_I_file_rast = src.read(1)  # Assuming you're interested in the first band

    # Calculate the difference between the two rasters and then round to 6 decimal places
    difference = MSI_I_rast - MSI_I_file_rast
    difference = np.round(difference, 6)

    # Setting up the plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    cmap = 'viridis'  # Choose a colormap for the original rasters
    diff_cmap = 'RdBu'  # Choose a colormap for the difference, one that centers on zero differences is ideal

    # Plotting the first raster
    im0 = ax[0].imshow(MSI_I_rast, cmap=cmap)
    ax[0].set_title('Raster in Memory')
    ax[0].axis('off')  # Turn off axis for better visualization
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # Plotting the second raster
    im1 = ax[1].imshow(MSI_I_file_rast, cmap=cmap)
    ax[1].set_title('Raster from File')
    ax[1].axis('off')  # Turn off axis for better visualization
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # Plotting the difference
    im2 = ax[2].imshow(difference, cmap=diff_cmap)
    ax[2].set_title('Difference (In Memory - File)')
    ax[2].axis('off')  # Turn off axis for better visualization
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    # Adjusting layout
    plt.tight_layout()

    # save the plot with the last character of the file path as the name, from the last backslash
    plt.savefig(f'./difference_plot_{raster_file_path.split("/")[-1]}.png')


if __name__ == "__main__":
    ################################ test the MSA_Calculator class ################################################
    land_use_raster_path = '/Users/peter/Documents/MSA R Code/Test_area_Andorra/DRC_LU_basin_6933.tif'
    roads_raster_or_shp_path = '/Users/peter/Documents/MSA R Code/Test_area_Andorra/Andorra_Roads/DRC_roads.shp'
    project_area_path = None # OPTIONAL: not tested as I didn't have a project area raster
    land_use_raster_path_extended = ''
    roads_raster_or_shp_path_extended = ''
    ################################################################################################################

    # EXTENDED AREA PRE-ANALYSIS
    pre_processor_extended = PreProcessor(land_use_raster_path_extended, roads_raster_or_shp_path_extended, evaluate_buffered=False)
    pre_processor_extended.pre_process()

    pre_processor_extended.write_raster('results', 
                                        'infra_frag_lu_extended', 
                                        pre_processor_extended.infra_fragmentation_lu, 
                                        pre_processor_extended.land_use_meta)
    
    pre_processor_extended.write_raster('results',
                                        'roads_extended', 
                                        pre_processor_extended.roads_array, 
                                        pre_processor_extended.land_use_meta)
    
    extended_area_infra_frag_lu_path = '/Users/peter/Documents/MSA R Code/results/infra_frag_lu_extended.tif'
    extended_area_roads_path =  '/Users/peter/Documents/MSA R Code/results/roads_extended.tif'
    
    
    # AREA OF ANALYSIS COMPUTATION
    calculator = MSA_Calculator(land_use_raster_path, 
                                roads_raster_or_shp_path, 
                                extended_area_infra_frag_lu_path, 
                                extended_area_roads_path,  
                                project_area_path)
    
    calculator.calculate_MSA_rasters()