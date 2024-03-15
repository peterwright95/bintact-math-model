### The constants need to be updated, also including the label for each categorical value 
### Something like ESA_Class = {'Cropland'; 10, etc.}

## Constants for ESA land cover classes

ESA_Class = [10,11,12,20,30,40,50,60,61,62,70,71,72,80,81,82,90,100,120,121,122,160,170,110,130,140,150,151,152,153,180,190,200,201,202,220,210]

## Constants for B-INTACT land cover classes

BINTACT_Class = [10,10,20,30,10,40,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,60,60,60,60,60,60,60,60,70,80,80,80,80,90]

## Recode mappings from ESA Class to B-INTACT Class

BIntact_recode = dict(zip(ESA_Class, BINTACT_Class))

## Constants for fragmentation classes - artificial 0, natural 1

Frag_Class = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1]

## Recode values from ESA Class to Fragmentation Class

Frag_recode = dict(zip(ESA_Class, Frag_Class))

## Final land use classes for MSA calculation
## This list will be improved

MSA_LU_Class = [10, 11, 20, 30, 31, 40, 50, 51, 52, 53, 54, 60, 61, 62, 70, 80, 81, 90, 100, 110, 120]

## Constants for MSA Land Use values 

MSA_LU_Value = [0.10, 0.30, 0.30, 0.05, 0.30, 0.50, 1.00, 0.70, 0.85, 0.50, 0.30, 1.00, 0.60, 0.30, 0.30, 0.05, 0.05, 0.90, 1.00, 0.00, 0.00]

## Recode values from MSA Land Use Class (which will eventually be the same as BIntact classes) ## to MSA_Land_Use reference values from GLOBIO.  

MSALU_Ref = dict(zip(MSA_LU_Class, MSA_LU_Value))

## Constants for MSA Infrastructure value

MSA_I_Min = 0.78

## Hectares conversion factor for MSA Fragmentation

hectares_conversion_factor = 0.0001

# Conditions for calculating MSA Fragmentation based on patch area values in hectares

patch_area_thresholds = [
    (0, 0),  # Special case for artificial LU categories
    (0, 100),
    (100, 1000),
    (1000, 10000),
    (10000, 100000),
    (100000, 1000000),
    (1000000, float('inf')),  # Use float('inf') for an upper bound that is effectively infinite
]

# Define formula components for MSA_F values corresponding to the thresholds
MSA_F_formula_components = [
    (1, None),  # For the first and last case, we use a static value of 1
    (0.0035, None),
    (0.000111111, 0.3389),
    (2.22222E-05, 0.4278),
    (2.77778E-06, 0.6222),
    (8.88889E-08, 0.8911),
    (1, None),
]

## Values to filter out for calculating Human Encroachment

Artificial_LU_values_HE = [10, 11, 20, 30, 31, 62, 70]


