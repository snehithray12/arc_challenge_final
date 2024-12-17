from utils import *
from vision import get_background_color, percieve_obj_map, percieve_top_bottom_left_right, isolate_object, zoom_in
from collections import defaultdict
import numpy as np
import math
import scipy.ndimage

##utility function
###given a number representing which cell in the object we want to access (1 for 1st cell, 2 for 2nd, and so on)
##returns the (x, y) global position of that cell
def get_cell_position(obj, cell_number):
    if obj is None or cell_number is None:
        return 0
    cell_number -= 1
    object_map = obj.get_object_map()
    index = cell_number % object_map.shape[0]
    return object_map[index]

####ACTION SKILLS 
"""
input: 
obj: numpy grid containing only object
dx: int, amount to move in the up and down direction in the grid
dy: int, amount to move in the left and right direction in the grid
output:
returns a numpy grid containing translated object
"""
def translate(obj, dx, dy):
    if obj is None or dx is None or dy is None:
        return np.zeros(1)
    obj = obj.astype(int)
    dx = int(dx)
    dy = int(dy)
    background_color = get_background_color(obj)
    # Create a new object filled with the background color
    new_obj = np.ones(obj.shape, dtype=int) * background_color
    # Get coordinates of non-background elements
    non_bg_coords = np.argwhere(obj != background_color)
    # Calculate new positions after translation
    new_coords = non_bg_coords + np.array([dx, dy])
    # Apply translation if the new positions are within bounds
    valid_mask = (new_coords[:, 0] >= 0) & (new_coords[:, 0] < obj.shape[0]) & \
                 (new_coords[:, 1] >= 0) & (new_coords[:, 1] < obj.shape[1])
    # Filter valid coordinates
    valid_new_coords = new_coords[valid_mask]
    valid_old_coords = non_bg_coords[valid_mask]
    # Place translated values into the new object
    new_obj[valid_new_coords[:, 0], valid_new_coords[:, 1]] = obj[valid_old_coords[:, 0], valid_old_coords[:, 1]]
    return new_obj


"""
input: 
obj: numpy grid containing only object
direction: string, "lr" or "ud" are options. "lr" for rotating abt left right, "ud" for up down. 
axis: int, a row or column in scene (depending on whether rotation direction is lr or ud) to rotate about
output:
returns a numpy grid containing rotated object
"""
def flip(obj, direction, axis):
    if obj is None or direction is None or axis is None:
        return np.zeros(1)
    obj = obj.astype(int)
    axis = int(axis)
    ##extract object using corner points
    scene = np.ones(obj.shape, dtype=int) * get_background_color(obj)
    obj_map = percieve_obj_map(obj)
    top, bottom, left, right = percieve_top_bottom_left_right(obj_map)
    isolated_obj = isolate_object(obj, top, bottom, left, right)
    ##flipping left/right or up/down
    if direction == 'lr':
        ##calculate how far axis is from the side to know where to place rotated object
        isolated_obj = np.fliplr(isolated_obj)
        if abs(axis - left) <= abs(axis - right):
            distance_from_axis = axis - left
        else:
            distance_from_axis = axis - right
        new_pivot_side = distance_from_axis + axis
        new_far_side = new_pivot_side + abs(left - right)
        if new_pivot_side >= new_far_side:
            target_bounds = top, bottom, new_far_side, new_pivot_side
        elif new_pivot_side < new_far_side:
            target_bounds = top, bottom, new_pivot_side, new_far_side
        flipped_obj = place_new(scene, isolated_obj, target_bounds)
    elif direction == 'ud':
        isolated_obj = np.flipud(isolated_obj)
        if abs(axis - bottom) <= abs(axis - top):
            distance_from_axis = axis - bottom
        else:
            distance_from_axis = axis - top
        new_pivot_side = distance_from_axis + axis
        new_far_side = new_pivot_side + abs(top - bottom)
        if new_pivot_side >= new_far_side:
            target_bounds = new_far_side, new_pivot_side, left, right
        elif new_pivot_side < new_far_side:
            target_bounds = new_pivot_side, new_far_side, left, right
        flipped_obj = place_new(scene, isolated_obj, target_bounds)
    return flipped_obj

"""
input: 
obj: numpy grid containing only object
number_cell: int, indicating which cell of object to extend. object cells are labeled left to right, top to bottom. 
Ex. Cell number 1 is the top-left object, and so on. Only cells in the object are given an object specific numbering. Background cells/other objects are skipped.
OR 
cell: cell itself. tuple indicating where to start extension. 
direction: string, "up", "down", "left", "right", "up-right", "up-left", "down-right", and "down-left" are the options. 
The direction you want the object to be extended in.
distance: int, the distance you want to extend the object by. 1 adds a single new cell/extends by a single new cell, and so on.
output:
returns a numpy grid containing extended object
"""
def extend(obj, cell, direction, distance, color = None):
    if color is None:
        color = obj[cell[0], cell[1]]
    if obj is None or cell is None or direction is None or distance is None:
        return np.zeros(1)
    obj = obj.astype(int)
    ##extension is done recursively. recurses until the total distance has been extended.
    def recurse_extend(obj, cell, direction, distance):
        if distance <= 0:
            return obj
        if direction == "up":
            target_cell = cell[0] - 1, cell[1]
        elif direction == "down":
            target_cell = cell[0] + 1, cell[1]
        elif direction == "left":
            target_cell = cell[0], cell[1] - 1
        elif direction == "right":
            target_cell = cell[0], cell[1] + 1
        elif direction == "up-right":
            target_cell = cell[0] - 1, cell[1] + 1
        elif direction == "up-left":
            target_cell = cell[0] - 1, cell[1] - 1
        elif direction == "down-right":
            target_cell = cell[0] + 1, cell[1] + 1
        elif direction == "down-left":
            target_cell = cell[0] + 1, cell[1] - 1
        if target_cell[0] < 0 or target_cell[0] >= obj.shape[0] or target_cell[1] < 0 or target_cell[1] >= obj.shape[1]:
            return obj
        obj[target_cell[0], target_cell[1]] = obj[cell[0], cell[1]]
        distance -= 1
        val = recurse_extend(obj=obj, cell=target_cell, direction=direction, distance=distance)
        return val
    ##gets map of object cell values
    obj_map = percieve_obj_map(obj)
    extended_obj = obj.copy()
    ##handling either case of data type value and setting cell to be proper type
    if type(cell) == int or type(cell) == float:
        if (cell - 1) < (obj_map.shape[0]):
            cell = int(obj_map[cell - 1][0]), int(obj_map[cell - 1][1])
        else:
            return obj
    else:
        cell = int(cell[0]), int(cell[1])
    ##start recursive extension in the right direction
    if direction == "up":
        return recurse_extend(obj=extended_obj, cell=cell, direction="up", distance=distance)
    elif direction == "down":
        return recurse_extend(obj=extended_obj, cell=cell, direction="down", distance=distance)
    elif direction == "left":
        return recurse_extend(obj=extended_obj, cell=cell, direction="left", distance=distance)
    elif direction == "right":
        return recurse_extend(obj=extended_obj, cell=cell, direction="right", distance=distance)
    elif direction == "up-right":
        return recurse_extend(obj=extended_obj, cell=cell, direction="up-right", distance=distance)
    elif direction == "up-left":
        return recurse_extend(obj=extended_obj, cell=cell, direction="up-left", distance=distance)
    elif direction == "down-right":
        return recurse_extend(obj=extended_obj, cell=cell, direction="down-right", distance=distance)
    elif direction == "down-left":
        return recurse_extend(obj=extended_obj, cell=cell, direction="down-left", distance=distance)
    


"""
input: 
obj: numpy grid containing only object
color: int, number between 0 and 10, target color to change object to
output:
returns a numpy grid containing object of new color
"""
def change_color(obj, color):
    if obj is None or color is None:
        return np.zeros(1)
    obj = obj.astype(int)
    color = int(color)
    background_color = get_background_color(obj)
    new_arr = obj.copy()
    if not (0 <= color <= 10):
        raise ValueError("new_color must be between 0 and 10")
    # Find the indices where the object is located (i.e., where the values are not 0)
    non_zero_indices = np.argwhere(new_arr != background_color)
    # Change the values of the object to the new color
    for idx in non_zero_indices:
        new_arr[idx[0], idx[1]] = color
    return new_arr

##unlike change color, paint_obj allows for more fine grained color changing, setting color by the cell
##input: obj. numpy grid.
#color_list: list of integers, list of colors.
##returns numpy grid with cell colors changed
def paint_obj(obj, color_list):
    if obj is None or color_list:
        return np.zeros(1)
    obj = obj.astype(int)
    obj_map = percieve_obj_map(obj)
    for x in range(len(color_list)):
        if x < obj_map.shape[0]:
            obj[obj_map[x][0], obj_map[x][1]] = color_list[x]
    return obj
    


# Function to shrink an object.
#Input: object, numpy grid
# scale_factor: float
#Output: numpy grid with shrunk object located in center of array
def scale_down(obj, scale_factor):
    if obj is None or scale_factor is None:
        return np.zeros(1)
    obj = obj.astype(int)
    scale_factor = int(scale_factor)
    scene = np.ones(obj.shape, dtype=int) * get_background_color(obj)
    obj = zoom_in(obj)
    if scale_factor >= 1:
        raise ValueError("Scale factor must be less than 1 for shrinking.")
    # Use scipy's zoom function to shrink the array
    shrunk_obj = scipy.ndimage.zoom(obj, scale_factor, order=0)
    # Find the starting indices for placing the shrunk_obj in the scene
    start_x = (scene.shape[0] - shrunk_obj.shape[0]) // 2
    start_y = (scene.shape[1] - shrunk_obj.shape[1]) // 2
    # Place the shrunk object in the center of the scene
    scene[start_x:start_x + shrunk_obj.shape[0], start_y:start_y + shrunk_obj.shape[1]] = shrunk_obj
    return scene

# Function to expand an object.
#Input: object, numpy grid
# scale_factor: float
#Output: numpy grid with expanded object located in center of array
def scale_up(obj, scale_factor):
    if obj is None or scale_factor is None:
        return np.zeros(1)
    obj = obj.astype(int)
    scale_factor = int(scale_factor)
    scene = np.ones(obj.shape, dtype=int) * get_background_color(obj)
    obj = zoom_in(obj)
    if scale_factor <= 1:
        raise ValueError("Scale factor must be greater than 1 for expanding.")
    # Use scipy's zoom function to expand the array
    expanded_obj = scipy.ndimage.zoom(obj, scale_factor, order=0)  # Nearest-neighbor interpolation to keep shape features
    start_x = (scene.shape[0] - expanded_obj.shape[0]) // 2
    start_y = (scene.shape[1] - expanded_obj.shape[1]) // 2
    # Place the shrunk object in the center of the scene
    scene[start_x:start_x + expanded_obj.shape[0], start_y:start_y + expanded_obj.shape[1]] = expanded_obj
    return scene

##takes in grid and degrees and rotates the grid by the specified amount. Needs to be by a multiple of 90.
##returns the rotated grid.
def rotate_grid(grid, degrees):
    if grid is None or degrees is None:
        return np.zeros(1)
    grid = grid.astype(int)
    if degrees not in [90, 180, 270]:
        return np.zeros(1)
    if degrees == 90:
        return np.rot90(grid, k=3)  # Rotate 90 degrees clockwise (equivalent to 270 counter-clockwise)
    elif degrees == 180:
        return np.rot90(grid, k=2)  # Rotate 180 degrees
    elif degrees == 270:
        return np.rot90(grid, k=1)  # Rotate 270 degrees clockwise (equivalent to 90 counter-clockwise)


##removes part of object specified by cells in cell_list
def remove_obj_portion(obj, cell_list):
    if obj is None or cell_list is None:
        return np.zeros(1)
    obj = obj.astype(int)
    new_obj = obj.copy()
    background_color = get_background_color(obj)
    for cells in cell_list:
        if (0 < cells[0] < obj.shape[0]) and (0 < cells[1] < obj.shape[1]):
            new_obj[cells[0], cells[1]] = background_color
    return new_obj


"""
input: 
scene: numpy grid containing all objects/entire scene
obj: numpy grid containing only object
target_bounds: tuple of format (top, bottom, left, right). where all elements 
are ints representing where in the grid the object is to be placed. tuple is a target location to copy the object to.
output:
returns a numpy grid containing scene with object added. 
""" 
def place_new(scene, obj, target_bounds):
    top, bottom, left, right = map(int, target_bounds)  # Ensure bounds are integers
    scene_height, scene_width = scene.shape
    obj_height, obj_width = obj.shape
    # Ensure the left bound is less than or equal to the right bound
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    # Adjust bounds to not exceed the scene dimensions
    bottom = min(bottom, scene_height - 1)
    right = min(right, scene_width - 1)
    # Determine valid scene rows and columns for the object placement
    scene_rows = np.arange(top, bottom + 1)
    scene_cols = np.arange(left, right + 1)
    # Calculate the corresponding rows and columns in the object
    obj_rows = scene_rows - top
    obj_cols = scene_cols - left
    # Ensure the object rows and columns are within object bounds
    valid_obj_rows = obj_rows[obj_rows < obj_height]
    valid_obj_cols = obj_cols[obj_cols < obj_width]
    # Create mesh grids to map the scene and object coordinates
    scene_row_grid, scene_col_grid = np.meshgrid(valid_obj_rows + top, valid_obj_cols + left, indexing='ij')
    obj_row_grid, obj_col_grid = np.meshgrid(valid_obj_rows, valid_obj_cols, indexing='ij')
    # Assign object values to the corresponding scene positions
    scene[scene_row_grid, scene_col_grid] = obj[obj_row_grid, obj_col_grid]
    return scene


##adds completely new object. fills in an area specified by bounds. To fill in a single cell, make the top cell equal the bottom cell
##and the left cell equal the right cell
def create_new_object(grid_shape, background_color, obj_color, obj_bounds):
    if grid_shape is None or background_color is None or obj_color is None or obj_bounds is None:
        return np.zeros(1)
    top, bottom, left, right = obj_bounds
    top = int(top) 
    bottom = int(bottom)
    left = int(left)
    right = int(right)
    new_obj_grid = np.ones(grid_shape, dtype=int) * background_color
    scene_height, scene_width = new_obj_grid.shape  # Corrected to retrieve the shape dimensions
    # Check if the bounds are outside the grid
    if (bottom >= scene_height) or (right >= scene_width) or (top < 0) or (left < 0):
        return new_obj_grid
    # Ensure the bounds are valid
    top, bottom = min(top, bottom), max(top, bottom)
    left, right = min(left, right), max(left, right)
    # Case for a single point
    if (top == bottom) and (left == right):
        new_obj_grid[top, left] = obj_color
    else:
        # Vectorized assignment using slicing
        new_obj_grid[top:bottom + 1, left:right + 1] = obj_color
    return new_obj_grid


# Function to rotate the object in place about its center point
def rotate_in_place(obj, angle):
    if obj is None:
        return np.zeros(1)
    # Rotate without changing the array size, using nearest-neighbor interpolation
    rotated_obj = scipy.ndimage.rotate(obj, angle, reshape=False, order=0)
    if rotated_obj.shape != obj.shape:
        return rotated_obj
    return rotated_obj
    


##performs the proper transformation based on transformation category and corresponding objects
def perform_transformation(grid, transformation_category, params):
    obj = grid.astype(int)
    if transformation_category == "translate":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        transformed_object = translate(obj=obj, dx=dx, dy=dy)
        return transformed_object
    elif transformation_category == "flip":
        direction = params.get("direction", "lr")
        axis = params.get("axis", 1)
        transformed_object = flip(obj=obj, direction=direction, axis=axis)
        return transformed_object
    elif transformation_category == "extend":
        direction = params.get("direction", "up")
        distance = params.get("distance", 0)
        color = params.get("color", None)
        if "cell" in params:
            cell = params['cell']
            transformed_object = extend(obj=obj, cell=cell, direction=direction, distance=distance, color = color)
            return transformed_object
        elif 'number_cell' in params:
            number_cell = params.get("number_cell", 1)
            transformed_object = extend(obj=obj, cell=number_cell, direction=direction, distance=distance, color = color)
            return transformed_object
    elif transformation_category == "change_color":
        color = params.get("color", 1)
        transformed_object = change_color(obj=obj, color=color)
        return transformed_object
    # New action functions
    elif transformation_category == "zoom_in":
        transformed_object = zoom_in(obj=obj)
        return transformed_object
    elif transformation_category == "scale_up":
        scale_factor = params.get("scale_factor", 1)
        transformed_object = scale_up(obj=obj, scale_factor=scale_factor)
        return transformed_object
    elif transformation_category == "scale_down":
        scale_factor = params.get("scale_factor", 1)
        transformed_object = scale_down(obj=obj, scale_factor=scale_factor)
        return transformed_object
    elif transformation_category == "remove_obj_portion":
        cell_list = params.get("cell_list", [])
        transformed_object = remove_obj_portion(obj=obj, cell_list=cell_list)
        return transformed_object
    elif transformation_category == "rotate_in_place":
        angle = params.get("angle", 90)
        transformed_object = rotate_in_place(obj = obj, angle = angle)
    elif transformation_category == "composite":
        transformations_list = params.get("transformations_list", [])
        parameters_list = params.get("parameters_list", [])
        transformed_object = obj
        for x in range(len(transformations_list)):
            transformation_category = transformations_list[x]
            parameters = parameters_list[x]
            transformed_object = perform_transformation(grid=transformed_object, transformation_category=transformation_category, params=parameters)
        return transformed_object
    return obj



