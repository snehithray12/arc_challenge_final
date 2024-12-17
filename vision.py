import numpy as np
from numpy.linalg import norm
import json
import os
from matplotlib import pyplot as plt
from utils import get_json_filenames, get_train_pairs, get_test_pairs
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, generate_binary_structure
import cv2
from vision_old import get_surrounding, state_decomposition_old
from scipy.signal import correlate2d


EVAL_FILES = get_json_filenames("ARC-AGI/data/evaluation/")
TRAIN_FILES = get_json_filenames("ARC-AGI/data/training/")


"""
isolates just the object in an array.
"""    
def isolate_object(arr, top, bottom, left, right):
    if arr.shape == (1,):
        return arr
    arr = arr.astype(int)
    new_arr = arr.copy()
    sub_array = new_arr[top:bottom+1, left:right+1]
    return sub_array


##returns a grid that is a zoomed in version of an object in the original grid
#input: numpy array of just the object
#output: numpy array of just the zoomed in object. Size of array is now the size of the object in the old grid.
def zoom_in(obj, obj_map = None):
    obj = obj.astype(int)
    if obj_map is None:
        obj_map = percieve_obj_map(obj)
    top, bottom, left, right = percieve_top_bottom_left_right(obj_map = obj_map)
    return isolate_object(obj, top, bottom, left, right)


def get_background_color(grid):
    grid = grid.astype(int)
    if 0 in grid:
        background_color = 0
    else:
        background_color = np.bincount(grid.flatten()).argmax()
    return background_color

##takes in grid and returns a list containing the size of the grid and a series of numpy arrays with each isolated object
def state_decomposition_new(grid, background_color=-1):
    state_info = []
    grid = grid.astype(int)
    if background_color <= 0:
        background_color = get_background_color(grid)
    background_array = np.ones(grid.shape, dtype=int) * background_color
    state_info.append(background_array)
    object_list = find_objects(grid, segment_gaps=False, break_down_loops=False)
    state_info = state_info + object_list
    update_background_leftovers(original=grid, state_decomposition=state_info)
    return state_info




##takes in state decomposition and superposes all the isolated object arrays to recreate the original grid
def update_background_leftovers(original, state_decomposition):
    background_array = state_decomposition[0]
    background_color = get_background_color(background_array)
    new_array = background_array.copy()
    for items in state_decomposition[1:]:
        for x in range(items.shape[0]):
            for y in range(items.shape[1]):
                if items[x][y] != background_color:
                    new_array[x][y] = items[x][y]
    for x in range(original.shape[0]):
        for y in range(original.shape[1]):
            if original[x][y] != new_array[x][y]:
                background_array[x][y] = original[x][y]


"""
get dictionary representation of a single problem, with both state and raw observation representations for all the input/output pairs. 
"""
def get_problem_setup(index, which_mode):
    if which_mode == "eval":
        filenamelist = EVAL_FILES
    elif which_mode == "train":
        filenamelist = TRAIN_FILES
    train_file = filenamelist[index]
    train_pairs = get_train_pairs(train_file)
    test_pairs = get_test_pairs(train_file)
    observed_inputs = []
    observed_outputs = []
    for train_pair in train_pairs:
        train_input = train_pair[0]
        train_output = train_pair[1]
        observed_inputs.append(train_input)
        observed_outputs.append(train_output)
    return {"observed_inputs": observed_inputs, "observed_outputs": observed_outputs, "solution_input": test_pairs[0][0], "solution_output": test_pairs[0][1]}


############OPENCV VISION FUNCTIONS
##gets the color pallete of the object
def get_unique_colors(scene, background_color=None):
    scene = scene.astype(int)
    # Flatten the scene array and use numpy's unique function to find all unique colors
    unique_colors = np.unique(scene)
    if background_color == None:
        background_color = get_background_color(scene)
    # Convert to list and return
    palette = unique_colors.tolist()
    palette.remove(background_color)
    return palette


###gets most frequent non background colors from an array
def get_dominant_color(arr, background_color = None):
    """
    Finds the dominant (most frequent) color in the array.
    Assumes non-zero values represent colors and zero is the background.
    """
    arr = arr.astype(int)
    if background_color == None:
        background_color = get_background_color(arr)
    non_background_elements = arr[arr != background_color]  # Extract non-zero elements
    if non_background_elements.size == 0:
        return None  # No colors present
    return np.bincount(non_background_elements).argmax()  # Return the most frequent color


##takes in array of an object shape and tracks the polygon properties, including corners, sides, angles
def find_polygon_properties(array):
    array = array.astype(int)
    # Convert the array to a binary image (255 for shape, 0 for background)
    if 0 in array:
        background_color = 0
    else:
        background_color = np.bincount(array.flatten()).argmax()
    # Create a binary image for the background
    binary_image = np.where(array == background_color, 0, 255).astype(np.uint8)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        ###if no contours are found, treat the whole grid as a single object and return attributes
        num_sides = 0
        num_corners = 0
        polygon_points = [(0, 0)]
        angles = 0
        area = 0
        perimeter = 0
        return num_sides, num_corners, polygon_points, angles, area, perimeter
    # Assuming the largest contour is the object of interest
    contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to a polygon (the corners of the shape)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    # Number of sides and corners
    num_sides = len(polygon)
    num_corners = len(polygon)
    # Calculate area and perimeter
    area = np.count_nonzero(array != background_color)
    perimeter = cv2.arcLength(contour, True)
    # Function to calculate the angle between three points
    def angle(pt1, pt2, pt3):
        a = np.array(pt1) - np.array(pt2)
        b = np.array(pt3) - np.array(pt2)
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    # Calculate angles for each corner
    angles = []
    num_points = len(polygon)
    for i in range(num_points):
        pt1 = polygon[i-1][0]
        pt2 = polygon[i][0]
        pt3 = polygon[(i+1) % num_points][0]
        angles.append(angle(pt1, pt2, pt3))
    # Convert polygon points to a list of tuples
    polygon_points = [(pt[0][1], pt[0][0]) for pt in polygon]
    return num_sides, num_corners, polygon_points, angles, area, perimeter


##uses opencv object detection to extract objects from grid scene
def find_objects(scene, segment_gaps = False, break_down_loops = False, should_color_clump = True, black_background = True):
    scene = scene.astype(int)
    # if a 0 is in the scene, 0 is the background color
    if black_background:
        if 0 in scene:
            background_color = 0
        else:
            ##otherwise, the most frequently occuring color is treated as the background
            background_color = np.bincount(scene.flatten()).argmax()
    else:
        background_color = np.bincount(scene.flatten()).argmax()
    # Create a binary image for the background
    binary_background = np.where(scene == background_color, 0, 255).astype(np.uint8)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    ##using cv2 contour functionality for object detection
    ##contours are abrupt changes in texture/color. can be treated as object boundary.
    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(binary_background)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        # Extract the object using the mask
        obj = cv2.bitwise_and(scene, scene, mask=mask)
        #uses naive contiguous color clumping to further break the objects down
        broken, return_val = further_break(obj)
        if broken and should_color_clump:
            for obj in return_val[1:]:
                obj[obj == 0] = background_color
                objects.append(obj)
        else:
            obj[obj == 0] = background_color
            objects.append(obj)
    ##option to extract objects from inside gaps and treat them as their own objects. Further break.
    if segment_gaps:
        new_objects_list = []
        for obj in objects:
            if not any(np.array_equal(obj, x) for x in new_objects_list):
                island_objs = segment_inside_gaps(grid = scene, obj = obj)
                if len(island_objs) > 0:
                    new_objects_list.append(obj)
                    new_objects_list += island_objs
                else:
                    new_objects_list.append(obj)
        objects = new_objects_list
    ##option to use loop detection to further break down detection objects. continuous loops are 
    ##detected and pulled out as their own objects
    if break_down_loops:
        new_objects_list = []
        for obj in objects:
            loops = find_loops(obj)
            loop_obs = object_detection_loop(scene, loops)
            if len(loop_obs) > 0:
                new_objects_list += loop_obs
            else:
                new_objects_list.append(obj)
        objects = new_objects_list
    return objects

##takes an object with some noise and removes the noise from it
def perform_smoothing(obj):
    obj = obj.astype(int)
    obj_new = obj.copy()
    for x in range(obj.shape[0]):
        for y in range(obj.shape[1]):
            if obj[x][y] != 0:
                new_color, surrounding = get_surrounding(arr=obj, loc=(x,y), should_smooth=True)
                if new_color != None:
                    obj_new[x][y] = new_color
    return obj_new

##logic: if a color smoothing happens and the object array stays the same, break up different colored sections into
##different objects. 
##otherwise, keep obj the same. 
def further_break(obj):
    obj = obj.astype(int)
    # visualize_single_object(obj)
    new_obj = perform_smoothing(obj)
    # visualize_single_object(new_obj)
    if np.array_equal(new_obj, obj):
        num_sides = find_polygon_properties(new_obj)[0]
        if num_sides == 4:
            return False, obj
        new_obj_list = state_decomposition_old(obj)
        return True, new_obj_list
    else:
        return False, obj

##gets adjacent, non diagonal neighbors of a cell in a numpy grid
def get_adjacent_neighbors(grid, row, col):
    grid = grid.astype(int)
    neighbors = []
    rows, cols = grid.shape
    if row > 0:  # Check top neighbor
        neighbors.append((row - 1, col))
    if row < rows - 1:  # Check bottom neighbor
        neighbors.append((row + 1, col))
    if col > 0:  # Check left neighbor
        neighbors.append((row, col - 1))
    if col < cols - 1:  # Check right neighbor
        neighbors.append((row, col + 1))
    return neighbors


##takes in grid of scene or object. Searches for loops of any color and returns them as separate objects.
def find_loops(grid):
    grid = grid.astype(int)
    ##gets all of the unique colors in the grid
    unique_colors = get_unique_colors(grid)
    previous_loops = []
    ##checks if a cell is in a previously closed loop, to not waste computational resources
    def check_if_in_previous_loops(found_loops, cell):
        for loops in found_loops:
            if cell in loops:
                return True
        return False
    ##finds all loops of a specific color
    def find_loop_color(grid, color):
        found_loops = []
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                ##if the cell is the color you are searching for
                if grid[x][y] == color:
                    cell = (x,y)
                    ##if the cell is already in the loop list of a previously found loop dont start searching
                    if check_if_in_previous_loops(found_loops = found_loops, cell=cell):
                        continue
                    ##if the cell is "fresh" (not seen before in a loop list) start searching
                    else:
                        ##check if a loop is found in ANY direction
                        loop_found = False
                        neighbors  = get_adjacent_neighbors(grid=grid, row=x, col=y)
                        for neighbor in neighbors:
                            ##only start search if the neighbor is the proper color
                            if grid[neighbor[0], neighbor[1]] == color:
                                loop_list = [cell]
                                loop_list_instance = loop_search_recurse(grid=grid, start_cell=cell, current_cell=neighbor, loop_list=loop_list, loop_found=loop_found, color=color)
                                #print("loop found instance ", loop_found_instance, "loop_list_instance ", loop_list)
                                if loop_list_instance[0] == loop_list_instance[-1]:
                                    found_loops.append(loop_list_instance)
                                    break
        return found_loops
    #looks for loops of each unique color in grid and returns what's found
    for colors in unique_colors:
        found_loops = find_loop_color(grid, colors)
        for items in found_loops:
            previous_loops.append(items)
    return previous_loops
        
                

##recursively does a search for a loop of a certain color, starting search from a certain 
def loop_search_recurse(grid, start_cell, current_cell, loop_list, color, loop_found):
    grid = grid.astype(int)
    ##create a new loop list for every specific path taken
    loop_list.append(current_cell)
    row, col = current_cell
    neighbors = get_adjacent_neighbors(grid, row, col)
    background_color = get_background_color(grid)
    ##if cell is an "interior cell" (it is not at one of the extrema corners and is only surrounded by cells of the same color)
    ##exit out. loop detection only happens with outermost cells
    if len(neighbors) <= 2 and all(item == color for item in neighbors):
        return loop_list
    #checking cells in neighbors.
    for cells in neighbors:
        ##check that the color of the cell equals the color we want before we search down it
        if grid[cells[0], cells[1]] == color:
            #if a cell is not visited, it is not the start cell continue searching down that path
            if cells not in loop_list and cells != start_cell:
                return loop_search_recurse(grid, start_cell, cells, loop_list, color, loop_found)
            #if cell is in loop list (already visited)
            elif cells in loop_list:
                #completed loop, reached the original starting cell. returns that loop was found and loop_list
                #checking to see that it doesnt immediately stop once start_cell is directly next to cell after search stops
                if (cells == start_cell) and (len(loop_list) > 2):
                    loop_list.append(cells)
                    return loop_list
    return loop_list

###performs object detection using loops as a determining factor
##finds bounds of loops and uses that to extract all values on/in a loop as an object
def object_detection_loop(grid, loops, background_color = None):
    grid = grid.astype(int)
    if background_color == None:
        background_color = get_background_color(grid)
    obj_list = []
    ##detect loops in obj grid
    for loop in loops:
        new_obj = np.ones(grid.shape, dtype=int) * background_color
        x_values, y_values = zip(*loop)
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        ##detect the top, bottom, left, and right rows/columns of the loop
        top = min_x
        bottom = max_x
        left = min_y
        right = max_y
        ##extract that part of the grid from original grid
        sub_grid = grid[top:bottom+1, left:right+1]
        ###adding it to the new obj created so that it's by itself
        new_obj[top:bottom+1, left:right+1] = sub_grid
        obj_list.append(new_obj)
    return obj_list


####PERCEPTION SKILLS
#takes in object grid, returns a list of tuples where every tuple is individual cells in the object
def percieve_obj_map(obj):
    obj = obj.astype(int)
    return np.argwhere(obj != get_background_color(obj))

#takes in object grid and returns a dictionary of shape attributes
def percieve_shape_info(grid):
    grid = grid.astype(int)
    num_sides, num_corners, polygon_points, angles, area, perimeter = find_polygon_properties(grid)
    return {"num_sides":num_sides, "num_corners":num_corners, "angles":angles, "area": area, "perimeter":perimeter, "corner_points":polygon_points}

#takes in an object grid and returns tuple indicating center cell of object
def percieve_object_center(obj_map):
    if len(obj_map) == 0:
        return (0, 0)
    average_point = obj_map.mean(axis=0)
    return int(average_point[0]), int(average_point[1])



#takes in object grid and returns top, bottom, left, right
def percieve_top_bottom_left_right(obj_map):
    if len(obj_map) == 0:
        return 0, 0, 0, 0
    max_x, max_y = obj_map.max(axis=0)
    min_x, min_y = obj_map.min(axis=0)
    top, bottom, left, right = min_x, max_x, min_y, max_y
    return int(top), int(bottom), int(left), int(right)

#takes in object grid and returns a list of "gap cells" in object. Returns a list of lists of tuples.
def percieve_object_gaps(grid, obj, obj_map = None):
    grid = grid.astype(int)
    obj = obj.astype(int)
    def is_in_contact(cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        # Check if the cells are adjacent horizontally, vertically, or diagonally
        return (abs(x1 - x2) == 1 and abs(y1 - y2) == 1) or \
            (x1 == x2 and abs(y1 - y2) == 1) or \
            (y1 == y2 and abs(x1 - x2) == 1)
    if obj_map is None:
        obj_map = percieve_obj_map(obj)
    top_row, bottom_row, left_column, right_column = percieve_top_bottom_left_right(obj_map = obj_map)
    background_color = get_background_color(grid)
    gap_list = []
    ###looping through bounds of an object
    for i in range(top_row, bottom_row + 1):
        for j in range(left_column, right_column + 1):
            ##value of current cell
            value = grid[i][j]
            #if the value is eligible to be considered a gap (i.e, it is a part of the background)
            #add it to gap_list
            if value == background_color:
                if len(gap_list) == 0:
                    ##if gap list is empty, add your cell to gap list
                    gap_list.append([(i, j)])
                else:
                    exit = False
                    ##look for which list in gap list to add cell to. do this based on which group it is in contact with.
                    for lists in gap_list:
                        for cells in lists:
                            if is_in_contact(cells, (i, j)):
                                lists.append((i, j))
                                exit = True
                                break
                        if exit:
                            break
                    if exit == False:
                        gap_list.append([(i, j)])
    return gap_list

#takes in full scene grid and object array and returns all of the observable features defined earlier
##in a dictionary
def percieve_object_attributes(grid, obj):
        grid = grid.astype(int)
        obj = obj.astype(int)
        obj_attributes = {}
        obj_attributes["object"] = obj
        obj_attributes["object_colors"] = get_unique_colors(obj)
        obj_attributes["object_map"] = percieve_obj_map(obj)
        shape_info = percieve_shape_info(obj)
        ##adds keys and values for num_sides, num_corners, corner_points, angles, area, perimeter 
        obj_attributes.update(shape_info)
        obj_attributes["object_center"] = percieve_object_center(obj_map=obj_attributes["object_map"])
        top, bottom, left, right = percieve_top_bottom_left_right(obj_map=obj_attributes["object_map"])
        obj_attributes["top_row"] = top
        obj_attributes["bottom_row"] = bottom
        obj_attributes["left_column"] = left
        obj_attributes["right_column"] = right
        obj_attributes["object_gaps"] = percieve_object_gaps(grid=grid, obj=obj, obj_map=obj_attributes["object_map"])
        return obj_attributes


##given an object, percieves what might be a subobject or pattern inside the object and returns it
def percieve_sub_pattern(obj):
    if obj.shape == (1,):
        return obj
    obj = obj.astype(int)
    ##zooms into object first
    obj = zoom_in(obj)
    ##get background color of zoomed in object
    background_color = get_background_color(obj)
    ##new array made, initialized with 0s in background by default.
    new_array = np.zeros(obj.shape, dtype=int)
    for x in range(obj.shape[0]):
        for y in range(obj.shape[1]):
            if obj[x][y] != background_color:
                new_array[x][y] = obj[x][y]
    return new_array


##given an object, percieves inside hole s.t it can properly treat everything inside a hole as its own object,
##which is a gap in the current vision implementation.
def segment_inside_gaps(grid, obj):
    grid = grid.astype(int)
    obj = obj.astype(int)
    ##gets the maximum bounds of the gap sections
    def get_gap_bounds(points):
        if len(points) == 0:
            return -1, -1, -1, -1
        min_x, max_x = points[0][0], points[0][0]
        min_y, max_y = points[0][1], points[0][1]
        for x, y in points:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        top = min_x
        bottom = max_x
        left = min_y
        right = max_y
        return top, bottom, left, right
    #first, percieve the gaps
    obj_gaps = percieve_object_gaps(grid, obj)
    #get background color to compare 
    background_color = get_background_color(grid)
    ##then, get bounds of gaps, top, bottom, left, right. Area to search in and segment from.
    islands = []
    for gap in obj_gaps:
        if len(gap) == 1:
            continue
        #now, run segmentation code. Any section inside gaps bounds that is "on an island", use value of surroundings
        #and proximity to top, bottom, left, or right to determine edge "island" cells.
        top, bottom, left, right = get_gap_bounds(gap)
        if top == -1:
            continue
        island_border_points = []
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                value = grid[i, j]
                isolated_vertical = False
                isolated_horizontal = False
                # Do something with value (e.g., print or process)
                ##if distance to top is equal to or smaller than distance to bottom, check that above value is blank
                if abs(top - i) <= abs(bottom - i):
                    if i != 0:
                        cell_above = grid[i - 1, j]
                        isolated_vertical = (cell_above == background_color)
                    elif i == 0:
                        cell_below = grid[i + 1, j]
                        isolated_vertical = (cell_below == background_color)
                #if distance to top is greater than distance to bottom, check that below value is background color/blank
                elif abs(top - i) > abs(bottom - i):
                    if (i != grid.shape[0] - 1):
                        cell_below = grid[i + 1, j]
                        isolated_vertical = (cell_below == background_color)
                    elif (i == grid.shape[0] - 1):
                        cell_above = grid[i - 1, j]
                        isolated_vertical = (cell_above == background_color)
                #if distance to left is smaller than distance to right, check that left most cell is blank.
                if abs(left - j) <= abs(right - j):
                    if j != 0:
                        cell_left = grid[i, j - 1]
                        isolated_horizontal = (cell_left == background_color)
                    elif j == 0:
                        cell_right = grid[i, j + 1]
                        isolated_horizontal = (cell_right == background_color)
                #if distance to left is larger than distance to right, check that right cell is blank.
                elif abs(left - j) > abs(right - j):
                    if j != grid.shape[1] - 1:
                        cell_right = grid[i, j + 1]
                        isolated_horizontal = (cell_right == background_color)
                    elif j == grid.shape[1] - 1:
                        cell_left = grid[i, j - 1]
                        isolated_horizontal = (cell_left == background_color)
                ##if isolated both horizontally and vertically AND inside the bounds of the gaps, this is an interior island border cell
                if (isolated_horizontal and isolated_vertical) and (value != background_color):
                    island_border_points.append((i,j))
        ##gets the bounds of the island border points
        top, bottom, left, right = get_gap_bounds(island_border_points)
        ##gets the interior points in the gap
        island_interior_points = []
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                value = grid[i][j]
                if value != background_color:
                    island_interior_points.append((i, j))
        island_points = list(set(island_border_points) | set(island_interior_points))
        islands.append(island_points)
    ##pull island values out and make them their own objects
    island_objects = []
    for island in islands:
        new_obj = np.ones(grid.shape, dtype=int) * background_color
        for points in island:
            ##setting values from island into new object grid
            new_obj[points[0], points[1]] = grid[points[0], points[1]]
            ##removing objects from old object
            obj[points[0], points[1]] = background_color
        island_objects.append(new_obj)
    return island_objects

##used to help with refining object permanence in agent's vision.
##takes in an object that might have chunks bitten out because it is "behind" another object in 3d.
##naive 2d vision has no way of handling that by itself, so coding that in. 
#returns complete object, what object "actually is"
def percieve_object_permanence(grid, state_list):
    grid = grid.astype(int)
    state_list_edit = state_list.copy()
    color_map = {}
    ##looping through objects, ignoring blank background 
    for state in state_list_edit:
        #getting dominant color of object
        color = get_dominant_color(state)
        ##if color is in map, overlay it with object of that color that already exists
        if color in color_map:
            color_map[color] = color_map[color] + state
            color_map[color] = np.where(color_map[color] != 0, color_map[color], state)
        else:
            color_map[color] = state
    ##gets dominant color of object and use it to get the new version with all the fragments added
    new_obj_list = []
    for color_key in color_map:
        new_obj_array = color_map[color_key]
        new_obj_array = continue_pattern_obj(grid = grid, obj = new_obj_array)
        new_obj_list.append(new_obj_array)
    return new_obj_list

###object detection with all cells of one color treated as an object
def percieve_obj_as_color_map(grid):
    grid = grid.astype(int)
    background_color = get_background_color(grid)
    object_colors = get_unique_colors(grid, background_color)
    obj_list = []
    for color in object_colors:
        result = np.full_like(grid, background_color)
        result[grid == color] = color
        obj_list.append(result)
    return obj_list
    

##takes in overall grid and object that may have gaps in it and returns the object with gaps properly 
##filled in via pattern continuation logic
def continue_pattern_obj(grid, obj):
    grid = grid.astype(int)
    obj = obj.astype(int)
    background_color = get_background_color(obj)
    color_palette = get_unique_colors(obj, background_color)
    if color_palette is None:
        return obj
    ##getting area to search in
    non_background_indices = np.argwhere(obj != background_color)
    if len(non_background_indices) == 0:
        return obj
    top = non_background_indices[:, 0].min()
    bottom = non_background_indices[:, 0].max()
    left = non_background_indices[:, 1].min()
    right = non_background_indices[:, 1].max()
    # search for maximally informative row
    maximally_informative_rows = []
    for x in range(obj.shape[0]):
        if (x >= top) and (x <= bottom):
            row = obj[x]
            information_gain_score = 0
            for y in range(len(row)):
                if (y >= left) and (y <= right):
                    cell = row[y]
                    if ((cell in color_palette) or (cell == 0)) and ((grid[x][y] in color_palette) or (grid[x][y] == 0)):
                        information_gain_score += 1
            if (information_gain_score - 1) == (right - left):
                array_exists = any(np.array_equal(arr, row) for arr in maximally_informative_rows)
                if array_exists is not True:
                    maximally_informative_rows.append(row)
    ##searching for maximally informative columns
    maximally_informative_columns = []
    for col in range(obj.shape[1]):
        if (col >= left) and (col <= right):
            column = obj[:, col]
            information_gain_score = 0
            for y in range(len(column)):
                if (y >= top) and (y <= bottom):
                    cell = column[x]
                    if ((cell in color_palette) or (cell == 0)) and ((grid[y][col] in color_palette) or (grid[y][col] == 0)):
                        information_gain_score += 1
            if (information_gain_score - 1) == (bottom - top):
                array_exists = any(np.array_equal(arr, column) for arr in maximally_informative_columns)
                if array_exists is not True:
                    maximally_informative_columns.append(column)
    ##do one fill in pass across the rows using the maximally informative rows
    ##idea is that some row or column will be fully exposed and give an idea about the pattern
    ##use similarity between fully exposed informative rows/columns and partially occluded rows/columns 
    ## to determine what to fill in to covered areas
    if len(maximally_informative_rows) > 0:
        current_add_counter = 0
        for x in range(obj.shape[0]):
            if (x >= top) and (x <= bottom):
                row = obj[x]
                array_exists = any(np.array_equal(arr, row) for arr in maximally_informative_rows)
                if array_exists:
                    continue
                else:
                    match_scores_row = [0] * len(maximally_informative_rows)
                    for y in range(len(row)):
                        cell = row[y]
                        if ((cell in color_palette) or (cell == 0)) and ((grid[x][y] in color_palette) or (grid[x][y] == 0)):
                            for z in range(len(maximally_informative_rows)):
                                key_row = maximally_informative_rows[z]
                                if row[y] == key_row[y]:
                                    match_scores_row[z] += 1
                highest_match_score = max(match_scores_row)
                if highest_match_score > 0:
                    most_similar_index = match_scores_row.index(highest_match_score)
                    most_similar_row = maximally_informative_rows[most_similar_index]
                    ##filling in obj with most similar row
                    obj[x] = most_similar_row
                else:
                    ###if the occluded rows are completely occluded, add by trying to maintain
                    ##some percieved pattern. Assumes that key rows are contiguous and form some repeating pattern
                    current_add_index = current_add_counter % len(maximally_informative_rows)
                    row_to_add = maximally_informative_rows[current_add_index]
                    obj[x] = row_to_add
                current_add_counter += 1
    ##do one fill in pass across the columns using the maximally informative columns
    if len(maximally_informative_columns) > 0:
        current_add_counter = 0
        for col in range(obj.shape[1]):
            if (col >= left) and (col <= right):
                column = obj[:, col]
                array_exists = any(np.array_equal(arr, column) for arr in maximally_informative_columns)
                if array_exists:
                    continue
                else:
                    match_scores_columns = [0] * len(maximally_informative_columns)
                    for y in range(len(column)):
                        cell = column[y]
                        if ((cell in color_palette) or (cell == 0)) and ((grid[y][col] in color_palette) or (grid[y][col] == 0)):
                            for z in range(len(maximally_informative_columns)):
                                key_column = maximally_informative_columns[z]
                                if column[y] == key_column[y]:
                                    match_scores_columns[z] += 1
                highest_match_score = max(match_scores_columns)
                if highest_match_score > 0:
                    most_similar_index = match_scores_columns.index(highest_match_score)
                    most_similar_column = maximally_informative_columns[most_similar_index]
                    ##filling in obj with most similar row
                    obj[:, col] = most_similar_column
                else:
                    ###if the occluded columns are completely occluded, add by trying to maintain
                    ##some percieved pattern. Assumes that key cols are contiguous and form some repeating pattern
                    current_add_index = current_add_counter % len(maximally_informative_columns)
                    column_to_add = maximally_informative_columns[current_add_index]
                    obj[:, col] = column_to_add
                current_add_counter += 1
    return obj


##takes in landmark and the object cell that we are dealing with and calculates whether the landmark
##is above/below/level and left/right/level with the object cell
def determine_direction_landmark(landmark, object_cell, up_down_only = False, left_right_only = False):
    first_str = ''
    second_str = ''
    ##if x value is bigger, landmark is "below"
    if landmark[0] > object_cell[0]:
        first_str = 'down'
    ##if x value is equal, level
    elif landmark[0] == object_cell[0]:
        first_str = ''
    ##if x value is less, landmark is "above"
    elif landmark[0] < object_cell[0]:
        first_str = 'up'
    ##if y value is greater, landmark is to the right
    if landmark[1] > object_cell[1]:
        second_str = 'right'
    ##if y value is equal, landmark is level
    elif landmark[1] == object_cell[1]:
        second_str = ''
    ##if y value is less, it is to the right
    elif landmark[1] < object_cell[1]:
        second_str = 'left'
    if up_down_only:
        return first_str
    if left_right_only:
        return second_str
    if len(first_str) > 0 and len(second_str) > 0:
        return '{}-{}'.format(first_str, second_str)
    elif len(first_str) > 0 and len(second_str) == 0:
        return first_str
    elif len(first_str) == 0 and len(second_str) > 0:
        return second_str
    else:
        return 'up'


###caller function that performs different types of vision based on type parameter outside ARC_object constructor
def perform_object_detection_type(grid, type_vision = 1):
    obj_list = []
    if type_vision == 1:
        obj_list = find_objects(grid, should_color_clump = True)
    elif type_vision == 2:
        obj_list = state_decomposition_old(grid = grid)[1:]
    elif type_vision == 3:
        obj_list = state_decomposition_old(grid = grid)[1:]
        obj_list = percieve_object_permanence(grid = grid, state_list = obj_list)
    elif type_vision == 4:
        obj_list = percieve_obj_as_color_map(grid = grid)
    elif type_vision == 5:
        obj_list = find_objects(grid, should_color_clump = False)
    elif type_vision == 6:
        obj_list = find_objects(grid, black_background = False)   
    return obj_list