import numpy as np


"""
iterates through grid and for every cell makes list of itself and its surrounding cells in contact. returns the list and the most occuring
color in the list. its like a convolution in that it smooths the area over and makes object detection robust to noise. 
Also used to determine if the cells surrounding a cell have already been added to an object list. 
input: grid to search in and coordinate to check the surrounding of
output: color in surrounding and cells in surrounding
"""
def get_surrounding(arr, loc, should_smooth = False):
    rows, cols = arr.shape
    i, j = loc
    surrounding = []
    ##loop gets all possible surrounding cells of a cell, along with that cell itself
    for x in range(max(0, i-1), min(rows, i+2)):
        for y in range(max(0, j-1), min(cols, j+2)):
            surrounding.append((x, y, arr[x, y]))
    ##inner function takes in surrounding array and finds the most frequently occuring non-black color. smoothing effect.
    def process_surrounding(surrounding, current_color):
        counter = {}
        return_val = 0
        for tups in surrounding:
            nums = tups[2]
            if nums != 0:
                if nums in counter:
                    counter[nums] += 1
                else:
                    counter[nums] = 1
        if counter != {}:
            key_w_max_value = max(counter, key=counter.get)
            max_value = counter[key_w_max_value]
            instances_current_color = counter[current_color]
            if max_value > instances_current_color and instances_current_color == 1:
                return_val = key_w_max_value
            else:
                return_val = current_color
        return (return_val, surrounding)
    if should_smooth:
        current_color = arr[i][j]
        return process_surrounding(surrounding=surrounding, current_color=current_color)
    return surrounding

"""
checks if the surrounding cells of the cell of interest are already in another object. If they are, return those object indices so that cell
can be added, since contiguous blocks of the same smoothed color are treated as an object. If not, a new object is created. 
input:
object info: already existing list of objects in grid
surrounding color: most dominant color of the area that the cell is in. 
surrounding cells: the cells around the cell of interest, plus cell itself.
output:
overlap_lists: returns all lists/objects that touch the current cell while being the same color. 
"""
def check_overlap(object_info, surrounding_color, surrounding_cells):
    overlap_lists = []
    ##loops through already existing objects
    for y in range(len(object_info)):
        ##checks if color of cell area matches color of object
        if surrounding_color == object_info[y][0][2]:
            ##checks if overlap exists/the object touches the current cell
            if any(x in surrounding_cells for x in object_info[y]):
                overlap_lists.append(y)
    return overlap_lists

"""
merges all lists that have any overlaps left/combines all smaller objects that are 1. in contact. 2. the same color
input: original list of objects
output: new list of combined objects
"""
def mergify(list_of_lists):
    merged_lists = []
    while list_of_lists:
        first, *rest = list_of_lists
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if any(x in first for x in r):
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2
        merged_lists.append(list(first))
        list_of_lists = rest
    return merged_lists



"""
main function that implements object level vision in grid. object level vision is an expected prior ability for the challenge. An object is being defined as blocks that are in contact while being the same color, 
with some tolerance for noise because of the smoothing.
input: grid, numpy array of grid from task with no processing. 
output: object_info, a list of all of the found objects in the grid. object represented as a list of tuples, with a tuple represented as (x, y, color)
"""
def parse_objects(grid):
    object_info = []
    ##loops through grid to build object list
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            current_cell = grid[i][j]
            ##skips if the current cell is black bc it won't be added to the object.
            if current_cell != 0:
                ##gets all of the cells that this current cell is in contact with, along with the most dominant surrounding color. that color is used instead
                ##of the color of the cell itself to make the vision noise tolerant, which is expected as a prior ability for the challenge. 
                surrounding_cells = get_surrounding(grid, (i,j))
                ##gets all the objects that contact current cell while being the same color. 
                # overlap_indices = check_overlap(object_info, surrounding_color, surrounding_cells)
                ##passing in just the color of the cell instead of the most dominant surrounding color. No smoothing. 
                overlap_indices = check_overlap(object_info=object_info, surrounding_color=current_cell, surrounding_cells=surrounding_cells)
                ##if no other objects in contact, there is no contiguous block to add to and the cell starts a new object.
                if len(overlap_indices) == 0:
                    ##new object
                    object_info.append([(i, j, current_cell)])
                else:
                    for index in overlap_indices:
                        ##adds to old object(s)
                        object_info[index].append((i, j, current_cell))
    ##combines all objects of same color in contact with each other as larger objects. 
    return mergify(object_info)

##takes in grid and returns a list containing the size of the grid and a series of numpy arrays with each isolated object
def state_decomposition_old(grid):
    state_info = []
    background_array = np.zeros(grid.shape, dtype=int)
    state_info.append(background_array)
    object_list = parse_objects(grid)
    for objects in object_list:
        object_array = np.zeros(grid.shape, dtype=int)
        for locations in objects:
            object_array[locations[0]][[locations[1]]] = int(locations[2])
        state_info.append(object_array)
    del object_list
    return state_info


