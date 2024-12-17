from utils import *
from vision import *
from vision_old import state_decomposition_old
from actions import *
from collections import defaultdict
import numpy as np

###we want object number to be left to right, top to bottom in the grid, like words on a page. 
###updates object numbers for GridObjects in a Grid object to enforce that property
def update_object_numbers(grid_obj):
    object_list = grid_obj.get_object_list()
    obj_and_loc_list = []
    for x in range(len(object_list)):
        obj = object_list[x]
        ##if a blank background array does show up for some reason, do not increment its count.
        if len(np.unique(obj.get_object_grid())) > 1:
            obj_and_loc_list.append((obj, obj.get_object_center()))
    sorted_obj_and_loc_list = sorted(obj_and_loc_list, key=lambda item: (item[1][0], item[1][1]))
    for x in range(len(sorted_obj_and_loc_list)):
        obj, loc = sorted_obj_and_loc_list[x]
        obj.set_object_number(x + 1)

class GridObject:
    def __init__(self, priority, context_grid_obj, object_grid):
        self._priority = priority
        self._context_grid_obj = context_grid_obj
        self._context_grid = context_grid_obj.get_grid()
        self._object_grid = object_grid
        self._object_number = 0
        self._zoomed_object_grid = None
        self._object_colors = []
        self._dominant_color = 0
        self._object_map = []
        self._num_sides = 0
        self._num_corners = 0
        self._angles = []
        self._area = 0
        self._corner_points = []
        self._object_center = (0, 0)
        self._top_row = 0
        self._bottom_row = 0
        self._left_column = 0
        self._right_column = 0
        self._object_gaps = []
        self._sub_pattern = None
        if object_grid is not None:
            self._update_attributes()

    def __eq__(self, other):
        if isinstance(other, GridObject):
            return np.array_equal(other.get_object_grid(), self._object_grid) and np.array_equal(other.get_context_grid(), self._context_grid)
        return False
    


    ##updates all of the attributes given the object grid object
    def _update_attributes(self):
        obj_attributes = percieve_object_attributes(grid=self._context_grid, obj=self._object_grid)
        self._object_colors = obj_attributes['object_colors']
        self._dominant_color = get_dominant_color(arr=self._object_grid)
        self._object_map = obj_attributes['object_map']
        shape_info_dict = percieve_shape_info(grid=self._object_grid)
        self._num_sides = shape_info_dict['num_sides']
        self._num_corners = shape_info_dict['num_corners']
        self._angles = shape_info_dict['angles']
        self._area = shape_info_dict['area']
        self._corner_points = shape_info_dict['corner_points']
        self._object_center = percieve_object_center(obj_map = self._object_map)
        self._zoomed_object_grid = zoom_in(obj = self._object_grid, obj_map = self._object_map)
        self._top_row = obj_attributes["top_row"]
        self._bottom_row = obj_attributes["bottom_row"]
        self._left_column = obj_attributes["left_column"]
        self._right_column = obj_attributes["right_column"]
        if self._num_sides == 4:
            self._object_center = ((self._top_row + self._bottom_row)/2, (self._left_column + self._right_column)/2)
        self._object_gaps = percieve_object_gaps(grid=self._context_grid, obj=self._object_grid, obj_map = self._object_map)
        self._sub_pattern = percieve_sub_pattern(self._object_grid)

    ##if the object grid is ever transformed, method for updating grid and object values
    def set_object_grid(self, grid):
        self._object_grid = grid
        self._update_attributes()

    ##gets priority value of object
    def get_priority(self):
        return self._priority

    
    def get_context_grid(self):
        return self._context_grid
    
    def get_context_grid_obj(self):
        return self._context_grid_obj

    def get_object_grid(self):
        return self._object_grid

    def get_object_colors(self):
        self._object_colors.sort()
        return self._object_colors
    
    def get_dominant_color(self):
        return self._dominant_color

    def get_object_map(self):
        return self._object_map

    def get_num_sides(self):
        return self._num_sides

    def get_num_corners(self):
        return self._num_corners

    def get_angles(self):
        self._angles.sort()
        return self._angles

    def get_area(self):
        return self._area

    def get_corner_points(self):
        return self._corner_points

    def get_object_center(self):
        return self._object_center

    def get_top_row(self):
        return self._top_row

    def get_bottom_row(self):
        return self._bottom_row

    def get_left_column(self):
        return self._left_column

    def get_right_column(self):
        return self._right_column

    def get_object_gaps(self):
        return self._object_gaps
    
    def get_sub_pattern(self):
        return self._sub_pattern

    def get_shape(self):
        return self._context_grid_obj.get_shape()
    
    ##numerical representation of where object occurs in grid, left to right, top to bottom. set by grid object that contains
    ##this GridObject
    def set_object_number(self, number):
        self._object_number = number

    
    def get_object_number(self):
        return self._object_number

    
    ##gets single cell position of the cell in the place indicated by the position number
    ##returns single tuple
    def find_cell_pos(self, position):
        return get_cell_position(self, position)
    
    ##returns cell/cells that equal or dont equal a specified color. returns a list of global locations
    ##input: color (int): color that cells should either equal or not equal
    ##equals (bool): value that indicates whether we are looking for cells that EQUAL or DONT equal a color.
    ##if equals, is true, looking for cells that equal a color. if equals is false, looking for cells that DO NOT equal a color.
    ##returns a list of global cells
    def find_cell_color(self, color, equals):
        if color is None or equals is None:
            return []
        grid = self._zoomed_object_grid
        global_cell_list = []
        if equals:
            cells = np.argwhere(grid == color)
        else:
            cells = np.argwhere(grid != color)
        width = grid.shape[1]
        for cell in cells:
            cell_number = (cell[0] * (width)) + cell[1] + 1
            global_cell_pos = get_cell_position(self, cell_number)
            global_cell_list.append(global_cell_pos)
        return global_cell_list
        
    ##finds cells within object that meet certain global bounds specifications. Ex. find cells in object on top row
    ##OR cells in object in row that is BELOW the bottom of another object.
    ##input:
    #index: which index of cell to compare, 0 or 1. 
    #comparator: string, can be >,<, or ==. what comparison you want to do with the bounds value.
    #value: int. the value to check against.
    def find_cell_bounds(self, index, comparator, value):
        global_cell_list = []
        for cells in self._object_map:
            if index == 0:
                if comparator == ">":
                    if cells[0] > value:
                        global_cell_list.append(cells)
                elif comparator == "<":
                    if cells[0] < value:
                        global_cell_list.append(cells)
                elif comparator == "==":
                    if cells[0] == value:
                        global_cell_list.append(cells)
            elif index == 1:
                if comparator == ">":
                    if cells[1] > value:
                        global_cell_list.append(cells)
                elif comparator == "<":
                    if cells[1] < value:
                        global_cell_list.append(cells)
                elif comparator == "==":
                    if cells[1] == value:
                        global_cell_list.append(cells)
        return global_cell_list


class Grid:
    def __init__(self, grid, perception_mode = 5):
        self._grid = grid
        self._background_array = np.zeros(self._grid.shape)
        self._background_color = get_background_color(self._grid)
        self._information_content = np.count_nonzero(self._grid != self._background_color)
        self._perception_mode = perception_mode
        self._shape = grid.shape
        if self._shape[0] > 1:
            x = self._shape[0]/2
        else:
            x = 0
        if self._shape[1] > 1:
            y = self._shape[1]/2
        else:
            y = 0
        self._center = (x, y)
        self._object_list = []
        self._color_occurrences = {}
        self._all_object_sizes = []
        self._all_object_centers = []
        self._top_most_row = 0
        self._bottom_most_row = 0
        self._left_most_column = 0
        self._right_most_column = 0
        if grid is not None:
            self._fill_object_list()
            self._update_values()
    

    def __eq__(self, other):
        if isinstance(other, Grid):
            return np.array_equal(other.get_grid(), self._grid)
        return False

    def get_grid(self):
        self._update_grid()
        return self._grid
    
    def get_background_array(self):
        return self._background_array

    def get_background_color(self):
        return self._background_color

    def get_shape(self):
        return self._shape

    def get_center(self):
        return self._center

    def get_object_list(self):
        return self._object_list

    def get_color_occurrences(self):
        return self._color_occurrences

    def get_all_object_sizes(self):
        return self._all_object_sizes
    
    def get_largest_object(self):
        index = max(self._all_object_sizes)
        for obj in self._object_list:
            obj_size = obj.get_area()
            if obj_size == index:
                return obj

    def get_smallest_object(self):
        index = min(self._all_object_sizes)
        for obj in self._object_list:
            obj_size = obj.get_area()
            if obj_size == index:
                return obj
            

    def get_all_object_centers(self):
        return self._all_object_centers
    
    
    def get_top_most_obj_row(self):
        return self._top_most_row
    

    def get_bottom_most_obj_row(self):
        return self._bottom_most_row
    

    def get_right_most_obj_col(self):
        return self._right_most_column
    
    def get_left_most_obj_col(self):
        return self._left_most_column
    
    def get_information_content(self):
        return self._information_content
    
    ##determines if an obj is already in the object list for the Grid
    def is_in_object_list(self, obj):
        obj_grid = obj.get_object_grid()
        match_value = np.sum(obj_grid ** 2)
        correlation = correlate2d(self._grid, obj_grid, mode='valid')
        if np.any(correlation == match_value):
            return True
        for current_obj in self._object_list:
            current_obj_grid = current_obj.get_object_grid()
            return np.array_equal(obj_grid, current_obj_grid)
        return False

    
    ##percieves objects in grid and fills object list
    def _fill_object_list(self):
        if self._perception_mode == 1:
            obj_list = find_objects(self._grid, should_color_clump = True)
        elif self._perception_mode == 2:
            obj_list = state_decomposition_old(grid = self._grid)[1:]
        elif self._perception_mode == 3:
            obj_list = state_decomposition_old(grid = self._grid)[1:]
            obj_list = percieve_object_permanence(grid = self._grid, state_list = obj_list)
        elif self._perception_mode == 4:
            obj_list = percieve_obj_as_color_map(grid = self._grid)
        elif self._perception_mode == 5:
            obj_list = find_objects(self._grid, should_color_clump = False)
        elif self._perception_mode == 6:
            obj_list = find_objects(self._grid, black_background = False)           

        for x in range(len(obj_list)):
            obj = obj_list[x]
            grid_object = GridObject(priority = 1, context_grid_obj = self, object_grid = obj)
            ##checking if object is not blank and is the correct shape
            if (len(np.unique(grid_object.get_object_grid())) > 1) and (grid_object.get_object_grid().shape == self._background_array.shape):
                self._object_list.append(grid_object)
            else:
                del grid_object
    
    ##takes parts of grid that werent percieved as being individual objects and adds them to the background array
    def _update_object_background(self):
        self._background_array = np.ones(self._grid.shape, dtype=int) * self._background_color
        new_array = self._background_array.copy()
        for grid_obj in self._object_list:
            items = grid_obj.get_object_grid()
            if items.shape == (1,):
                continue
            for x in range(items.shape[0]):
                for y in range(items.shape[1]):
                    if items[x][y] != self._background_color:
                        new_array[x][y] = items[x][y]
        for x in range(self._grid.shape[0]):
            for y in range(self._grid.shape[1]):
                if self._grid[x][y] != new_array[x][y]:
                    self._background_array[x][y] = self._grid[x][y]


    ##runs an update on the grid with the objects in object list
    ##composes all objects in object_list into a single output grid
    def _update_grid(self):
        self._update_object_background()
        self._sort_by_priority()
        background_color = self._background_color
        new_array = self._background_array.copy()
        for items in self._object_list:
            items_grid = items.get_object_grid()
            new_array = np.where(items_grid != background_color, items_grid, new_array)
        self._grid = new_array

    ##updates size list based on sizes of objects in object list
    def _update_object_sizes(self):
        self._all_object_sizes = []
        for x in range(len(self._object_list)):
            obj = self._object_list[x]
            self._all_object_sizes.append(obj.get_area())
    
    ##reads object list and updates dictionary with color frequencies
    def _update_color_occurrences(self):
        for obj in self._object_list:
            obj_colors = obj.get_object_colors()
            for color in obj_colors:
                if color in self._color_occurrences:
                    self._color_occurrences[color] += 1
                else:
                    self._color_occurrences[color] = 1
    
    ##reads object list and updates list of centers with 3object centers
    def _update_object_centers(self):
        self._all_object_centers = []
        for x in range(len(self._object_list)):
            obj = self._object_list[x]
            self._all_object_centers.append(obj.get_object_center())
    

    ###sets extrema values of grid (i.e, bottommost/topmost rows where objects exist)
    def _update_grid_extrema(self):
        min_x = math.inf
        max_x = -1
        min_y = math.inf
        max_y = -1
        if len(self._object_list) == 0:
            self._top_most_row = 0
            self._bottom_most_row = self._shape[0] - 1
            self._left_most_column = 0
            self._right_most_column = self._shape[1] - 1
            return        
        for obj in self._object_list:
            top = obj.get_top_row()
            bottom = obj.get_bottom_row()
            left = obj.get_left_column()
            right = obj.get_right_column()
            if top < min_x:
                min_x = top
            if bottom > max_x:
                max_x = bottom
            if left < min_y:
                min_y = left
            if right > max_y:
                max_y = right
        self._top_most_row = min_x
        self._bottom_most_row = max_x
        self._left_most_column = min_y
        self._right_most_column = max_y
    
    
    ##updates all grid specific fields at once
    def _update_values(self):
        self._update_object_sizes()
        self._update_color_occurrences()
        self._update_object_centers()
        self._update_grid()
        self._update_grid_extrema()

    
    ##sort objects in list by priority
    def _sort_by_priority(self):
        self._object_list.sort(key=lambda x: x.get_priority(), reverse=True)

    #takes in object_grid numpy array, adds it to grid array and fills object list
    #object grid added must be a transformed version of some preexisting transformed grid
    def add_obj(self, obj):
        ##if the object being added is not blank, is the proper shape, and is not a duplicate
        if (self.is_in_object_list(obj) is False) and (len(np.unique(obj.get_object_grid())) > 1) and (obj.get_object_grid().shape == self._background_array.shape):
            self._object_list.append(obj)
            self._update_values()
            update_object_numbers(self)


    ###adds a grid as an object to the specified location
    def add_grid_as_obj(self, grid, bounds):
        height_0, height_1, width_0, width_1 = bounds
        if ((height_1 - height_0) != grid.shape[0]) and ((width_1 - width_0) != grid.shape[1]):
            AssertionError("Please pass in the correct bounds")
        self._grid[height_0:height_1, width_0:width_1] = grid




    ##takes in an object by index or object itself, removes it from object list,
    ## and deletes the object itself
    def delete_object(self, obj=None):
        if obj != None:
            for x in range(len(self._object_list)):
                obj_loop = self._object_list[x]
                object_array_loop = obj_loop.get_object_grid()
                if np.array_equal(object_array_loop, obj.get_object_grid()):
                    self._object_list.remove(obj_loop)
        del obj
        self._update_values()
        update_object_numbers(self)
    

    ##takes in query and returns object(s) that meets the specifications
    def find_obj(self, query_attributes, obj_of_interest = None):
        obj_list = []
        ##running an update on object numbers before a query to make sure everything is up to date
        update_object_numbers(self)
        for obj_index in range(len(self._object_list)):
            obj = self._object_list[obj_index]
            if np.array_equal(obj.get_object_grid(), obj_of_interest.get_object_grid()):
                continue
            key = query_attributes[0]
            relation = query_attributes[1]
            value = query_attributes[2]
            ##checking value properly based on key value, data type of value, and type of relation
            if key == 'object_number':
                if relation == 'equals':
                    if obj.get_object_number() == int(value):
                        obj_list.append(obj)
                        continue     
            elif (key == 'object_colors') or (key == 'object_color'):
                if relation == 'equals':
                    if obj.get_dominant_color() == int(value):
                        obj_list.append(obj)
                        continue
            elif key == 'num_sides':
                if relation == 'equals':
                    if obj.get_num_sides() == int(value):
                        obj_list.append(obj)
                        continue
            elif key == 'area':
                if relation == 'equals':
                    if obj.get_area() == value:
                        obj_list.append(obj)
                        continue
                elif relation == 'greater':
                    if obj.get_area() > value:
                        obj_list.append(obj)
                        continue
                elif relation == 'less':
                    if obj.get_area() < value:
                        obj_list.append(obj)
                        continue  
            elif key == 'object_center':
                if relation == 'equals':
                    if obj.get_object_center() == value:
                        obj_list.append(obj)
                        continue
            elif key == 'object_center[0]':
                if relation == 'equals':
                    if obj.get_object_center()[0] == value:
                        obj_list.append(obj)
                        continue
            elif key == 'object_center[1]':
                if relation == 'equals':
                    if obj.get_object_center()[1] == value:
                        obj_list.append(obj)
                        continue
            elif key == 'top_row':
                if (type(value) == int) or (type(value) == float):
                    if relation == 'equals':
                        if (obj.get_top_row() == value):
                            obj_list.append(obj)
                            continue
                    elif relation == 'greater':
                        if (obj.get_top_row() > value):
                            obj_list.append(obj)
                            continue
                    elif relation == "less":
                        if (obj.get_top_row() < value):
                            continue   
            elif key == 'bottom_row':
                if (type(value) == int) or (type(value) == float):
                    if relation == 'equals':
                        if (obj.get_bottom_row() == value):
                            obj_list.append(obj)
                            continue
                    elif relation == 'greater':
                        if (obj.get_bottom_row() > value):
                            obj_list.append(obj)
                            continue
                    elif relation == "less":
                        if (obj.get_bottom_row() < value):
                            obj_list.append(obj)
                            continue   
            elif key == 'left_column':
                if (type(value) == int) or (type(value) == float):
                    if relation == 'equals':
                         if (obj.get_left_column() == value):
                             obj_list.append(obj)
                             continue
                    elif relation == 'greater':
                        if (obj.get_left_column() > value):
                            obj_list.append(obj)
                            continue
                    elif relation == "less":
                        if (obj.get_left_column() < value):
                            obj_list.append(obj)
                            continue 
            elif key == 'right_column':
                if (type(value) == int) or (type(value) == float):
                    if relation == 'equals':
                         if (obj.get_right_column() == value):
                             obj_list.append(obj)
                             continue
                    elif relation == 'greater':
                        if (obj.get_right_column() > value):
                            obj_list.append(obj)
                            continue
                    elif relation == "less":
                        if (obj.get_right_column() < value)   :
                            obj_list.append(obj)
                            continue
        if obj_of_interest is not None:
            distance_list = []
            for obj in obj_list:
                distance = math.dist(obj.get_object_center(), obj_of_interest.get_object_center())
                ###if distance is not 0 (which means object is itself), add it to obj list we return
                if distance != 0:
                    distance_list.append((distance, obj))
            sorted_distance = sorted(distance_list, key=lambda x: x[0])
            obj_list = []
            for tup in sorted_distance:
                distance, obj = tup
                obj_list.append(obj)
            return obj_list
        return obj_list

    
