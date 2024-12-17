def create_object_centric_same_size_template_no_code(input_list, input_objects_list, output_list, output_objects_list, possible_actions):
    pairs = '\n'.join([
    f'Pair {i+1}:\nInput:\n {inp}\nObjects in input:\n' +
    '\n'.join([str(array) for array in objs]) + 
    f'\nOutput:\n {out}\nObjects in output:\n' +
    '\n'.join([str(array) for array in objs2]) + 
    '\nState in words what the rule seems to be for this pair.'
    for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, output_list, output_objects_list))
])
    code_template = """
    Your task is to describe the rule that transforms each input grid into the output grid. Each grid contains "objects" which can be various sizes, shapes, and colors.
    The rule is a series of transformations on objects in the input to create the output.
    Note: there exists a single rule that works for ALL of the input-output rules.
    The pattern may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which
    shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or
    repeating a pattern for a fixed number of time.
    There are other concepts that may be relevant.
    - Lines, rectangular shapes
    - Symmetries rotations, translations.
    - Shape upscaling or downscaling, elastic distortions.
    - Containing / being contained / being inside or outside of a perimeter.
    - Drawing lines, connecting points, orthogonal projections.
    - Copying, repeating objects.
    Possible actions that occur on an object: 
    translate, flip about some axis, rotate in place, change color, extend (make an object longer by adding cells in some direction), removing part of an object, keeping a certain object unchanged, and adding a brand new object (that is not a transformed 
    version of an existing object). 

    Here is a hint as for possible actions that may be relevant in this rule. This is based on a heuristic. It is not guaranteed to be correct, and is also
    an upper bound. Not all of the actions have to be involved: {}

    Pairs:
    {}
    State what seems to be the overall rule that applies to all input-output pairs.


    Final rule: ###have final rule written here with the label final rule. I will use the label to parse your response.
    """.format(possible_actions, pairs)
    return code_template


def create_rule_implementation_template(rule_description, input_grids, output_grids):
    # Create formatted string for each pair of input and output grids
    pairs = '\n'.join(
        [f"Pair {i+1}:\nInput Grid:\n{inp}\nOutput Grid:\n{out}"
         for i, (inp, out) in enumerate(zip(input_grids, output_grids))]
    )
    
    # Create the full prompt template
    prompt_template = """
Your task is to implement a transformation function that applies a consistent rule to turn input grids into output grids.
The rule, described below, should be followed exactly to produce the correct transformations for all provided examples.

Rule to implement:
{rule}

Here are the input-output pairs:

{pairs}

Fill in the function below to apply the rule:

def transform_grid(input_grid):
    # Your implementation here based on the rule
    return output_grid
""".format(rule=rule_description, pairs=pairs)

    return prompt_template

def create_object_centric_rule_implementation_template(rule_description, input_grids, output_grids):
    # Create formatted string for each pair of input and output grids
    pairs = '\n'.join(
        [f"Pair {i+1}:\nInput Grid:\n{inp}\nOutput Grid:\n{out}"
         for i, (inp, out) in enumerate(zip(input_grids, output_grids))]
    )
    
    # Create the full prompt template
    prompt_template = """
Your task is to give me code for a transformation function that applies a consistent rule to turn input grids into output grids.
The rule, described below, should be followed exactly to produce the correct transformations for all provided examples.

Rule to implement:
{rule}

Here are the input-output pairs:

{pairs}

The above rule will be implemented in code that follows a very specific format. The rule will be converted to a series of

if <object property>, then <transformation> statements. 
An example rule: if object is a red square, move it down towards the blue object.
If an object is blue, keep as is.

Your task is to give me: the object checkers (boolean expressions that check for an object property), transformation functions (transformations on object 
that occur if the object meets some condition), and add new object function (function that potentially adds objects 'from scratch'. may or may not exist) if the rule
calls for adding brand new objects that are not transformations of existing objects. Here is information about code that you will call to implement the answers.
Elementary action functions. Use these to implement the transformations. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
rotate_in_place
    rotates an object about itself while keeping position the same
    parameters:
        angle: int. 90, 180, or 270
create_new_object
    creates a new arbitrary, rectangular shaped object of single color. Shapes of object arrays must be same as shape of overall grid.
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        add_obj(obj):
        takes in GridObject instance and adds it to Grid object list
        input: obj. A GridObject instance 
        returns nothing.

        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
        get_object_number() gets the object number. objects are numbered left to right, top to bottom. I.E, topmost leftmost object is object 1, and so on.
    methods:
        set_object_grid(grid)
        used when updating object grid with new transformed version. takes in 2d numpy grid representing a transformed object grid
        and updates all GridObject attributes accordingly
        input: 2d numpy array of object against background. returns nothing

        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.
Generic helper functions (not part of any class)
determine_direction_landmark(landmark, object_cell)
takes in landmark (a tuple, (x,y)) and object_cell (a tuple, (x,y)) and determines which direction the landmark is with respect to
the object cell. returns a string indicating direction.

The first step is to come up with the correct object checkers. These will be filters that check for some object property 
to determine what should happen to the object.
Once an object checker has been made, the next step is to determine what should happen to objects that fit 
the checker.
The transformation will almost always be with respect to another object or grid level location.
For example, a translation will often be to move an object towards another object. An extension will often be towards another object or towards some part of the grid.
Color change will often be to the color of another object. Flipping will often be with respect to another object, and so on.
1. How many object-checker transformation pairs are there for this rule?
For each one, perform the following tasks. 
    a. Give me code for the object checker. Set it equal to a variable called obj_checker_n, with n being the number of the object checker.
    Example:
    if the overall rule is "change the color of the largest object to the color of the object closest to it", an object checker would be:
    
    obj_checker_1 = obj.get_area() == max(input_grid_obj.get_all_object_sizes())

    Other examples: 
        obj_checker_1 = (2 in obj.get_object_colors())
        obj_checker_2 = (4 in obj.get_object_colors())

    if the rule requires the checker to be with respect to another object, like with this one, "Extend the object that's aligned with the largest object
    in the grid until it touches the largest object," that would be done as follows:
    query_1 = [("area", "equals", max(input_grid_obj.get_all_object_sizes()))]
    reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
    obj_checker_1 = obj.get_object_center()[0] == reference_obj_1.get_object_center()[0]

        **object checkers must be formatted exactly like the examples above in one line

    b. For the transformation block that corresponds to the if-statement for this object checker, write a function that performs this transformation/series of transformations as some combination of elementary 
    actions. Function requirements. Must take in GridObject instance of the object we are on, the Grid instance for the input scene 
    as a whole, and the Grid instance for the output scene as a whole as the only parameters. 
    transformation_n(Grid input_grid_obj, Grid output_grid_obj, GridObject obj) <- the method signature must look like this, with n being the same number as the n in the corresponding object checker.
    Must perform transformations and update obj using set_object_grid(grid) and/or create new objects for transformed copies of an object and then 
    add them to output_grid_obj using add_obj(obj). Returns output_grid_obj with transformed GridObjects in it.
    You may use any combinations of elementary action functions as needed. Specific requirements are that other objects that are used when determining how to perform an action should be assigned to a variable called
    reference_obj_n, with n being a number. Any locations that the object is moving with respect to in any way should be assigned to a variable called landmark_n, with n being a number.
    Examples
    def transformation_1(input_grid_obj, output_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        ##adding original to output
        output_grid_obj.add_obj(obj)
        ##adding transformed object to output
        flip_params_1 = {{}}
        query_1 = [("object_colors", "equals", 6)]
        reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
        landmark_1 = reference_obj_1.get_object_center()
        flip_params_1["direction"] = "lr"
        flip_params_1["axis"] = landmark_1[1]
        transformed_obj_grid_1 = perform_transformation(grid = obj_grid, transformation_category="flip", params = flip_params_1) ##returns numpy array
        transformed_obj_1 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid=obj_grid_1) ##use numpy array to create GridObject
        output_grid_obj.add_obj(transformed_obj_1) #add GridObject to Grid 
        ##transform transformed object, add it to the grid
        transformed_obj_grid_2 = perform_transformation(grid = transformed_obj_grid_1, transformation_category = "flip", params = flip_params_1)
        transformed_obj_2 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid_2)
        output_grid_obj.add_obj(transformed_obj_2)
        return output_grid_obj

    def transformation_2(input_grid_obj, output_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        translate_params_1 = {{}}
        query_1 = [("num_sides", "equals", 4), ("object_colors", "equals", 5)]
        reference_objs = input_grid_obj.find_obj(query_attributes=query_1, obj_of_interest = obj)
        reference_obj_1 = reference_objs[0]
        landmark_1 = reference_obj_1.find_cell_color(color = reference_obj_1.get_dominant_color(), equals = False)
        translate_params_1['dx'] = (landmark_1[0][0] - obj.get_object_center()[0])
        translate_params_1['dy'] = landmark_1[0][1] - obj.get_object_center()[1]
        transformed_obj_grid = perform_transformation(obj_grid, "translate", translate_params_1) #returns numpy array
        obj.set_object_grid(transformed_obj_grid) #use set_object_grid to update GridObject with array
        ###remove part of object
        landmark_2 = obj.find_cell_bounds(index = 0, comparator = ">", value = reference_obj_1.get_bottom_row())
        transformed_obj_grid = remove_obj_portion(obj = obj.get_object_grid(), cell_list = landmark_2)
        obj.set_object_grid(transformed_obj_grid)
        output_grid_obj.add_obj(obj)
        return output_grid_obj

To deal with objects in the output grid that are NOT derived from the objects in the input grid, I will have a section
where I add "brand-new" objects. An object is considered "brand-new" if it cannot in any way be considered some combination of transformations
on an input object. Use this as a last resort. If an object can be considered a transformation of an existing one, do that. 

2. If brand new object(s) is being added to the grid, create a function called add_new_objects(input_grid_obj, output_grid_obj)
that performs this. It will be of the below format:

def add_new_objects(input_grid_obj, output_grid_obj):
    query = ##query to find what object we are adding new objects next to/with respect to
    reference_obj_list = input_grid_obj.find_obj(query)
    output_background_color = output_grid_obj.get_background_color()
    for reference_obj in reference_obj_list:
        landmark = (reference_obj.get_top_row() - 1, reference_obj.get_object_center()[1]) ##where we are adding object. change to wherever is correct.
        bounds = (landmark[0], landmark[0], landmark[1], landmark[1])
        new_obj_grid = create_new_object(grid_shape = output_grid_obj.get_shape(), background_color = output_background_color, obj_color = 4, obj_bounds = bounds)
        new_obj = GridObject(priority = 1, context_grid_obj = output_grid_obj, object_grid = new_obj_grid)
        output_grid_obj.add_obj(new_obj)
    return output_grid_obj


    The answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
    DO NOT give me different code for each input-output pair. Use '```python' and '```' as the start/end delimiters of code sections.
    Do not give any code that wasn't explicitly asked for. Stick exactly to the given function and variable naming conventions.
""".format(rule=rule_description, pairs=pairs)

    return prompt_template

def create_object_centric_same_size_template_some_code(input_list, input_objects_list, output_list, output_objects_list):
    pairs= str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nObjects in input: {objs}\nOutput: {out} \nObjects in output: {objs2}\n State in words what the rule seems to be for this pair.' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, output_list, output_objects_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. Each grid contains "objects" which can be various sizes, shapes, and colors.
The rule is a series of transformations on objects in the input to create the output.
Note: there exists a single rule that works for ALL of the input-output rules.
The pattern may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which
shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or
repeating a pattern for a fixed number of time.
There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
Pairs:
{}
Using the individual rules you came up with, state what seems to be the overall rule that applies to all input-output pairs.
Now you will assist me in implementing the rule programatically.
Elementary action functions. Use these to implement the transformations. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
create_new_object
    creates a new arbitrary, rectangular shaped object of single color. Shapes of object arrays must be same as shape of overall grid.
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        add_obj(obj):
        takes in GridObject instance and adds it to Grid object list
        input: obj. A GridObject instance 
        returns nothing.

        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
    methods:
        set_object_grid(grid)
        used when updating object grid with new transformed version. takes in 2d numpy grid representing a transformed object grid
        and updates all GridObject attributes accordingly
        input: 2d numpy array of object against background. returns nothing

        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.

I will implement the rule by looping through all of the objects in the grid and performing a series of:
if <object meets certain condition>, <perform certain transformation/series of transformations>

The first step is to come up with the correct object checkers. These will be filters that check for some object property 
to determine what should happen to the object.
Once an object checker has been made, the next step is to determine what should happen to objects that fit 
the checker.
The transformation will almost always be with respect to another object or grid level location.
For example, a translation will often be to move an object towards another object. An extension will often be towards another object or towards some part of the grid.
Color change will often be to the color of another object. Flipping will often be with respect to another object, and so on.
Answer the following questions. 
0. What will the background color of the output object be? Give the number only in your response.
1. How many object-checker transformation pairs are there for this rule?
For each one, answer the following questions. 
    a. State in words what property to check for in the object 
    b. Is this property being checked with respect to another object?
    c. If so, what properties describe the other object?
    d. Give me code for the object checker. Set it equal to a variable called obj_checker_n, with n being the number of the object checker.
    Example:
    if the overall rule is "change the color of the largest object to the color of the object closest to it", an object checker would be:
    
    obj_checker_1 = obj.get_area() == max(input_grid_obj.get_all_object_sizes())

    Other examples: 
        obj_checker_1 = (2 in obj.get_object_colors())
        obj_checker_2 = (4 in obj.get_object_colors())

    if the rule requires the checker to be with respect to another object, like with this one, "Extend the object that's aligned with the largest object
    in the grid until it touches the largest object," that would be done as follows:
    query_1 = [("area", "equals", max(input_grid_obj.get_all_object_sizes()))]
    reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
    obj_checker_1 = obj.get_object_center()[0] == reference_obj_1.get_object_center()[0]

        **object checkers must be formatted exactly like the examples above in one line

    For the transformation block that is under the if-statement for this object checker, answer the following questions.
    e. Should the object or any variation of it be added to the output grid?
    f. If so, how many times is the object or a transformation of it added to the output grid?
    For each one of those times, answer the following questions.
        i. What is the transformation/series of transformations? State these in terms of elementary actions.
        ii. What object does the transformation happen with respect to, if any? Give me properties that describe the object.
    g. Write a function that performs this transformation/series of transformations as some combination of elementary 
    actions. Function requirements. Must take in GridObject instance of the object we are on, the Grid instance for the input scene 
    as a whole, and the Grid instance for the output scene as a whole as the only parameters. 
    transformation_n(Grid input_grid_obj, Grid output_grid_obj, GridObject obj) <- the method signature must look like this, with n being the same number as the n in the corresponding object checker.
    Must perform transformations and update obj using set_object_grid(grid) and/or create new objects for transformed copies of an object and then 
    add them to output_grid_obj using add_obj(obj). Returns output_grid_obj with transformed GridObjects in it.
    You may use any combinations of elementary action functions as needed. Specific requirements are that other objects that are used when determining how to perform an action should be assigned to a variable called
    reference_obj_n, with n being a number. Any locations that the object is moving with respect to in any way should be assigned to a variable called landmark_n, with n being a number.
    Examples
    def transformation_1(input_grid_obj, output_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        ##adding original to output
        output_grid_obj.add_obj(obj)
        ##adding transformed object to output
        flip_params_1 = {{}}
        query_1 = [("object_colors", "equals", 6)]
        reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
        landmark_1 = reference_obj_1.get_object_center()
        flip_params_1["direction"] = "lr"
        flip_params_1["axis"] = landmark_1[1]
        transformed_obj_grid_1 = perform_transformation(grid = obj_grid, transformation_category="flip", params = flip_params_1) ##returns numpy array
        transformed_obj_1 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid=obj_grid_1) ##use numpy array to create GridObject
        output_grid_obj.add_obj(transformed_obj_1) #add GridObject to Grid 
        ##transform transformed object, add it to the grid
        transformed_obj_grid_2 = perform_transformation(grid = transformed_obj_grid_1, transformation_category = "flip", params = flip_params_1)
        transformed_obj_2 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid_2)
        output_grid_obj.add_obj(transformed_obj_2)
        return output_grid_obj

    def transformation_2(input_grid_obj, output_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        translate_params_1 = {{}}
        query_1 = [("num_sides", "equals", 4), ("object_colors", "equals", 5)]
        reference_objs = input_grid_obj.find_obj(query_attributes=query_1, obj_of_interest = obj)
        reference_obj_1 = reference_objs[0]
        landmark_1 = reference_obj_1.find_cell_color(color = reference_obj_1.get_dominant_color(), equals = False)
        translate_params_1['dx'] = (landmark_1[0][0] - obj.get_object_center()[0])
        translate_params_1['dy'] = landmark_1[0][1] - obj.get_object_center()[1]
        transformed_obj_grid = perform_transformation(obj_grid, "translate", translate_params_1) #returns numpy array
        obj.set_object_grid(transformed_obj_grid) #use set_object_grid to update GridObject with array
        ###remove part of object
        landmark_2 = obj.find_cell_bounds(index = 0, comparator = ">", value = reference_obj_1.get_bottom_row())
        transformed_obj_grid = remove_obj_portion(obj = obj.get_object_grid(), cell_list = landmark_2)
        obj.set_object_grid(transformed_obj_grid)
        output_grid_obj.add_obj(obj)
        return output_grid_obj

To deal with objects in the output grid that are NOT derived from the objects in the input grid, I will have a section
where I add "brand-new" objects. An object is considered "brand-new" if it cannot in any way be considered some combination of transformations
on an input object. Use this as a last resort. If an object can be considered a transformation of an existing one, do that. 

2. Are there "brand new" objects added to the grid? Yes or no.
3. If so, how many? Give only the number in your response.
4. for each one, determine:
    a. its color. You must give exactly one color in numeric form. Give only the number in your response. 
    b. where to add it. Give this as a 4-tuple indicating top,bottom,left,right. Similar to before, this is usually with
    respect to another object.
        i. List the properties that define the reference object and how the target bounds are related to the reference object.
    c. Determine if any transformation occurs on the object once it is added. If not, say so and move on to the next one. If so, similar to before,
    state the transformation that occurs. State which objects it occurs with respect to and how. (ex. object is extended until 1 cell past _____ object)
    Then, implement the transformation as specified earlier and give it back. This time, name the method new_object_transformation_n, with n being the number of the object, to differentiate it from the existing transformation methods
5. If brand new object(s) is being added to the grid, create a function called add_new_objects(input_grid_obj, output_grid_obj)
that performs this. It will be of the below format:

def add_new_objects(input_grid_obj, output_grid_obj):
    ##example of adding new object functionality. The structure and logic can be whatever you deem correct.
    ##the important part is that you use create_new_object to make the grid for the new GridObject. 
    ##This function can only create squares and rectangles
    query_1 = [("object_colors", "equals", 6)]
    ##example bound setting logic
    reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
    landmark_1 = reference_obj_1.get_object_center()
    bounds_1 = landmark_1[0] - 2, landmark_1[0] + 2, landmark_1[1] - 2, landmark_1[1] + 2
    new_obj_grid_1 = create_new_object(grid_shape = output_grid_obj.get_shape(), background_color = output_background_color, obj_color = color_1, obj_bounds = bounds_1)
    new_obj = GridObject(priority = 1, context_grid_obj = output_grid_obj, object_grid = new_obj_grid_1)
    output_grid_obj.add_obj(new_obj)
    return output_grid_obj


    The answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
    DO NOT give me different code for each input-output pair. Use '```python' and '```' as the start/end delimiters of code sections.
    Do not give any code that wasn't explicitly asked for. Stick exactly to the given function and variable naming conventions.
""".format(pairs)
    return code_template

def create_object_centric_smaller_size_template_some_code_1(input_list, input_objects_list, output_list, output_objects_list):
    pairs= str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nObjects in input: {objs}\nOutput: {out} \nObjects in output: {objs2}\n State in words what the rule seems to be for this pair.' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, output_list, output_objects_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. Transformations of an object are always with respect to another object or key location in the grid. Ex. An object won't just be translated down 2 cells for no reason, 
it will be translated down towards something (that something could be another object or the grid center, or something else).
The size of the grids will also get smaller from input to output, so you must determine the rule that changes grid size as well.
Note: there exists a single rule that works for ALL of the input-output rules.
Example: Change the color of an object of a certain shape and then zoom in to it.
Pairs:
{}
Using the individual rules you came up with, state what seems to be the overall rule that applies to all input-output pairs.
Now you will assist me in implementing the rule programatically.
Elementary action functions you have access to when describing/implementing rules. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
zoom_in
    parameters:
        obj: object 2d numpy grid
scale_down
    parameters:
        obj: object 2d numpy grid
        scale_factor: int, factor to scale up object by. shape, orientation, color do not change.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
create_new_object
    creates a new arbitrary, rectangular shaped object of single color
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
    methods:
        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.

Answer the following questions. 
1a. Are there objects in the output grid that are a transformed or untransformed version of an object in the input grid?
1b. List the properties of each of the objects.
    i. Create an object checker that can filter out and select objects that have the desired properties.
    Give me proper code for the object checker. Set it equal to a variable called obj_checker_n, with n being the number of the object checker.
    Example:
    if the overall rule is "add the largest object in the input grid to the output grid", an object checker would be:
    obj_checker_1 = obj.get_area() == max(input_grid_obj.get_all_object_sizes()) #all one line please
    Other examples: 
        obj_checker_1 = (2 in obj.get_object_colors())
        obj_checker_2 = (4 in obj.get_object_colors())
    ii. Are any transformations performed on the object before adding it to the output grid? Yes or no.
    iii. If so, what are the transformations? State in terms of elementary actions. For each transformation, answer the following question.
        x. What object does the transformation happen with respect to, if any? Give me properties that describe the object.
        y. Now, write a function that performs this transformation/series of transformations as some combination of elementary 
        actions. Function requirements. Must take in GridObject instance of the object we are on and the Grid instance for the input scene 
        as a whole as the only parameters. 
        transformation_n(Grid input_grid_obj, GridObject obj) <- the method signature must look like this, with n being the same number as the n in the object checker.
        Must return a list of numpy grids to add to the output grid object.
        You may use any combinations of elementary action functions as needed. Remember 
        actions are usually with respect to other objects. Specific requirements are that other objects that are used when determining how to perform an action should be assigned to a variable called
        reference_obj_n, with n being a number. Any locations that the object is moving with respect to in any way should be assigned to a variable called landmark_n, with n being a number.
    Examples of transformation methods:
    def transformation_1(input_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        transformed_obj_list = []
        transformed_obj_list.append(obj_grid) ##adding original object
        ##rotating about object that has color 6
        query_1 = [("object_colors", "equals", 6)]
        reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
        landmark_1 = reference_obj_1.get_object_center()
        flip_params_1 = {{}}
        flip_params_1["direction"] = "lr"
        flip_params_1["axis"] = landmark_1[1]
        obj_grid_1 = perform_transformation(grid=obj_grid, transformation_category="flip", params = flip_params_1)
        transformed_obj_list.append(obj_grid_1)
        return transformed_obj_list
    
    def transformation_2(input_grid_obj, obj):
        ##translating object until it is on top of cell that has certain color, adding translated object
        obj_grid = obj.get_object_grid()
        transformed_obj_list = []
        query_1 = [("num_sides", "equals", 4), ("object_colors", "equals", 1)]
        reference_objs = input_grid_obj.find_obj(query_attributes=query_1, obj_of_interest = obj)
        reference_obj_1 = reference_objs[0]
        landmark_1 = reference_obj_1.find_cell_color(color = reference_obj_1.get_dominant_color(), equals = False)
        translate_params_1 = {{}}
        translate_params_1['dx'] = (landmark_1[0][0] - obj.get_object_center()[0])
        translate_params_1['dy'] = landmark_1[0][1] - obj.get_object_center()[1]
        transformed_obj_grid_1 = perform_transformation(obj_grid, "translate", translate_params_1)
        transformed_obj_list.append(transformed_obj_grid)
        return transformed_obj_list
        
1c. Since the output grid is smaller than the input grid, we need to determine a rule for changing the size of the output grid to be the correct value
as well. The grid may be cropped, and if it is, the bounds are "locationally meaningful" in the grid. In other words, the bounds of the crop to create
the final output grid will often be bounds of an object, or bounds of some subdivision of the whole scene, or something else. State in words what the bounds are
with respect to objects or locations in the grid.
Implement in code some slicing of the output grid based on locationally meaningful bounds
have the code return the sliced output grid as the final answer.

Answer the below questions to consider the second "case" of a possible rule.
2a. Are there objects in the output grid that do not exist in any form in the input grid? I.e, no physical transformations on any objects
in the input grid would create the object(s) in the output grid. Answer yes or no.
2b. If so, even if the objects in the output grid are not "physically" related to the objects in the input grid, they are likely "logically" related. 
This means that some property of the objects in the output grid(color, size, location, shape, etc) has a logical connection to some property of some object(s) in the 
input. An example of this can be an object of a certain color whose height equals the number of instances of objects of that color in the input grid. This is just one example.
2c. What is the size of the output grid? How does it relate to the size of the input grid?
2d. What is the background color of the output grid?
2e. Programatically implement a section of code that sets a variable called output_obj_grid equal to a 2d numpy array of the proper shape and background color based on the size/color 
rule you just described. Make it of the form output_obj_grid = np.ones(induced_shape, dtype=int) * induced_background_color
2f. How many "brand new, logically related" objects are there in the output grid?
2g. For each one, answer the following questions:
    i. What is its color? How does its color relate to some property of some object in the input grid?
    ii. What is its size/shape? How does its size/shape relate to some property of some object in the input grid?
    iii. What properties can describe that "reference" object?
    iv. Implement code that adds these new objects to the grid in the format that follows

new_object_grid_list = []
for x in range(num_new_objects):
    if x == 1:
        #Write a programmatic statement that defines the reference object/value you are using to determine new object properties
        #this can be an individual object or grid level properties (number of total objects, etc)
        query_1 = [("object_colors", "equals", 6)]
        ##example property setting logic
        reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
        color_1 = reference_obj_1.get_dominant_color()
        bounds_1 = 0, input_obj.get_largest_object().get_area(), 0, 1 #example of bounds being logically related to something in input scene
        new_obj_grid_1 = create_new_object(color_1, bounds_1)
    elif x == 2:
        continue
        ##continue doing above logic for all new objects


    Remember, the answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
    DO NOT give me different code for each input-output pair. State what you believe the overall rule is before you answer the questions and give me code.
    Some rules regarding logic and formatting:
        only number the answers to the questions that are numbered. I use the numbers and letters to extract values from the prompt. Keep those numbers the same and add none of your own.
        If you are searching for certain object(s) using find_obj, the query variable MUST be named query_n, with n being the number of query
        If you are using another object (not the one you are manipulating) as a reference, it must be called reference_obj_n
        If you have a specific location to move with respect to, that variable must be called landmark_n, with n being a number
        Object checkers must be named obj_checker_n, with n being a number
        Parameters for a transformation must be set in a parameters dictionary. Each key must be set on its own line, like so.
        translate_params_1 = {{}}
        translate_params_1["dx"] = 1
        Parameters variables must be named as <action name>_params_n, with n being a number, like with the example directly above.
        Preferable to implement transformation using combinations of elementary action functions and not from scratch.
        Number in transformation function name must match number of its corresponding obj checker
""".format(pairs)
    return code_template



def create_object_centric_smaller_size_template_some_code_2(input_list, input_objects_list, output_list, output_objects_list):
    pairs= str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nObjects in input: {objs}\nOutput: {out} \nObjects in output: {objs2}\n State in words what the rule seems to be for this pair.' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, output_list, output_objects_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. Transformations of an object are always with respect to another object or key location in the grid. Ex. An object won't just be translated down 2 cells for no reason, 
it will be translated down towards something (that something could be another object or the grid center, or something else).
The size of the grids will also get smaller from input to output, so you must determine the rule that changes grid size as well.
Note: there exists a single rule that works for ALL of the input-output rules.
Example rules: Change the color of an object of a certain shape and then zoom in to it.
Crop out the section of an input grid that is subdivided by objects in a certin way.


etc
Rules can be more intricate than this, but they will be of this type.

Pairs:
{}
Using the individual rules you came up with, state what seems to be the overall rule that applies to all input-output pairs.
Now you will assist me in implementing the rule programatically.
Elementary action functions you have access to when describing/implementing rules. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
zoom_in
    parameters:
        obj: object 2d numpy grid
scale_down
    parameters:
        obj: object 2d numpy grid
        scale_factor: int, factor to scale up object by. shape, orientation, color do not change.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
create_new_object
    creates a new arbitrary, rectangular shaped object of single color
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
    methods:
        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.

Answer the following questions. 
1a. Are there objects in the output grid that do not exist in any form in the input grid? I.e, no physical transformations on any objects
in the input grid would create the object(s) in the output grid. Answer yes or no.
1b. If so, even if the objects in the output grid are not "physically" related to the objects in the input grid, they are likely "logically" related. 
This means that some property of the objects in the output grid(color, size, location, shape, etc) has a logical connection to some property of some object(s) in the 
input. An example of this can be an object of a certain color whose height equals the number of instances of objects of that color in the input grid. This is just one example.
1c. What is the size of the output grid? How does it relate to the size of the input grid?
1d. What is the background color of the output grid?
1e. Programatically implement a section of code that sets a variable called output_obj_grid equal to a 2d numpy array of the proper shape and background color based on the size/color 
rule you just described. Make it of the form output_obj_grid = np.ones(induced_shape, dtype=int) * induced_background_color
1f. How many "brand new, logically related" objects are there in the output grid?
1g. For each one, answer the following questions:
    i. What is its color? How does its color relate to some property of some object in the input grid?
    ii. What is its size/shape? How does its size/shape relate to some property of some object in the input grid?
    iii. What properties can describe that "reference" object?
    iv. Implement code that adds these new objects to the grid in the format that follows

new_object_grid_list = []
for x in range(num_new_objects):
    if x == 1:
        #Write a programmatic statement that defines the reference object/value you are using to determine new object properties
        #this can be an individual object or grid level properties (number of total objects, etc)
        query_1 = [("object_colors", "equals", 6)]
        ##example property setting logic
        reference_obj_1 = input_grid_obj.find_obj(query_1)[0]
        color_1 = reference_obj_1.get_dominant_color()
        bounds_1 = 0, input_obj.get_largest_object().get_area(), 0, 1 #example of bounds being logically related to something in input scene
        new_obj_grid_1 = create_new_object(color_1, bounds_1)
    elif x == 2:
        continue
        ##continue doing above logic for all new objects


Remember, the answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
DO NOT give me different code for each input-output pair. State what you believe the overall rule is before you answer the questions and give me code.
Some rules regarding logic and formatting:
    only number the answers to the questions that are numbered. I use the numbers and letters to extract values from the prompt. Keep those numbers the same and add none of your own.
    If you are searching for certain objects(s) using find_obj, the query variable MUST be named query_n, with n being the number of query
    If you are using another object (not the one you are manipulating) as a reference, it must be called reference_obj_n
    If you have a specific location to move with respect to, that variable must be called landmark_n, with n being a number
    Object checkers must be named obj_checker_n, with n being a number
    Parameters for a transformation must be set in a parameters dictionary. Each key must be set on its own line, like so.
    translate_params_1 = {{}}
    translate_params_1["dx"] = 1
    Parameters variables must be named as <action name>_params_n, with n being a number, like with the example directly above.
""".format(pairs)
    return code_template

def create_object_centric_larger_size_template_some_code(input_list, input_objects_list, output_list, output_objects_list):
    pairs= str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nObjects in input: {objs}\nOutput: {out} \nObjects in output: {objs2}\n State in words what the rule seems to be for this pair.' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, output_list, output_objects_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. The size of the grid will also get bigger from input to output, so determine how grid size changes from input to output. Transformations of an object are always with respect to another object or key location in the grid. Ex. An object won't just be translated down 2 cells for no reason, 
it will be translated down towards something (that something could be another object or the grid center, or something else).
Note: there exists a single rule that works for ALL of the input-output rules.
Example rules: If an object is a square, paint it red. If it is not, paint it blue. 
Make copies of an object, flip them about some axis, and add the copies to the output grid along with the original.
Extend an object until it touches the left side of the object to the right of it.
etc
Rules can be more intricate than this, but they will be of this type.

Pairs:
{}
Using the individual rules you came up with, state what seems to be the overall rule that applies to all input-output pairs.
Now you will assist me in implementing the rule programatically.
Elementary action functions you have access to when describing/implementing rules. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
zoom_in
    parameters:
        obj: object 2d numpy grid
scale_up
    parameters:
        obj: object 2d numpy grid
        scale_factor: int, factor to scale up object by. shape, orientation, color do not change.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
create_new_object
    creates a new arbitrary, rectangular shaped object of single color
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
    methods:
        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.

I will implement the rule by looping through all of the objects in the grid and performing a series of:
if <object meets certain condition>, <perform certain transformation/series of transformations>
I can either: 
1. Add the object or not add the object to the output grid
and/or
2. perform transformations on the object and add the transformed object grid.
The first step is to come up with the correct object checkers. These will be filters that check for some object property 
to determine what should happen to the object. Object checkers can either: 
1. Check an object property directly against some constant value
2. Check an object property against another object in the grid. 
Once an object checker has been made, the next step is to determine what should happen to objects that fit 
the checker.
The transformation will almost always be with respect to another object or grid level location.
For example, a translation will often be to move an object towards another object. An extension will often be towards another object or towards some part of the grid.
Color change will often be to the color of another object. Flipping will often be with respect to another object, and so on.
Answer the following questions. 
0a. What will the background color of the output object be? Give the number only in your response.
0b. What will the shape of the output grid be? What is the relationship between the shape of the output grid and the shape of the input grid?
0c. Implement in code the setting of output grid shape, as a constant or as some function on input grid shape. Make it one line of the format below.
output_grid_shape = 2 * grid.shape[0], 2 * grid.shape[1] ##example

1. How many object-checker transformation pairs are there for this rule?
For each one, answer the following questions. 
    a. State in words what property to check for in the object 
    b. Is this property being checked with respect to another object? Yes or no. i.e, checking if object is same color as another object, checking 
    if object's center is between the top row and bottom row of another object, etc. 
    c. If so, what properties describe the other object?
    d. Give me proper code for the object checker. Set it equal to a variable called obj_checker_n, with n being the number of the object checker.
    Example:
    if the overall rule is "change the color of the largest object to the color of the object closest to it", an object checker would be:
    
    obj_checker_1 = obj.get_area() == max(input_grid_obj.get_all_object_sizes()) #all one line please

    Other examples: 
        obj_checker_1 = (2 in obj.get_object_colors())
        obj_checker_2 = (4 in obj.get_object_colors())

    if the rule requires the checker to be with respect to another object, like with this one, "Extend the object that's aligned with the largest object
    in the grid until it touches the largest object," that would be done as follows:
    query_1 = [("area", "equals", max(input_grid_obj.get_all_object_sizes()))]
    reference_obj_1 = input_grid_obj.find_obj(query_1)[0]
    obj_checker_1 = obj.get_object_center()[0] == reference_obj_1.get_object_center()[0]

        **object checkers must be formatted exactly like the examples above

    Now, for the transformation block that is under the if-statement for this object checker, answer the following questions.
    e. Should the object or any variation of it be added to the output grid? Yes or no.
    f. If so, how many times is the object or a transformation of it added to the output grid? Give me a number.
    For each one of those times, answer the following questions.
        i. What is the transformation/series of transformations? State in terms of elementary actions.
        ii. What object does the transformation happen with respect to, if any? Give me properties that describe the object.
    g. Now, write a function that performs this transformation/series of transformations as some combination of elementary 
    actions. Function requirements. Must take in GridObject instance of the object we are on and the Grid instance for the input scene 
    as a whole as the only parameters. 
    transformation_n(Grid input_grid_obj, GridObject obj) <- the method signature must look like this, with n being the same number as the n in the object checker.
    Must return a list of numpy grids to add to the output grid object.
    You may use any combinations of elementary action functions as needed. Remember 
    actions are usually with respect to other objects. Specific requirements are that other objects that are used when determining how to perform an action should be assigned to a variable called
    reference_obj_n, with n being a number. Any locations that the object is moving with respect to in any way should be assigned to a variable called landmark_n, with n being a number.
    Examples of transformation methods:
    def transformation_1(input_grid_obj, obj):
        obj_grid = obj.get_object_grid()
        transformed_obj_list = []
        transformed_obj_list.append(obj_grid) ##adding original object
        ##rotating about object that has color 6
        query_1 = [("object_colors", "equals", 6)]
        reference_obj_1 = input_grid_obj.find_obj(query_1)[0]
        landmark_1 = reference_obj_1.get_object_center()
        flip_params_1 = {{}}
        flip_params_1["direction"] = "lr"
        flip_params_1["axis"] = landmark_1[1]
        obj_grid_1 = perform_transformation(grid=obj_grid, transformation_category="flip", params = flip_params_1)
        transformed_obj_list.append(obj_grid_1)
        return transformed_obj_list
    
    def transformation_2(input_grid_obj, obj):
        ##translating object until it is on top of cell that has certain color, adding translated object
        obj_grid = obj.get_object_grid()
        transformed_obj_list = []
        query_1 = [("num_sides", "equals", 4), ("object_colors", "equals", 1)]
        reference_objs = input_grid_obj.find_obj(query_attributes=query_1)
        reference_obj_1 = reference_objs[0]
        landmark_1 = reference_obj_1.find_cell_color(color = reference_obj_1.get_dominant_color(), equals = False)
        translate_params_1 = {{}}
        translate_params_1['dx'] = (landmark_1[0][0] - obj.get_object_center()[0])
        translate_params_1['dy'] = landmark_1[0][1] - obj.get_object_center()[1]
        transformed_obj_grid_1 = perform_transformation(obj_grid, "translate", translate_params_1)
        transformed_obj_list.append(transformed_obj_grid)
        return transformed_obj_list

Finally, to deal with objects in the output grid that are NOT derived from the objects in the input grid, I will have a section
where I add "brand-new" objects. An object is considered "brand-new" if it cannot in any way be considered some combination of transformations
on an input object. Use this as a last resort. If an object can be considered a transformation of an existing one, do that. Answer the following questions. 

2. Are there "brand new" objects added to the grid? Yes or no. Only have yes or no in the response.
3. If so, how many? Give only the number in your response.
4. for each one, determine:
    a. its color. You must give exactly one color in numeric form. Give only the number in your response. 
    b. where to add it. Give this as a 4-tuple indicating top,bottom,left,right. Similar to before, this is usually with
    respect to another object.
        i. List the properties that define the reference object and how the target bounds are related to the reference object.
    c. Determine if any transformation occurs on the object once it is added. If not, say so and move on to the next one. If so, similar to before,
    state the transformation that occurs. State which objects it occurs with respect to and how. (ex. object is extended until 1 cell past _____ object)
    Then, implement the transformation as specified earlier and give it back.
5. If brand new object(s) is being added to the grid, implement the add block programatically in the format that follows:
new_object_grid_list = []
for x in range(num_new_objects):
    if x == 1:
        color_1 = 1 ##set color here
        #Write a programmatic statement that defines a landmark using other values in the grid and then uses that to set the bounds
        #Example.
        query_1 = [("object_colors", "equals", 6)]
        ##example bound setting logic
        reference_obj_1 = input_grid_obj.find_obj(query_1)[0]
        landmark_1 = reference_obj_1.get_object_center()
        bounds_1 = landmark_1[0] - 2, landmark_1[0] + 2, landmark_1[1] - 2, landmark_1[1] + 2
        new_obj_grid_1 = create_new_object(color_1, bounds_1)
        ##determine if any transformations occur to the new object. If any do occur, use a similar process to implementing transformations as before to do it.
        ##this time, name the method new_object_transformation_n, with n being the number of the object, to differentiate it from the existing transformation methods
        ##if transformation occurs, add transformed object(s) to new_object_grid_list(). Otherwise, add the new_obj_grid_n variable to the list and move on.
    elif x == 2:
        continue
        ##continue doing above logic for all new objects


Remember, the answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
DO NOT give me different code for each input-output pair. State what you believe the overall rule is before you answer the questions and give me code.
Some rules regarding logic and formatting:
    only number the answers to the questions that are numbered. I use the numbers and letters to extract values from the prompt. Keep those numbers the same and add none of your own.
    If you are searching for certain object(s) using find_obj, the query variable MUST be named query_n, with n being the number of query
    If you are using another object (not the one you are manipulating) as a reference, it must be called reference_obj_n, with n being a number
    If you have a specific location to move with respect to, that variable must be called landmark_n, with n being a number
    Object checkers must be named obj_checker_n, with n being a number
    Parameters for a transformation must be set in a parameters dictionary. Each key must be set on its own line, like so.
    translate_params_1 = {{}}
    translate_params_1["dx"] = 1
    Parameters variables must be named as <action name>_params_n, with n being a number.
    Preferable to implement transformation using combinations of elementary action functions and not from scratch.
    Number in transformation function name must match number of its corresponding obj checker. Use perform_transformation always, do not directly call transformation functions.
""".format(pairs)
    return code_template



def create_grid_as_object_template(input_list, output_list):
    pairs= str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nOutput: {out} \n State in words what the rule seems to be for this pair.' for i, (inp, out) in enumerate(zip(input_list, output_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. The rule will encompass change in both the contents of the grid and the 
shape/size of it.
There exists a single rule that works for ALL of the input-output rules.

Pairs:
{}
Using the individual rules you came up with, state what seems to be the overall rule that applies to all input-output pairs.
Now you will assist me in implementing the rule programatically.

Answer the following questions. 
0. Does the shape of the grid change from input to output? If so, does it get larger or smaller?
1. What is the size of the output grid? How does it relate to the size of the input grid?
2. Are there any sections of the input grid that are "blanked out"? If so, does it seem like the output rule involves "filling in"
the blank sections?
3. Does the output grid contain the input grid (or a flipped or rotated version of it) as a subsection? If so, what is the transformation/combination
of transformations that create the output grid from the input grid?

Now, fill in the below code template
###do not change the method name
def grid_as_object_template(input_grid):
    output_grid = input_grid.copy()
    ###fill this in with the transformations that turn the input grid into the output grid
    return output_grid

""".format(pairs)
    return code_template


def create_mutate_object_centric_program_prompt(input_list, input_objects_list, actual_output_list, output_objects_list, pred_output_list, current_program, original_rule):
    pairs = str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nObjects in input: {objs[1:]}\nOutput: {out} \nObjects in output: {objs2[1:]}\n' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, input_objects_list, actual_output_list, output_objects_list))]))
    pairs_2 = str('\n'.join([f'Pair {i+1}:\nPredicted Output: {pred_out}\nActual Output: {act_out} \n State in words what seems to be the difference between the predicted output and actual output.' for i, (pred_out, act_out) in enumerate(zip(pred_output_list, actual_output_list))]))
    code_template = """
Your task is to describe the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. Transformations of an object are always with respect to another object or key location in the grid.

Pairs:
{}
Elementary action functions. Use these to implement the transformations. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
rotate_in_place
    rotates an object about itself while keeping position the same
    parameters:
        angle: int. 90, 180, or 270
create_new_object
    creates a new arbitrary, rectangular shaped object of single color. Shapes of object arrays must be same as shape of overall grid.
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        add_obj(obj):
        takes in GridObject instance and adds it to Grid object list
        input: obj. A GridObject instance 
        returns nothing.

        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: List of query tuples. Each tuple contains:
            key: Attribute to check (e.g., "area", "color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj([("area", "equals", 10)], obj).
        To locate object(s) by color: input_grid_obj.find_obj([("object_colors", "equals", 7)], obj).
    GridObject: class for individual object within the grid.
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
        get_object_number() gets the object number. objects are numbered left to right, top to bottom. I.E, topmost leftmost object is object 1, and so on.
    methods:
        set_object_grid(grid)
        used when updating object grid with new transformed version. takes in 2d numpy grid representing a transformed object grid
        and updates all GridObject attributes accordingly
        input: 2d numpy array of object against background. returns nothing

        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.

Generic helper functions (not part of any class)
determine_direction_landmark(landmark, object_cell)
takes in landmark (a tuple, (x,y)) and object_cell (a tuple, (x,y)) and determines which direction the landmark is with respect to
the object cell. returns a string indicating direction.

I have the below program that is attempting to implement a rule. This program is the result of a genetic algorithm, not a human written program,
so there can be aspects that are bad/nonsensical.

{}

The output of the program in comparison to the correct answers are listed below.


{}

Original rule that the program was trying to implement:
{}

What seems to be the rule that the program is currently implementing?
What seems to be the trend as for the difference between the correct output and the predicted output?
What is incorrect, if anything, about the program/rule that is being implemented above?

Now, make edits to the code given and return the new program, and all helper functions given, such that they correctly implement the rule 
that transforms each input given to its corresponding output. Keep the code style as similar as possible. Make as few changes as necessary. Keep variable naming conventions the same.
Use '```python' and '```' as the start/end delimiters of code sections.
""".format(pairs, current_program, pairs_2, original_rule)
    return code_template







































###returns string that has the information about the problem and code that the LLM will use as context on all further 
###LLM calls
def create_context_for_caching():
    context_str = """Your task is to give me code for a transformation function that applies a consistent rule to turn input grids into output grids.
The below information will provide information about the rule and code you will use to implement the rule.

The rule will be represented as a series of:

if <object property>, then <transformation> statements. 
An example rule: if object is a red square, move it down towards the blue object.
If an object is blue, keep as is.

Your task is to give me: the object checkers (boolean expressions that check for an object property), transformation functions (transformations on object 
that occur if the object meets some condition), and add new object function (function that potentially adds objects 'from scratch'. may or may not exist) if the rule
calls for adding brand new objects that are not transformations of existing objects. Here is information about code that you will call to implement the answers.
Elementary action functions. Use these to implement the transformations. All of these return a numpy grid
with the transformed object:
translate
    parameters:
        obj: object 2d numpy grid
        dx: int, up/down offset
        dy: int, left/right offset
flip
    parameters:
        obj: object 2d numpy grid
        direction: "ud" for up-down, "lr" for left-right
        axis: axis to flip about, row or column depending on direction
extend
    parameters:
        obj: object 2d numpy grid
        cell: int indicating which cell (1 for 1st cell, 2 for 2nd, etc. cells number left to right, top to bottom) or x,y for cell location
        direction: "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right" are options
        distance: int, how much to extend chosen cell of object in chosen direction
change_color
    parameters:
        obj: object 2d numpy grid
        color: int, between 0 and 10. All colors are represented as ints between 0 and 10.
remove_obj_portion
    removes part of an object
    parameters:
        obj: object 2d numpy array
        cell_list: list of cells to remove from object
rotate_in_place
    rotates an object about itself while keeping position the same
    parameters:
        angle: int. 90, 180, or 270
scale_up
    parameters:
        obj: object 2d numpy grid
        scale_factor: int, factor to scale up object by. shape, orientation, color do not change.
create_new_object
    creates a new arbitrary, rectangular shaped object of single color. Shapes of object arrays must be same as shape of overall grid.
    parameters:
        grid_shape: tuple. x,y indicating shape of overall grid object is in
        background_color: int, 0 to 10. color of background to place object against.
        obj_color: color of object to create, int 0-10
        obj_bounds: tuple. top, bottom, left, right. Where to place new object.
Classes and properties you have access to when describing/implementing rules:
Grid: class for the entire grid for an input or output. 
constructor: Grid(grid)
grid: the numpy array representing the whole scene
    properties in the form of getters:
        get_grid() 2d numpy grid 
        get_background_array() numpy grid of same size as main grid with background values
        get_background_color() int
        get_shape() x,y dimensions of grid
        get_center() x,y center location
        get_object_list() list of GridObject objects for all objects in scene.
        get_color_occurrences() dict where keys are color ints and values are how frequently they occur (int)
        get_all_object_sizes() list of ints, list of all object areas
        get_largest_object() GridObject instance for object with largest area
        get_smallest_object() GridObject instance for object with smallest area
        get_all_object_centers() list of x,y centers of all objects in scene
    methods:
        add_obj(obj):
        takes in GridObject instance and adds it to Grid object list
        input: obj. A GridObject instance 
        returns nothing.

        find_obj(query_attributes, obj_of_interest):
        Finds an object that matches a set of attribute conditions.
        Input:
        query_attributes: A query tuple. The tuple contains:
            key: Attribute to check (e.g., "area", "object_color").
            relation: How to compare ("equals", "greater", "less").
            value: The value to compare against (e.g., color = 7, area > 10).
        obj_of_interest: GridObject instance for object we are currently "operating on"
        Output:
        Returns a list of the GridObject object that satisfy the conditions from query_attributes. List is sorted by default
        based on distance, such as the closest possible reference object comes first, and farthest last.
        Example Use Case:
        To find object(s) by size: input_grid_obj.find_obj(("area", "equals", 10), obj).
        To locate object(s) by color: input_grid_obj.find_obj(("object_colors", "equals", 7), obj).
        Possible keys for find_obj query: 'object_number', 'object_color', 'num_sides', 'area', 'object_center', 'object_center[0]', 
        'object_center[1]', 'top_row', 'bottom_row', 'left_column', 'right_column'
    GridObject: class for individual object within the grid.
    constructor: GridObject(priority, context_grid_obj, object_grid)
    priority: int indicating whether it should show up on top of other objects if there is an overlap. higher number means overlap priority given.
    context_grid_obj: Grid object for the overall scene that this GridObject is a part of
    object_grid: numpy grid for this object
    properties in the form of getters:
        get_object_grid() 2d numpy grid of object against plain background
        get_object_colors() list of colors (ints) that make up object
        get_dominant_color() a single int
        get_object_map() N by 2 numpy array of all of the x,y cells that make up object
        get_num_sides()
        get_num_corners()
        get_angles() returns list of ints/floats of angles of all corners of object
        get_area()
        get_corner_points() list of x,y locations of object corners
        get_object_center() x,y location
        get_top_row() topmost row of object, int
        get_bottom_row()
        get_left_column() leftmost column of object, int
        get_right_column()
        get_object_gaps() list of lists of x,y cell locations. Each sublist is a gap or hole within object.
        get_sub_pattern() 2d array, any pattern that occurs within object
        get_object_number() gets the object number. objects are numbered left to right, top to bottom. I.E, topmost leftmost object is object 1, and so on.
    methods:
        set_object_grid(grid)
        used when updating object grid with new transformed version. takes in 2d numpy grid representing a transformed object grid
        and updates all GridObject attributes accordingly
        input: 2d numpy array of object against background. returns nothing

        find_cell_pos(position)
        takes integer indicating cell position number and returns tuple (x,y) indicating cell position. 
        Ex. passing in 1 means finding position of "first" cell in object, passing in 3 means getting location of "third" object in cell, etc
        Cells are numbered on object left to right, top to bottom. cell 1 is the topmost/leftmost cell of object, etc.

        find_cell_color(color, equals)
        color: int, number between 0 and 10 indicating color value.
        equals: boolean, True or False. indicates whether we are looking for cells that EQUAL or DO NOT EQUAL a color.
        returns: a list of tuples (x,y) containing cells from object that are of a certain color/that do not equal a certain color.

        find_cell_bounds(index, comparator, value)
        index: int between 0 and 1, indicates whether we are comparing index 0 (along the rows) or index 1 (along the columns)
        comparator: string among ('>', '<', and '==') that indicates what comparison to do.
        value: value indicating which row/column to compare with
        returns: a list of tuples (x,y) containing cells that are above or below a certain row OR to the left or right of a certain column
        Example usage
        obj.find_cell_bounds(0, "==", 3). Finds all cells in object that are on the 3rd row.
Generic helper functions (not part of any class)
determine_direction_landmark(landmark, object_cell)
takes in landmark (a tuple, (x,y)) and object_cell (a tuple, (x,y)) and determines which direction the landmark is with respect to
the object cell. returns a string indicating direction.

The first step is to come up with the correct object checkers. These will be filters that check for some object property 
to determine what should happen to the object.
Once an object checker has been made, the next step is to determine what should happen to objects that fit 
the checker.
The transformation will almost always be with respect to another object or grid level location.
For example, a translation will often be to move an object towards another object. An extension will often be towards another object or towards some part of the grid.
Color change will often be to the color of another object. Flipping will often be with respect to another object, and so on.

For each object checker, set it equal to a variable called obj_checker_n, with n being the number of the object checker.
Example:
if the overall rule is "change the color of the largest object to the color of the object closest to it", an object checker would be:

obj_checker_1 = obj.get_area() == max(input_grid_obj.get_all_object_sizes())

Other examples: 
    obj_checker_1 = (2 in obj.get_object_colors())
    obj_checker_2 = (4 in obj.get_object_colors())

if the rule requires the checker to be with respect to another object, like with this one, "Extend the object that's aligned with the largest object
in the grid until it touches the largest object," that would be done as follows:
query_1 = [("area", "equals", max(input_grid_obj.get_all_object_sizes()))]
reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
obj_checker_1 = obj.get_object_center()[0] == reference_obj_1.get_object_center()[0]

    **object checkers must be formatted exactly like the examples above in one line

For the transformation block that corresponds to the if-statement for each object checker, write a function that performs this transformation/series of transformations as some combination of elementary 
actions. Function requirements. Must take in GridObject instance of the object we are on, the Grid instance for the input scene 
as a whole, and the Grid instance for the output scene as a whole as the only parameters. 
transformation_n(Grid input_grid_obj, Grid output_grid_obj, GridObject obj) <- the method signature must look like this, with n being the same number as the n in the corresponding object checker.
Must perform transformations and update obj using set_object_grid(grid) and/or create new objects for transformed copies of an object and then 
add them to output_grid_obj using add_obj(obj). Returns output_grid_obj with transformed GridObjects in it.
You may use any combinations of elementary action functions as needed. Specific requirements are that other objects that are used when determining how to perform an action should be assigned to a variable called
reference_obj_n, with n being a number. Any locations that the object is moving with respect to in any way should be assigned to a variable called landmark_n, with n being a number.
Examples
def transformation_1(input_grid_obj, output_grid_obj, obj):
    obj_grid = obj.get_object_grid()
    ##adding original to output
    output_grid_obj.add_obj(obj)
    ##adding transformed object to output
    flip_params_1 = {{}}
    query_1 = ("object_colors", "equals", 6)
    reference_obj_1 = input_grid_obj.find_obj(query_1, obj)[0]
    landmark_1 = reference_obj_1.get_object_center()
    flip_params_1["direction"] = "lr"
    flip_params_1["axis"] = landmark_1[1]
    transformed_obj_grid_1 = perform_transformation(grid = obj_grid, transformation_category="flip", params = flip_params_1) ##returns numpy array
    transformed_obj_1 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid=obj_grid_1) ##use numpy array to create GridObject
    output_grid_obj.add_obj(transformed_obj_1) #add GridObject to Grid 
    ##transform transformed object, add it to the grid
    transformed_obj_grid_2 = perform_transformation(grid = transformed_obj_grid_1, transformation_category = "flip", params = flip_params_1)
    transformed_obj_2 = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid_2)
    output_grid_obj.add_obj(transformed_obj_2)
    return output_grid_obj

def transformation_2(input_grid_obj, output_grid_obj, obj):
    obj_grid = obj.get_object_grid()
    translate_params_1 = {{}}
    query_1 = ("num_sides", "equals", 4)
    reference_objs = input_grid_obj.find_obj(query_attributes=query_1, obj_of_interest = obj)
    reference_obj_1 = reference_objs[0]
    landmark_1 = reference_obj_1.find_cell_color(color = reference_obj_1.get_dominant_color(), equals = False)
    translate_params_1['dx'] = (landmark_1[0][0] - obj.get_object_center()[0])
    translate_params_1['dy'] = landmark_1[0][1] - obj.get_object_center()[1]
    transformed_obj_grid = perform_transformation(obj_grid, "translate", translate_params_1) #returns numpy array
    obj.set_object_grid(transformed_obj_grid) #use set_object_grid to update GridObject with array
    ###remove part of object
    landmark_2 = obj.find_cell_bounds(index = 0, comparator = ">", value = reference_obj_1.get_bottom_row())
    transformed_obj_grid = remove_obj_portion(obj = obj.get_object_grid(), cell_list = landmark_2)
    obj.set_object_grid(transformed_obj_grid)
    output_grid_obj.add_obj(obj)
    return output_grid_obj

To deal with objects in the output grid that are NOT derived from the objects in the input grid, I will have a section
where I add "brand-new" objects. An object is considered "brand-new" if it cannot in any way be considered some combination of transformations
on an input object. Use this as a last resort. If an object can be considered a transformation of an existing one, do that. 

If brand new object(s) is being added to the grid, create a function called add_new_objects(input_grid_obj, output_grid_obj)
that performs this. It will be of the below format:

def add_new_objects(input_grid_obj, output_grid_obj):
    query = ##query to find what object we are adding new objects next to/with respect to
    reference_obj_list = input_grid_obj.find_obj(query)
    output_background_color = output_grid_obj.get_background_color()
    for reference_obj in reference_obj_list:
        landmark = (reference_obj.get_top_row() - 1, reference_obj.get_object_center()[1]) ##where we are adding object. change to wherever is correct.
        bounds = (landmark[0], landmark[0], landmark[1], landmark[1])
        new_obj_grid = create_new_object(grid_shape = output_grid_obj.get_shape(), background_color = output_background_color, obj_color = 4, obj_bounds = bounds)
        new_obj = GridObject(priority = 1, context_grid_obj = output_grid_obj, object_grid = new_obj_grid)
        output_grid_obj.add_obj(new_obj)
    return output_grid_obj


The answers for the questions and the code you give me must pertain to the single rule that works for all input-output pairs. 
DO NOT give me different code for each input-output pair. Use '```python' and '```' as the start/end delimiters of code sections.
Do not give any code that wasn't explicitly asked for. Stick exactly to the given function and variable naming conventions.
    """
    return context_str




def create_generate_code_prompt_body(rule_description, input_grids, output_grids):
    # Create formatted string for each pair of input and output grids
    pairs = '\n'.join(
        [f"Pair {i+1}:\nInput Grid:\n{inp}\nOutput Grid:\n{out}"
         for i, (inp, out) in enumerate(zip(input_grids, output_grids))]
    )
    code_prompt_body = """Your task is to give me code for a transformation function that applies a consistent rule to turn input grids into output grids.
The rule, described below, should be followed exactly to produce the correct transformations for all provided examples.

Rule to implement:
{}

Here are the input-output pairs:

{}

First, convert the rule to a series of if <object property>, then <transformation> statements. 

Now, using the information and code given in the context, implement a series of object-checker and transformation pairs. 
For each pair in the rule

1. give me an object checker in the format obj_checker_n = <some boolean expression checking object property>
2. A transformation on the object implemented in a function called transformation_n, with n being a number. 

If brand new objects are added to the grid (objects that are not a transformation of any existing object), implement a function called
add_new_objects that creates and adds the brand new objects to the grid.
Use '```python' and '```' as the start/end delimiters of code sections. Put new ones around each code snippet. I.e each object checker and new function is in its own
'```python' and '```' pair, rather than all of the code being put in a single '```python' and '```' block. Avoid while loops to avoid risk of infinite looping.
    """.format(rule_description, pairs)
    return code_prompt_body

def create_mutate_code_prompt_body(input_list, actual_output_list, pred_output_list, current_program):
    pairs = str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nOutput: {out} \n' for i, (inp, objs, out, objs2) in enumerate(zip(input_list, actual_output_list))]))
    pairs_2 = str('\n'.join([f'Pair {i+1}:\nPredicted Output: {pred_out}\nActual Output: {act_out} \n' for i, (pred_out, act_out) in enumerate(zip(pred_output_list, actual_output_list))]))
    code_template = """
Your task is to implement the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. Transformations of an object are always with respect to another object or key location in the grid.

Pairs:
{}

I have the below program that is attempting to implement a rule. This program is the result of a genetic algorithm, not a human written program,
so there can be aspects that are bad.

{}

The output of the program in comparison to the correct answers are listed below.


{}

What seems to be the rule the code is trying to implement?
What seems to be the trend as for the difference between the correct output and the predicted output?
Given the differences, is the original rule correct? If not, what needs to change?


Now, make edits to the code given and return the new program, and all helper functions given, such that they correctly implement the rule 
that transforms each input given to its corresponding output. Keep the code style as similar as possible. Make as few changes as necessary. Keep variable naming conventions the same.
Use '```python' and '```' as the start/end delimiters of code sections.
""".format(pairs, current_program, pairs_2)
    return code_template





def create_fill_in_partial_program_code_prompt_body(input_list, actual_output_list, pred_output_list, current_program):
    pairs = str('\n'.join([f'Pair {i+1}:\nInput: {inp}\nOutput: {out} \n' for i, (inp, out) in enumerate(zip(input_list, actual_output_list))]))
    pairs_2 = str('\n'.join([f'Pair {i+1}:\nPredicted Output: {pred_out}\nActual Output: {act_out} \n' for i, (pred_out, act_out) in enumerate(zip(pred_output_list, actual_output_list))]))
    code_template = """
Your task is to implement the rule that transforms each input grid into the output grid. Each grid is composed of "objects" which can be various sizes, shapes, and colors.
Think of the rule as a series of transformations on objects in the input to create the output. Transformations of an object are always with respect to another object or key location in the grid.

Pairs:
{}

I have the below program that is attempting to implement a rule. The program is the result of a search over partial programs. Assume the rule the program is partially implementing is correct and that the 
program is "on the right track", but is incomplete. Induce the rule the program is trying to implement and complete it such that the predicted outputs fully match the actual outputs. 
Use the same style to add the rest of the missing code. 

{}

The output of the program in comparison to the correct answers are listed below.


{}

What seems to be the rule the code is trying to implement?



Now, continue the code given and return the new program, and all helper functions given, such that they correctly implement the rule 
that transforms each input given to its corresponding output. Keep the code style as similar as possible. Keep variable naming conventions the same.
Use '```python' and '```' as the start/end delimiters of code sections.
""".format(pairs, current_program, pairs_2)
    return code_template