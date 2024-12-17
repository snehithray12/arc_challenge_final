from evolution_search_utils import indent
from Program import SameSizeObjectCentricProgram, extract_transformation_number, get_empty_program_params
from evolution_search_utils import get_variable_value, process_plus_string, count_leading_spaces, extract_extend_parameter_name
import re
import random
import copy
from actions import zoom_in, flip, rotate_in_place, change_color
import numpy as np
from scipy.signal import correlate2d
import math
from visualizer import visualize_single_pair
import ast


##takes in all input grids for a problem and returns a dictionary of dictionaries that indicates the possible object 
##property values and how often they occur
def get_properties_domain(input_grid_objs):
    ##create a dictionary of all possible values for all major unary properties that could be checked
    ##this will be a "domain" for sampled object checkers. for example, if a red object does not exist in the grids, we will not sample
    ##an object checker that checks for red. waste of computation.
    def update_occurrences(num_occurrences_each_property, sub_num_occurrences, property_name, key):
        """Update num_occurrences_each_property and sub_num_occurrences for a given property and key."""
        if property_name in num_occurrences_each_property:
            if key in num_occurrences_each_property[property_name]:
                if key in sub_num_occurrences[property_name]:
                    num_occurrences_each_property[property_name][key] += 1
                    sub_num_occurrences[property_name][key] += 1
                    return
                else:
                    num_occurrences_each_property[property_name][key] += 1
                    sub_num_occurrences[property_name][key] = 1
                    return
            else:
                num_occurrences_each_property[property_name][key] = 1
                sub_num_occurrences[property_name][key] = 1
                return
    num_occurrences_each_property = {"dominant_color": {}, "color_list": {}, "num_sides": {}, "num_corners": {}, "area": {}, "object_gaps": {}}
    for input_grid_obj in input_grid_objs:
        sub_num_occurrences = {"dominant_color": {}, "color_list": {}, "num_sides": {}, "num_corners": {}, "area": {}, "object_gaps": {}}
        for input_obj in input_grid_obj.get_object_list():
            obj_color = int(input_obj.get_dominant_color())
            obj_color_list = input_obj.get_object_colors()
            obj_color_list.sort()
            obj_num_sides = input_obj.get_num_sides()
            obj_num_corners = input_obj.get_num_corners()
            obj_area = input_obj.get_area()
            obj_gaps = input_obj.get_object_gaps()
            ###filling out num_occurrences_each_property to know how often each property occurs
            ##if we already came across the color, increment the counter tracking how many times it shows up
            # Update "dominant_color"
            update_occurrences(num_occurrences_each_property, sub_num_occurrences, "dominant_color", obj_color)
            # Update "color_list"
            update_occurrences(num_occurrences_each_property, sub_num_occurrences, "color_list", str(obj_color_list))
            # Update "num_sides"
            update_occurrences(num_occurrences_each_property, sub_num_occurrences, "num_sides", obj_num_sides)
            # Update "num_corners"
            update_occurrences(num_occurrences_each_property, sub_num_occurrences, "num_corners", obj_num_corners)
            # Update "area"
            update_occurrences(num_occurrences_each_property, sub_num_occurrences, "area", obj_area)
            # Update "object_gaps"
            if "object_gaps" in num_occurrences_each_property:
                gap_len_list = sorted(len(gaps) for gaps in obj_gaps)
                update_occurrences(num_occurrences_each_property, sub_num_occurrences, "object_gaps", str(gap_len_list))
        ###loop through sub number of object occurrences and look for object properties that have ALL properties equal the same value, no diversity. 
        ##i.e, all objects in grid are brown, all have 4 shapes, etc
        for prop in sub_num_occurrences:
            for values in sub_num_occurrences[prop]:
                num_occurrences = sub_num_occurrences[prop][values]
                if num_occurrences == len(input_grid_obj.get_object_list()):
                    if prop in num_occurrences_each_property:
                        del num_occurrences_each_property[prop]
    return num_occurrences_each_property


##takes in the input grids and output grids for this problem and, based on which properties change across the various 
##input output grid pairs, determines which are the possible actions
##returns a pruned list of all of the possible actions that it makes sense searching over
def prune_action_set(input_grid_objs, output_grid_objs):
    ###feature matching. determine which objects in the input grid match with objects in the output_grid, and with what
    ##loop through input grids in input_grid_obj
    valid_actions_list = []
    for x in range(len(input_grid_objs)):
        input_grid_obj = input_grid_objs[x]
        output_grid_obj = output_grid_objs[x]
        ##pick an object in the input grid
        pixel_count_input = np.sum(input_grid_obj.get_grid() != input_grid_obj.get_background_color())
        pixel_count_output = np.sum(output_grid_obj.get_grid() != output_grid_obj.get_background_color())
        object_count_input = len(input_grid_obj.get_object_list())
        object_count_output = len(output_grid_obj.get_object_list())
        ##if there are fewer objects in the output grid than in the input grid and pixel count is smaller,
        ##that means object(s) were deleted
        if (pixel_count_input > pixel_count_output) and (object_count_input > object_count_output):
            valid_actions_list.append("delete")
        ##if there are fewer active pixels in the output than in the input BUT not fewer objects
        elif (pixel_count_input > pixel_count_output):
            valid_actions_list.append("remove_obj_portion")
        ##if there are more active pixels in the output than the input but not necessarily more objects, 
        #extend object probably occurred
        elif (pixel_count_input < pixel_count_output) and (not (object_count_output > object_count_input)):
            valid_actions_list.append("extend")
        for input_obj in input_grid_obj.get_object_list():
            # print("on input object ")
            # print(input_obj.get_object_grid())
            ##loop through objects in output_grid
            input_color = input_obj.get_object_colors()
            input_color.sort()
            input_size = input_obj.get_area()
            input_num_sides = input_obj.get_num_sides()
            input_num_corners = input_obj.get_num_corners()
            input_center = input_obj.get_object_center()
            input_obj_gap_len = len(input_obj.get_object_gaps())
            for output_obj in output_grid_obj.get_object_list():
                # print("on output object ")
                # print(output_obj.get_object_grid())
                ##only process output_obj if it is NOT in the input grid already
                if ((not input_grid_obj.is_in_object_list(output_obj)) or (np.array_equal(input_obj.get_object_grid(), output_obj.get_object_grid()))):
                    ##track which features in output grid match with input grid
                    output_color = output_obj.get_object_colors()
                    output_color.sort()
                    output_size = output_obj.get_area()
                    output_num_sides = output_obj.get_num_sides()
                    output_num_corners = output_obj.get_num_corners()
                    output_center = output_obj.get_object_center()
                    output_obj_gap_len = len(output_obj.get_object_gaps())
                    ##we found an exact match, so one of the actions at least should be keeping the object as is
                    if np.array_equal(input_obj.get_object_grid(), output_obj.get_object_grid()):
                        valid_actions_list.append("keep_as_is")
                    ##check if shape is the same between input and output object
                    elif (input_num_sides == output_num_sides) and (input_num_corners == output_num_corners) and (input_size == output_size) and (input_obj_gap_len == output_obj_gap_len):
                        ##normalize location and color by zooming in and changing color
                        color_normalized_input = change_color(input_obj.get_object_grid(), 1)
                        zoomed_input = zoom_in(color_normalized_input)
                        color_normalized_output = change_color(output_obj.get_object_grid(), 1)
                        zoomed_output = zoom_in(color_normalized_output)
                        ##if objects are equal before normalizing for color but NOT before, then change color was an action taken
                        if np.array_equal(zoomed_input, zoomed_output) and (input_color != output_color):
                            valid_actions_list.append("change_color")
                        ##if we find a size and shape match but the locations are different, check 3 cases. rotate, translate, and flip.
                        if (input_center != output_center):
                            # print("location was not in the same place for the matches ")
                            ##first case. shapes are equal. sizes are equal. locations are NOT equal but the location normalized 
                            ##objects are equal (this means orientations are equal --> translate is a valid action)
                            if np.array_equal(zoomed_input, zoomed_output):
                                valid_actions_list.append("translate")
                            else:
                                ###perform flipping and rotating on input object to see if any of them work. if we get a match, add flip or rotate
                                ##to the action list. If not, do not add it.
                                flipped_obj_list = []
                                flipped_obj_list.append(flip(obj = input_obj.get_object_grid(), direction = "ud", axis = input_obj.get_top_row()))
                                flipped_obj_list.append(flip(obj = input_obj.get_object_grid(), direction = "ud", axis = input_obj.get_bottom_row()))
                                flipped_obj_list.append(flip(obj = input_obj.get_object_grid(), direction = "ud", axis = input_obj.get_top_row()))
                                flipped_obj_list.append(flip(obj = input_obj.get_object_grid(), direction = "ud", axis = input_obj.get_bottom_row()))
                                found = False
                                for flipped_obj in flipped_obj_list:
                                    flipped_obj = change_color(zoom_in(flipped_obj), 1)
                                    if np.array_equal(flipped_obj, zoomed_output):
                                        valid_actions_list.append("flip")
                                        found = True
                                if found:
                                    continue
                                else:
                                    rotated_obj_list = []
                                    for angle in [90, 180, 270]:
                                        rotated_obj = change_color(zoom_in(rotate_in_place(input_obj.get_object_grid(), angle)), 1)
                                        if np.array_equal(rotated_obj, zoomed_output):
                                            valid_actions_list.append("rotate_in_place")
                    ##check for location overlap between two object arrays (output object is at least partly located where the input object was)
                    ##check IFF the shapes of input object and output object does NOT match. only considering cases where overlapping objects
                    ##are same color
                    # elif (np.any(np.bitwise_and(input_obj.get_object_grid(), output_obj.get_object_grid()))) and (input_color == output_color):
                    #     if output_size > input_size:
                    #         valid_actions_list.append("extend")
                    #     elif output_size < input_size:
                    #         valid_actions_list.append("remove_obj_portion")
        ###final check. if the number of objects in the output is greater but no "flip", "translate", "extend", "remove_obj_portion" or "rotate" occurred (meaning)
        ##that no matching feature copying occurred, then we can say that a brand new object should be added
        # elements = ["flip", "rotate", "translate", "extend", "remove_obj_portion"]
        # if (object_count_input < object_count_output) and (all(element not in valid_actions_list for element in elements)):
        #     valid_actions_list.append("add_new_object")
    return list(set(valid_actions_list))





###gets all possible unary action checkers in a list, used for enumerative program search
def get_unary_obj_checker_domain(input_grid_objs, properties_set):
    num_occurrences_each_property = properties_set
    possible_obj_checkers_strong = []
    all_possible_obj_checkers = []
    for keys in num_occurrences_each_property:
        ##looking at all listed possible values for all properties
        for property_values in num_occurrences_each_property[keys]:
            ##gets the number of times it occurs
            num_occurrences = num_occurrences_each_property[keys][property_values]
            ##gets the object checker that checks if object property equals ______ possible value
            if keys == "dominant_color":
                checker = "(obj.get_dominant_color() == {})".format(property_values)
            # elif keys == "color_list":
            #     checker = "(obj.get_object_colors() == {})".format(property_values)
            elif keys == "num_sides":
                checker = "(obj.get_num_sides() == {})".format(property_values)
            # elif keys == "num_corners":
            #     checker = "(obj.get_num_corners() == {})".format(property_values)
            elif keys == "area":
                checker = "(obj.get_area() == {})".format(property_values)
            elif keys =="object_gaps":
                number_gaps = len(ast.literal_eval(property_values))
                checker = "(len(obj.get_object_gaps()) == {})\n".format(number_gaps)
            ##if the property num occurrences equals the number of total grids, then it PROBABLY occurs in all of the grids
            ##if so, it could be a pattern worth focusing on. I.E, if there is a red object of a certain shape in all grids,
            ##the rule could be related to that 
            if num_occurrences >= len(input_grid_objs):
                print("property {} occurs {} times ".format(property_values, num_occurrences))
                ##if we are dealing with color property checker, add its negation too. 
                if 'dominant_color' in checker:
                    new_checker = "(obj.get_dominant_color() != {})".format(property_values)
                    possible_obj_checkers_strong.append(new_checker)
                    all_possible_obj_checkers.append(new_checker)
                possible_obj_checkers_strong.append(checker)
                all_possible_obj_checkers.append(checker)
            else:
                all_possible_obj_checkers.append(checker)
    ##adding object number checkers to the list of possible object checkers
    ##first, determine if input_grid_objs all have the same number of objects. Then, the object number may actually be meaningful.
    all_lengths_same = all(len(input_grid_obj.get_object_list()) == len(input_grid_objs[0].get_object_list()) for input_grid_obj in input_grid_objs)
    if all_lengths_same:
        ##add a checker for each possible object number
        for x in range(len(input_grid_objs[0].get_object_list())):
            checker = "(obj.get_object_number() == {})".format(x + 1)
            possible_obj_checkers_strong.append(checker)
    return possible_obj_checkers_strong, all_possible_obj_checkers


def constrain_obj_checker_domain_llm(llm_obj_checkers, obj_checker_domain):
    properties_to_check = []
    for obj_checker in llm_obj_checkers:
        if 'sides' in obj_checker:
            properties_to_check.append('num_sides')
        elif 'color' in obj_checker:
            properties_to_check.append('dominant_color')
        elif 'gap' in obj_checker:
            properties_to_check.append('object_gaps')
        elif 'area' in obj_checker or 'size' in obj_checker:
            properties_to_check.append('area')
    properties_to_check = list(set(properties_to_check))
    new_obj_checker_domain = []
    for prop in properties_to_check:
        for obj_checker in obj_checker_domain:
            if prop == "object_gaps":
                if 'gap' in obj_checker:
                    new_obj_checker_domain.append(obj_checker)
            elif prop == "num_sides":
                if 'sides' in obj_checker:
                    new_obj_checker_domain.append(obj_checker)
            elif prop == "dominant_color":
                if 'dominant_color' in obj_checker:
                    new_obj_checker_domain.append(obj_checker)
            elif prop == "area":
                if "area" in obj_checker:
                    new_obj_checker_domain.append(obj_checker)
    return new_obj_checker_domain





##samples obj checker given the number the obj checker should have and the input grid objects to determine the properties 
##to include in the obj checker. Samples object checkers that check object property against a single value, without accounting
##for other objects or grid level information
def sample_obj_checker_unary(input_grid_objs, properties_set):
    possible_obj_checkers_strong, all_possible_obj_checkers = get_unary_obj_checker_domain(input_grid_objs, properties)
    chosen_line = ""
    if len(possible_obj_checkers_strong) == 0:
        choice = 2
    else:
        choice = 1
    if choice == 1:
        chosen_line = random.choice(possible_obj_checkers_strong)
        return chosen_line
    elif choice == 2:
        ##handle case where all possible checker length list is 0
        if len(all_possible_obj_checkers) == 0:
            return 'obj.get_object_number() == 1'
        chosen_line = random.choice(all_possible_obj_checkers)
        return chosen_line

##gets a list of all possible grid related obj checkers, used for enumerative search
def get_grid_related_obj_checker_domain():
    domain = [
"True",
"(obj.get_area() == max(input_grid_obj.get_all_object_sizes()))",
"(obj.get_area() == min(input_grid_obj.get_all_object_sizes()))",
"(obj.get_dominant_color() == max(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
"(obj.get_dominant_color() == min(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
"(obj.get_top_row() == input_grid_obj.get_top_most_obj_row())",
"(obj.get_bottom_row() == input_grid_obj.get_bottom_most_obj_row())",
"(obj.get_left_column() == input_grid_obj.get_left_most_obj_col())",
"(obj.get_right_column() == input_grid_obj.get_right_most_obj_col())"
    ]
    return domain
    
##samples object checkers that check grid related attributes. I.E, is the object the largest object, 
##smallest object, is it of the most frequently occurring color, etc
##takes in list of input_grid_obj objects and returns the sampled object checker
def sample_obj_checkers_grid_related():
    domain = ["True",
"(obj.get_area() == max(input_grid_obj.get_all_object_sizes()))",
"(obj.get_area() == min(input_grid_obj.get_all_object_sizes()))",
"(obj.get_dominant_color() == max(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
"(obj.get_dominant_color() == min(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
"(obj.get_top_row() == input_grid_obj.get_top_most_obj_row())",
"(obj.get_bottom_row() == input_grid_obj.get_bottom_most_obj_row())",
"(obj.get_left_column() == input_grid_obj.get_left_most_obj_col())",
"(obj.get_right_column() == input_grid_obj.get_right_most_obj_col())",
"(obj.get_object_center() == input_grid_obj.get_center())"
    ]
    return random.choice(domain)

###samples a possible object checker that does its checking with respect  to a reference object
##Ex. if object is aligned with nearby red object, etc
def sample_obj_checker_reference_related(input_grid_objs, properties_set):
    sampled_relation = random.choice(["==", ">", "<"])
    domain = [
        "obj.get_dominant_color() == reference_obj.get_dominant_color()",
        "obj.get_object_colors() == reference_obj.get_object_colors()",
        "obj.get_area() {} reference_obj.get_area()".format(sampled_relation),
        "obj.get_object_center()[0] {} reference_obj.get_object_center()[0]".format(sampled_relation),
        "obj.get_object_center()[1] {} reference_obj.get_object_center()[1]".format(sampled_relation),
        "obj.get_top_row() {} reference_obj.get_top_row()".format(sampled_relation),
        "obj.get_bottom_row() {} reference_obj.get_bottom_row()".format(sampled_relation),
        "obj.get_left_column() {} reference_obj.get_left_column()".format(sampled_relation),
        "obj.get_right_column() {} reference_obj.get_right_column()".format(sampled_relation),
        "obj.get_num_sides() == reference_obj.get_num_sides()",
        "obj.get_num_corners() == reference_obj.get_num_corners()",
        "(np.array_equal(obj.get_sub_pattern(), reference_obj.get_sub_pattern()))"
    ]
    sampled_query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_line = random.choice(domain)
    obj_checker_line = """query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
{}
""".format(sampled_query, sampled_line)
    return obj_checker_line

##calls both simple obj checker and grid obj checker domain functions to create an overall obj checker domain
def get_total_obj_checker_domain(input_grid_objs, properties_set, constrain = False):
    if constrain:
        obj_checkers_simple = get_unary_obj_checker_domain(input_grid_objs, properties_set)[0]
    else:
        obj_checkers_simple = get_unary_obj_checker_domain(input_grid_objs, properties_set)[1]
    obj_checkers_grid = get_grid_related_obj_checker_domain()
    return obj_checkers_grid + obj_checkers_simple

###randomly calls one of the three possible object checker forms
def sample_obj_checker(input_grid_objs, properties_set):
    choices = random.randint(1,2)
    if choices == 1:
        return sample_obj_checker_unary(input_grid_objs = input_grid_objs, properties_set = properties_set)
    elif choices == 2:
        return sample_obj_checkers_grid_related()
    # elif choices == 3:
    #     return sample_obj_checker_reference_related(input_grid_objs = input_grid_objs, properties_set = properties_set)


###string manipulation to make query obj checkers fit the format
def process_query_obj_checker(number, query_obj_checker):
    return_str = ""
    for line in query_obj_checker.splitlines():
        if 'query' in line:
            return_str += line + '\n'
            continue
        elif 'find_obj' in line:
            return_str += line + '\n'
            continue
        ##make sure obj checker is not being assigned to a blank line
        elif len(line) > 0:
            return_str += "obj_checker_{} = ".format(number) + line + '\n'
            continue
    return return_str



##returns a new sampled trans function
##input: number of the function, to set its name properly
#input_grid_objs: the input grids, to determine proper values to include in queries in the transformation function
#output_grid_objs. output grids. can be used in heuristics to determine which actions to take
def sample_trans_function(number, input_grid_objs, output_grid_objs, action_set, properties_set):
    ###randomly select a transformation from action set, that is the transformation we are using for 
    ##sampling the trans function
    function_body = ""
    ##if action set length is 0 for some reason, instantiate it to be equal to the full action_set, no pruning
    if len(action_set) == 0:
        action_set = ["keep_as_is" ,"flip", "translate", "extend", "change_color", "remove_obj_portion", "rotate_in_place"]
    chosen_action = random.choice(action_set)
    if chosen_action == "keep_as_is":
        function_body = "output_grid_obj.add_obj(obj)"
    elif chosen_action == "delete":
        function_body = ""
    elif chosen_action == "remove_obj_portion":
        function_body = sample_remove_obj_portion_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action == "extend":
        function_body = sample_extend_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action == "change_color":
        function_body = sample_change_color_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action == "translate":
        function_body = sample_translate_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action == "flip":
        function_body = sample_flip_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action =="rotate_in_place":
        #####NEED TO WRITE ROTATE IN PLACE SAMPLERS
        function_body = sample_rotate_in_place_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif chosen_action == "composite":
        function_body = sample_composite_trans_code(input_grid_objs = input_grid_objs, action_set = action_set, properties_set = properties_set) + add_create_and_add_new_object_string()
    ##indent function body
    function_body = indent(function_body, 4)
    function_str = """def transformation_{}(input_grid_obj, output_grid_obj, obj):
{}
    return output_grid_obj""".format(number, function_body)
    return function_str


###get all possible trans functions for a given task. 
def get_trans_function_domain(number, input_grid_objs, output_grid_objs, action_set, properties_set):
    possible_functions = []
    ###two cases. 1 where object is added to list as is, another where only the transformation is added.
    ##loop through the actions we are working with
    all_trans_snippets = []
    for action in action_set:
        if action == "translate":
            print("added translate to domain ")
            all_trans_snippets += get_translate_domain(input_grid_objs = input_grid_objs, properties_set = properties_set)
        elif action == "extend":
            print("added extend to domain ")
            all_trans_snippets += get_extend_domain(input_grid_objs = input_grid_objs, output_grid_objs = output_grid_objs, properties_set = properties_set)
        elif action == "flip":
            print("added flip to domain ")
            all_trans_snippets += get_flip_domain(input_grid_objs = input_grid_objs, properties_set = properties_set)
        elif action == "rotate_in_place":
            print("added rotate in place to domain ")
            all_trans_snippets += get_rotate_in_place_domain(input_grid_objs = input_grid_objs, properties_set = properties_set)
        elif action == "change_color":
            print("added change color to domain ")
            all_trans_snippets += get_color_change_domain(input_grid_objs = input_grid_objs, output_grid_objs = output_grid_objs, properties_set = properties_set)
    for trans_snippet in all_trans_snippets:
        if trans_snippet != "" and "GridObject" not in trans_snippet:
            trans_snippet = trans_snippet + add_create_and_add_new_object_string()
        function_str_1 = """def transformation_{}(input_grid_obj, output_grid_obj, obj):
    output_grid_obj.add_obj(obj)
{}
    return output_grid_obj""".format(number, indent(trans_snippet, 4))
        function_str_2 = """def transformation_{}(input_grid_obj, output_grid_obj, obj):
{}
    return output_grid_obj""".format(number, indent(trans_snippet, 4))
        possible_functions.append(function_str_1)
        possible_functions.append(function_str_2)    
    return possible_functions


##first possible mutation. increase the number of object checkers and transformations in the program,
##makes it "consider more cases", more "fine grained" reasoning, not just 1 obj_checker transformation case covering 
##all instances. Can be thought of as an added branch in a logic/decision tree. 
##inputs:
##program_str: string representing rule program
##input_grid_objs. takes in list of input grid objects
##output_grid_objs. takes in list of output grid objects
def add_checker_transformation_block(program, input_grid_objs, output_grid_objs, action_set, properties_set):
    max_number_of_objects = -1
    ##finding the most number of objects in the input grids to have an upper bound of obj checker/transformation blocks to have
    for input_grid_obj in input_grid_objs:
        num_objects = len(input_grid_obj.get_object_list())
        if num_objects > max_number_of_objects:
            max_number_of_objects = num_objects
    ##finding current number of object checker/transformation pairs in object
    current_number_obj_checkers = len(program.get_obj_checkers())
    ##add new block if and only if you are under the upper bound of conditional statements
    number = current_number_obj_checkers + 1
    if number <= max_number_of_objects:
        old_params = program.get_parameters()
        new_params = copy.deepcopy(old_params)
        new_obj_checker = sample_obj_checker(input_grid_objs = input_grid_objs, properties_set = properties_set)
        new_trans_function = sample_trans_function(number = number, input_grid_objs = input_grid_objs, output_grid_objs = output_grid_objs, action_set = action_set, properties_set = properties_set)
        ##if we are dealing with a non query based single line obj checker, add it with the assignment string appended
        #make sure length of new_obj_checker is greater than 0
        if ('query' not in new_obj_checker) and (len(new_obj_checker) > 0):
            new_params["obj_checkers"].append("obj_checker_{} = ".format(number) + new_obj_checker)
        ##if multi line obj checker, pre process it first and then add
        elif 'query' in new_obj_checker:
            query_obj_checker = process_query_obj_checker(number = number, query_obj_checker = new_obj_checker)
            new_params["obj_checkers"].append(query_obj_checker)
        new_params["trans_functions"].append(new_trans_function)
        new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number= program.get_problem_number(), parameters=new_params)
        return new_program
    return program

###inserts a predefined obj checker + transformation pair into a program
def insert_checker_trans_enumerative_search(program, input_grid_objs, number, obj_checker, trans_function):
    max_number_of_objects = -1
    ##finding the most number of objects in the input grids to have an upper bound of obj checker/transformation blocks to have
    for input_grid_obj in input_grid_objs:
        num_objects = len(input_grid_obj.get_object_list())
        if num_objects > max_number_of_objects:
            max_number_of_objects = num_objects
    ##finding current number of object checker/transformation pairs in object
    current_number_obj_checkers = len(program.get_obj_checkers())
    if current_number_obj_checkers > max_number_of_objects:
        return
    old_params = program.get_parameters()
    new_params = copy.deepcopy(old_params)
    if (number > len(old_params["obj_checkers"])) and (number > len(old_params["trans_functions"])):
        new_params["obj_checkers"].append("obj_checker_{} = ".format(number) + obj_checker)
        new_params["trans_functions"].append(trans_function)
    else:
        new_params["obj_checkers"][number - 1] = "obj_checker_{} = ".format(number) + obj_checker
        new_params["trans_functions"][number - 1] = trans_function
    new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number= program.get_problem_number(), parameters=new_params)
    return new_program

##second mutation case: changing the first order logic specified in the object checker
##takes in program and randomly mutates a specified or randomly selected object checker
def mutate_obj_checker(program, input_grid_objs, properties_set, number = None):
    old_params = program.get_parameters()
    ##if there are no object checkers to mutate, return the old program
    if len(old_params["obj_checkers"]) == 0:
        return program
    new_params = copy.deepcopy(old_params)
    if number is None:
        number = random.randint(1, len(old_params["obj_checkers"]))
    ###if the first obj checker is setting directly equal to True, make that the one you mutate
    if 'True' in old_params["obj_checkers"][0]:
        number = 1
    new_obj_checker = sample_obj_checker(input_grid_objs = input_grid_objs, properties_set = properties_set)
    index = number - 1
    for x in range(len(old_params["obj_checkers"])):
        obj_checkers = old_params["obj_checkers"][x]
        if 'obj_checker_{}'.format(number) in obj_checkers:
            index = x
    ##if it is not a query obj checker with multiple lines, add the single line
    ##make sure that the sampled checker isn't blank
    if ('query' not in new_obj_checker) and (len(new_obj_checker) > 0):
        new_params["obj_checkers"][index] = "obj_checker_{} = ".format(number) + new_obj_checker
    ##if we have multiple lines for query based obj checker, process then add
    elif 'query' in new_obj_checker:
        query_obj_checker_str = process_query_obj_checker(number = number, query_obj_checker = new_obj_checker)
        new_params["obj_checkers"][index] = query_obj_checker_str
    new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
    return new_program

##third mutation case: increases the number of object mutations that occur on the program
##takes in program and adds a block of code that ADDs an object to the output_grid_obj
##ADD block involves adding a transformed version of an object to the output
def add_trans_object_block(program, input_grid_objs, output_grid_objs, action_set):
    ##if there are no trans functions to add block to, return the original program
    if len(program.get_trans_functions()) == 0:
        return program
    number_add_blocks = program.get_number_additions()
    max_num_objects_output = -1
    ###set average number of objects in output and max number of objects in output as heuristic for how many objects can be added
    for output_grid_obj in output_grid_objs:
        if len(output_grid_obj.get_object_list()) > max_num_objects_output:
            max_number_of_objects = len(output_grid_obj.get_object_list())
    ##only sample and insert a transform + add block if the number of add statements currently in the program
    ##is lower than the upper bound, which is the maximum number of objects in any of the outputs, period.
    if number_add_blocks < max_num_objects_output:
        random_index = random.randint(0, len(program.get_trans_functions()) - 1)
        chosen_trans_function = program.get_trans_functions()[random_index]
        new_trans_block = insert_add_object_block(trans_function = chosen_trans_function, input_grid_objs = input_grid_objs, action_set = action_set, current_transformations_list = program.get_transformation_list())
        new_params = copy.deepcopy(program.get_parameters())
        new_params["trans_functions"][random_index] = new_trans_block
        new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
        return new_program
    return program


##takes in trans function string and adds a transformation + add snippet of code to it
##input: string representation of function
##output: string representation of modified function
def insert_add_object_block(trans_function, input_grid_objs, action_set, current_transformations_list = None):
    new_trans_function = ""
    lines_list = trans_function.splitlines()
    for x in range(len(lines_list)):
        line = lines_list[x]
        if x + 1 < len(lines_list):
            next_line = lines_list[x + 1]
            if "return output_grid_obj" in next_line:
                ##sample transformation and addition code and add it
                new_code = line + '\n' + indent(sample_simple_transformation_code(input_grid_objs = input_grid_objs, action_set = action_set, properties_set = properties_set, current_transformations_set = current_transformations_list), 4)
                new_trans_function += new_code + '\n'
                continue
        new_trans_function += line + '\n'
    return new_trans_function


###creates and returns string that adds transformed object to grid
def add_create_and_add_new_object_string():
    return """
transformed_obj = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid)
output_grid_obj.add_obj(transformed_obj)
    """
    
##samples possible simple transformation blocks to add to the transformation function. Simple transformation means that 
##exactly one transformation occurs on the INPUT OBJECT and added to the output grid object. 
##returns string implementing such a transformation
def sample_simple_transformation_code(input_grid_objs, properties_set, action_set, current_transformations_list = None):
    if current_transformations_list is None or len(current_transformations_list) == 0:
        if len(action_set) == 0:
            current_transformations_list = ["keep_as_is" ,"flip", "translate", "extend", "change_color", "remove_obj_portion", "rotate_in_place"]
        else:
            current_transformations_list = action_set
    else:
        current_transformations_list = list(set(current_transformations_list + action_set))
    transformation = random.choice(current_transformations_list)
    if transformation == "keep_as_is":
        return "transformed_obj = obj"
    elif transformation == "flip":
        return sample_flip_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif transformation == "translate":
        return sample_translate_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif transformation == "extend":
        return sample_extend_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif transformation == "change_color":
        return sample_change_color_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif transformation == "remove_obj_portion":
        return sample_remove_obj_portion_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()
    elif transformation == "rotate_in_place":
        return sample_rotate_in_place_code(input_grid_objs = input_grid_objs, properties_set = properties_set) + add_create_and_add_new_object_string()

###gets domain of "general" queries, queries that would be valid for all problems and doesnt rely on individual properties
def get_general_query_domain():
    query_list = [
    "('area', 'equals', input_grid_obj.get_largest_object().get_area())",  "('area', 'equals', input_grid_obj.get_smallest_object().get_area())", 
    "('object_color', 'equals', max(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
    "('object_color', 'equals', min(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get))",
    "('top_row', 'equals', input_grid_obj.get_top_most_obj_row())",
    "('bottom_row', 'equals', input_grid_obj.get_bottom_most_obj_row())",
    "('left_column', 'equals', input_grid_obj.get_left_most_obj_col())",
    "('right_column', 'equals', input_grid_obj.get_right_most_obj_col())",
    "('object_center[0]', 'equals', obj.get_object_center()[0])",
    "('object_center[1]', 'equals', obj.get_object_center()[1])",
    "('object_color', 'equals', obj.get_dominant_color())"
    ]
    return query_list

##sampler built off of previous set of query strings
def sample_general_queries():
    query_list = get_general_query_domain()
    return random.choice(query_list)


###gets domain of property based queries 
def get_simple_query_domain(input_grid_objs, properties_set):
    num_occurrences_each_property = properties_set
    possible_queries_strong = []
    all_possible_queries = []
    for keys in num_occurrences_each_property:
        ##looking at all listed possible values for all properties
        for property_values in num_occurrences_each_property[keys]:
            ##gets the number of times it occurs
            num_occurrences = num_occurrences_each_property[keys][property_values]
            ##gets the object checker that checks if object property equals ______ possible value
            if keys == "dominant_color":
                query = "('object_color', 'equals', {})".format(property_values)
            # elif keys == "color_list":
            #     query = "('color_list', 'equals', {})".format(property_values)
            elif keys == "num_sides":
                query = "('num_sides', 'equals', {})".format(property_values)
            # elif keys == "num_corners":
            #     query = "('num_corners', 'equals', {})".format(property_values)
            elif keys == "area":
                query = "('area', 'equals', {})".format(property_values)
            ##if the property num occurrences equals the number of total grids, then it PROBABLY occurs in all of the grids
            ##if so, it could be a pattern worth focusing on. I.E, if there is a red object of a certain shape in all grids,
            ##the rule could be related to that 
            if num_occurrences == len(input_grid_objs):
                possible_queries_strong.append(query)
            else:
                all_possible_queries.append(query)
    ##adding object number checkers to the list of possible object checkers
    ##first, determine if input_grid_objs all have the same number of objects. Then, the object number may actually be meaningful.
    all_lengths_same = all(len(input_grid_obj.get_object_list()) == len(input_grid_objs[0].get_object_list()) for input_grid_obj in input_grid_objs)
    if all_lengths_same:
        ##add a checker for each possible object number
        for x in range(len(input_grid_objs[0].get_object_list())):
            query = "('object_number', 'equals', {})".format(x + 1)
            possible_queries_strong.append(query)
    return possible_queries_strong, all_possible_queries


##combines both domains and gets a domain for all queries
def get_total_query_domain(input_grid_objs, properties_set):
    return get_simple_query_domain(input_grid_objs, properties_set)[0] + get_general_query_domain()

##samples possible query values based on what exists in the input_grid_objs 
##query MUST be of a unary check format. I.E it must check for an intrinsic attribute of an object (is it red) and not 
##properties that are with respect to something else.
def sample_simple_query(input_grid_objs, properties_set):
    possible_queries_strong, all_possible_queries = get_simple_query_domain(input_grid_objs, properties_set)
    if len(possible_queries_strong) == 0:
        choice = 2
    else:
        choice = 1
    if choice == 1:
        ###make sure that the sampled query isn't something that we know causes errors
        chosen_query = random.choice(possible_queries_strong)
        final_query = "[{}]".format(chosen_query)          
    elif choice == 2:
        ##handle case where all possible queries are empty for some reason
        if len(all_possible_queries) == 0:
            return "[('object_number', 'equals', 1)]"
        ##make sure sampled query isnt already known to cause errors
        chosen_query = random.choice(all_possible_queries)
        final_query = "[{}]".format(chosen_query)
    return final_query


###gets all possible landmarks in a single domain
def get_domain_landmark():
    domain_list_1 = ["input_grid_obj.get_center()", "(input_grid_obj.get_shape()[0] - 1, 0)", 
                    "(0, input_grid_obj.get_shape()[1] - 1)", "(input_grid_obj.get_shape()[0] - 1, input_grid_obj.get_shape()[1] - 1)"]
    ##case 2. list of possible landmark strings for object level landmarks
    domain_list_2 = ["reference_obj.get_object_center()"]
    return domain_list_2

    


##returns possible landmark strings. Assumes we have a reference object variable available
def sample_landmarks(case = None):
    if case is None:
        case = random.randint(1,2)
    ##case 1. list of possible landmarks strings for grid level landmarks
    if case == 1:
        ##can add more cases if needed
        domain_list = ["input_grid_obj.get_center()", "(0, 0)", "(input_grid_obj.get_shape()[0], 0)", 
                    "(0, input_grid_obj.get_shape()[1] - 1)", "(input_grid_obj.get_shape()[0], input_grid_obj.get_shape()[1] - 1)",
                    "(input_grid_obj.get_center()[0], 0)", "(0, input_grid_obj.get_center()[1])", "(input_grid_obj.get_shape()[0] - 1, 0)",
                    "(input_grid_obj.get_shape()[0] - 1, input_grid_obj.get_shape()[1] - 1)", "(0, input_grid_obj.get_shape()[1] - 1)",
                    "(input_grid_obj.get_center()[0], input_grid_obj.get_shape()[1] - 1)", "(input_grid_obj.get_shape()[0] - 1, input_grid_obj.get_center()[1])"]
    ##case 2. list of possible landmark strings for object level landmarks
    if case == 2:
        ###ADD MORE CASES
        color_domain = ["obj.get_dominant_color()", "list(set(obj.get_object_colors()) & set(reference_obj.get_object_colors()))[0]"]
        sampled_color = random.choice(color_domain)
        domain_list = ["reference_obj.get_object_center()", 
                       "(reference_obj.get_top_row(), reference_obj.get_object_center()[1])",
                       "(reference_obj.get_bottom_row(), reference_obj.get_object_center()[1])",
                       "(reference_obj.get_object_center()[0], reference_obj.get_left_column())",
                       "(reference_obj.get_object_center()[0], reference_obj.get_right_column())",
                       "(reference_obj.get_object_gaps()[0][0])",
"reference_obj.find_cell_color(color = {}, equals = True)[0]".format(sampled_color)]
    
    sampled_landmark = random.choice(domain_list)
    return sampled_landmark













##generates lines of code that determine the axis of rotation parameter for a flip transformation
##samples a query, sets a landmark based on reference object, and uses landmark to determine axis of flipping
##returns string representing lines of code that set axis in flip_params dictionary. Assumes variable name for parameters dict
##is flip_params
def sample_flip_axis(input_grid_objs, direction, properties_set, query = None):
    ##can also make object rotate about its own top/bottom row or left/right column
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_landmark = sample_landmarks()
    ##default axis value to something
    axis_value = "landmark[0]"
    if direction == "ud":
        axis_value = random.choice(["landmark[0]", "input_grid_obj.get_top_row()", "input_grid_obj.get_bottom_row()"])
    elif direction == "lr":
        axis_value = random.choice(["landmark[1]", "input_grid_obj.get_left_column()", "input_grid_obj.get_right_column()"])
    flip_axis_code = """query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
landmark = {}
flip_params['axis'] = {}""".format(query, sampled_landmark, axis_value)
    return flip_axis_code



##assumes we are performing a single flip on the obj GridObject and adding it to the output_grid_obj
##takes in input_grid_objs to determine which parameters are possible
##returns string that contains code that 1. determines flip parameters. 2. performs the flip on obj. 3. adds flipped obj to output_grid_obj
def sample_flip_code(input_grid_objs, properties_set, query = None):
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_direction = random.choice(["lr", "ud"])
    sampled_axis = sample_flip_axis(input_grid_objs = input_grid_objs, direction = sampled_direction, properties_set = properties_set, query = query)
    flip_params_line = """flip_params = {{}}
flip_params['direction'] = '{}'
{}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'flip', params = flip_params)
""".format(sampled_direction, sampled_axis)
    return flip_params_line



###gets a list of all possible flip implementations in a list, for enumerative search
def get_flip_domain(input_grid_objs, properties_set):
    possible_flip_implementations = []
    for direction in ["lr", "ud"]:
        for queries in get_total_query_domain(input_grid_objs, properties_set):
            for landmark in get_domain_landmark():
                if direction == "lr":
                    axes = ["landmark[0]", "input_grid_obj.get_top_row() - 1", "input_grid_obj.get_bottom_row() + 1"]
                elif direction == "ud":
                    axes = ["landmark[1]", "input_grid_obj.get_left_column() - 1", "input_grid_obj.get_right_column() + 1"]
                for possible_axis in axes:
                    flip_axis_code = """query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
landmark = {}
flip_params['axis'] = {}"""
                    flip_axis_code = flip_axis_code.format(queries, landmark, possible_axis)
                    flip_params_line = """flip_params = {{}}
flip_params['direction'] = '{}'
{}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'flip', params = flip_params)"""
                    flip_params_line = flip_params_line.format(direction, flip_axis_code)
                    possible_flip_implementations.append(flip_params_line)
    return possible_flip_implementations
















###samples a potential value for dx to translate object by
def sample_dx():
    domain = ["landmark[0] - obj.get_object_center()[0]", 
              "(abs(landmark[0] - obj.get_object_center()[0]))/(landmark[0] - obj.get_object_center()[0]) if landmark[0] - obj.get_object_center()[0] != 0 else 0",
              "landmark[0] - obj.get_bottom_row()", "landmark[0] - obj.get_top_row()"]
    return random.choice(domain)

###samples a potential value for dy to translate object by
def sample_dy():
    domain = ["landmark[1] - obj.get_object_center()[1]", 
              "(abs(landmark[1] - obj.get_object_center()[1]))/(landmark[1] - obj.get_object_center()[1]) if landmark[1] - obj.get_object_center()[1] != 0 else 0",
              "landmark[1] - obj.get_left_column()", "landmark[1] - obj.get_right_column()"]
    return random.choice(domain)

##takes in the input grids for this problem and samples a translate code section 
def sample_translate_code(input_grid_objs, properties_set, query = None):
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_landmark = sample_landmarks()
    sampled_dx = sample_dx()
    sampled_dy = sample_dy()
    translate_params_line = """translate_params = {{}}
query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
landmark = {}
translate_params['dx'] = {}
translate_params['dy'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'translate', params = translate_params)
""".format(query, sampled_landmark, sampled_dx, sampled_dy)
    return translate_params_line


###get possible translate implementations in a list
def get_translate_domain(input_grid_objs, properties_set):
    possible_translate_implementations = []
    for queries in get_total_query_domain(input_grid_objs, properties_set):
        for landmark in get_domain_landmark():
            for dx in ["landmark[0] - obj.get_object_center()[0]"]:
                for dy in ["landmark[1] - obj.get_object_center()[1]"]:
                    translate_params_line = """translate_params = {{}}
query = {}
reference_obj_list = input_grid_obj.find_obj(query, obj)
for reference_obj in reference_obj_list:
    landmark = {}
    translate_params['dx'] = {}
    translate_params['dy'] = {}
    transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'translate', params = translate_params)
    transformed_obj = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid)
    output_grid_obj.add_obj(transformed_obj)"""
                    translate_params_line = translate_params_line.format(queries, landmark, dx, dy)
                    possible_translate_implementations.append(translate_params_line)
    return possible_translate_implementations












###samples an object cell to use as the cell to extend in an extend function
##either samples object specific landmark locations that have intrinsic meaning or a randomly sampled cell number
def sample_object_cell():
    choice = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
    if choice == 1:
        domain = ["obj.get_object_center()", "(obj.get_top_row(), obj.get_object_center()[1])",
                "(obj.get_bottom_row(), obj.get_object_center()[1])",
                "(obj.get_object_center()[0], obj.get_left_column())",
                "(obj.get_object_center()[0], obj.get_right_column())",
                "obj.find_cell_pos(1)"]
        return random.choice(domain)
    elif choice == 2:
        number = random.randint(1, 50)
        return "obj.find_cell_pos({})".format(number)


###samples code that performs extend action
def sample_extend_code(input_grid_objs, properties_set, query = None):
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_landmark = sample_landmarks()
    sampled_cell = sample_object_cell()
    extend_params_line = """extend_params = {{}}
query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
landmark = {}
extend_params['cell'] = {}
extend_params['direction'] = determine_direction_landmark(landmark = landmark, object_cell = {})
extend_params['distance'] = math.dist(landmark, {})
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'extend', params = extend_params)
""".format(query, sampled_landmark, sampled_cell, sampled_cell, sampled_cell)
    return extend_params_line


###gets list of possible extend implementations 
def get_extend_domain(input_grid_objs, output_grid_objs, properties_set):
    possible_extend_implementations = []
    colors_in_output = []
    for output_grid_obj in output_grid_objs:
        for output_obj in output_grid_obj.get_object_list():
            color_list = output_obj.get_object_colors()
            for color in color_list:
                if color not in colors_in_output:
                    colors_in_output.append(color)
    possible_colors = colors_in_output + ['obj.get_dominant_color()']
    for query in get_total_query_domain(input_grid_objs, properties_set):
        for landmark in get_domain_landmark():
            cell_list = ["obj.get_object_center()"]
            for cell in cell_list:
                # for color in possible_colors:
                determine_landmark_str_1 = "determine_direction_landmark(landmark = landmark, object_cell = {})".format(cell)
                determine_landmark_str_2 = "determine_direction_landmark(landmark = landmark, object_cell = {}, up_down_only = True)".format(cell)
                determine_landmark_str_3 = "determine_direction_landmark(landmark = landmark, object_cell = {}, left_right_only = True)".format(cell)
                possible_directions = [determine_landmark_str_1, determine_landmark_str_2, determine_landmark_str_3]
                for direction in possible_directions:
                    if 'up_down_only' in direction:
                        distance = "abs(landmark[0] - {}[0])".format(cell)
                    elif 'left_right_only' in direction:
                        distance = "abs(landmark[1] - {}[1])".format(cell)
                    else:
                        distance = "math.dist(landmark, {})".format(cell)
                    extend_params_line = """extend_params = {{}}
    query = {}
    reference_obj = input_grid_obj.find_obj(query, obj)[0]
    landmark = {}
    extend_params['cell'] = {}
    extend_params['direction'] = {}
    extend_params['distance'] = {}
    transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'extend', params = extend_params)
    """.format(query, landmark, cell, direction, distance)
                    possible_extend_implementations.append(extend_params_line)
    return possible_extend_implementations 
                




##sample code that implements change_color transformation on object
def sample_change_color_code(input_grid_objs, properties_set, query = None):
    colors_in_problem = []
    ##adding colors of input grid objs to color set
    for prop_dicts in properties_set['dominant_color']:
        for keys in prop_dicts:
            if keys not in colors_in_problem:
                colors_in_problem.append(keys)
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    has_query = random.randint(1,2)
    ##either query for a property and set a reference object or have that string empty
    if has_query == 1:
        query_lines = """query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]""".format(query)
    elif has_query == 2:
        query_lines = ''
    chosen_color = 1
    if len(colors_in_problem) > 0:
        chosen_color = random.choice(colors_in_problem)
    else:
        chosen_color = random.randint(1, 10)
    domain = ["max(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get)", "min(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get)"]
    domain.append(chosen_color)
    ##if we have a query block, add reference object dominant color as a possible option
    sampled_color_setter = random.choice(domain)
    if len(query_lines) > 0:
        domain.append("reference_obj.get_dominant_color()")
    change_color_params_line = """change_color_params = {{}}
{}
change_color_params['color'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'change_color', params = change_color_params)
""".format(query_lines, sampled_color_setter)
    return change_color_params_line

###get all possible color change snippets in a list
def get_color_change_domain(input_grid_objs, output_grid_objs, properties_set):
    possible_change_color_implementations = []
    colors_in_output = []
    for output_grid_obj in output_grid_objs:
        for output_obj in output_grid_obj.get_object_list():
            for color in output_obj.get_object_colors():
                if color not in colors_in_output:
                    colors_in_output.append(color)
    ####two cases. one, we are setting color equal to a color from the output object grid
    if len(colors_in_output) == 0:
        colors_in_output = [x for x in range(1, 10)]
    color_domain = colors_in_output + ["max(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get)", "min(input_grid_obj.get_color_occurrences(), key = input_grid_obj.get_color_occurrences().get)"]
    for color in color_domain:
        change_color_params_line = """change_color_params = {{}}
{}
change_color_params['color'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'change_color', params = change_color_params)
""".format("", color)
        possible_change_color_implementations.append(change_color_params_line)
    ##second case. change color based on other objects in the grid. 
    for query in get_total_query_domain(input_grid_objs, properties_set):
        query_lines = """query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]""".format(query)
        change_color_params_line = """change_color_params = {{}}
{}
change_color_params['color'] = reference_obj.get_dominant_color()
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'change_color', params = change_color_params)
""".format(query_lines)
        possible_change_color_implementations.append(change_color_params_line)
    return possible_change_color_implementations






##randomly samples a semantically meaningful set of cells to remove (if it's above or below something, remove it)
def sample_cell_list():
    sampled_index = random.randint(0, 1)
    sampled_comparator = random.choice([">", "<", "=="])
    if sampled_index == 0:
        domain = ["reference_obj.get_top_row()", "reference_obj.get_bottom_row()",
                  "reference_obj.get_object_center()[0]", 
                  "input_grid_obj.get_center()[0]"]
    if sampled_index == 1:
        domain = ["reference_obj.get_left_column()", "reference_obj.get_right_column()",
                  "reference_obj.get_object_center()[1]", 
                  "input_grid_obj.get_center()[1]"]
    sampled_value = random.choice(domain)
    sampled_bounds_str = "obj.find_cell_bounds(index = {}, comparator = '{}', value = {})".format(sampled_index, sampled_comparator, sampled_value)
    return sampled_bounds_str


###samples implementation of remove obj portion action
def sample_remove_obj_portion_code(input_grid_objs, properties_set, query = None):
    if query is None:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    sampled_cell_list = sample_cell_list()
    remove_obj_portion_code = """remove_obj_portion_params = {{}}
query = {}
reference_obj = input_grid_obj.find_obj(query, obj)[0]
remove_obj_portion_params['cell_list'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'remove_obj_portion', params = remove_obj_portion_params)
""".format(query, sampled_cell_list)
    return remove_obj_portion_code


def get_remove_obj_portion_domain(input_grid_objs, properties_set):
    possible_implementations = []
    for query in get_total_query_domain(input_grid_objs, properties_set):
        possible_cell_lists = []
        for cell in possible_cell_lists:
            continue
    return possible_implementations




##samples implementation of rotate in place
def sample_rotate_in_place_code(input_grid_objs, properties_set):
    sampled_angle = random.choice([90, 180, 270])
    rotate_in_place_code = """rotate_in_place_params = {{}}
rotate_in_place_params['angle'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'rotate_in_place', params = rotate_in_place_params)""".format(sampled_angle)
    return rotate_in_place_code



###returns all possible rotate implementations in a list
def get_rotate_in_place_domain(input_grid_objs, properties_set):
    possible_implementations = []
    angle_list = [90, 180, 270]
    for angle in angle_list:
        rotate_in_place_code = """rotate_in_place_params = {{}}
rotate_in_place_params['angle'] = {}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'rotate_in_place', params = rotate_in_place_params)
    """.format(angle)
        possible_implementations.append(rotate_in_place_code)
    return possible_implementations


###samples implementation of composite transformation (multiple transformations performed on the same object)
def sample_composite_trans_code(input_grid_objs, action_set, properties_set):
    composite_trans_implementation = ""
    ##first decision point is how many transformations occur. code chain limited at size 3 (even though an actual
    ##transformation could be longer) to prevent combinatorial explosion in the search process
    length_of_trans_code = random.randint(1,3)
    ##set domain to be the action set that makes sense to use based on this problem
    domain = deepcopy.copy(action_set)
    ##does not make sense to have delete or keep as is in domain as possible actions for composite action
    if "delete" in domain:
        domain.remove("delete")
    if "keep_as_is" in domain:
        domain.remove("keep_as_is")
    ##if domain is for some reason empty, use full function list
    if len(domain) == 0:
        domain = ["flip", "translate", "extend", "change_color", "remove_obj_portion"]
    #tracks how often an action occurs
    num_occurences_action = {}
    for actions in domain:
        num_occurences_action[actions] = 0
    ##in half of cases, randomly sample a query and use it for all of the transformations in the chain
    ##in the other half of cases, query is None and the query is set independently in individual cases
    choice = random.randint(1,2)
    if choice == 1:
        query = sample_simple_query(input_grid_objs = input_grid_objs, properties_set = properties_set)
    elif choice == 2:
        query = None
    for _ in range(length_of_trans_code):
        ##choosing an action randomly
        chosen_action = random.choice(domain)
        ##incrementing how often an action was performed
        num_occurences_action[chosen_action] += 1
        if chosen_action == "flip":
            sampled_action_implementation = sample_flip_code(input_grid_objs = input_grid_objs, properties_set = properties_set, query = query)
            ##it does not make sense to have more than 2 flips, as that is redundant
            if num_occurences_action["flip"] >= 2:
                domain.remove("flip")
        elif chosen_action == "translate":
            sampled_action_implementation = sample_translate_code(input_grid_objs = input_grid_objs, properties_set = properties_set, query = query)
            ##any thing accomplished by N number of translations can be done by a single translation
            domain.remove("translate")
        elif chosen_action == "extend":
            ##makes sense to have room for any number of extensions as possible
            sampled_action_implementation = sample_extend_code(input_grid_objs = input_grid_objs, properties_set = properties_set, query = query)
        elif chosen_action == "change_color":
            query = None
            sampled_action_implementation = sample_change_color_code(input_grid_objs = input_grid_objs, properties_set = properties_set, query = query)
            ##does not make sense to have more than one change color actions
            domain.remove("change_color")
        elif chosen_action == "remove_obj_portion":
            sampled_action_implementation = sample_remove_obj_portion_code(input_grid_objs = input_grid_objs, properties_set = properties_set, query = query)
            #does not usually make sense to have more than one remove object portion function
            domain.remove("remove_obj_portion")
        single_action_body = """{}
obj.set_object_grid(transformed_obj_grid)
""".format(sampled_action_implementation)
        composite_trans_implementation += single_action_body
    return composite_trans_implementation




##takes in a Program object, list of input_grid_objs and returns Program with one transformation function mutated
#randomly. Mutation involves changing parameters of input function. 
def mutate_trans_function(program, input_grid_objs, output_grid_objs, action_set, properties_set):
    new_params = copy.deepcopy(program.get_parameters())
    ###if there are no trans function snippets, there is nothing to mutate. return the old program.
    if len(list(program.get_transformation_implementation_snippets().keys())) == 0:
        return program
    random_key = random.choice(list(program.get_transformation_implementation_snippets().keys()))
    trans_func_of_interest = ""
    for x in range(len(program.get_trans_functions())):
        trans_functions = program.get_trans_functions()[x]
        if 'def transformation_{}'.format(random_key) in trans_functions:
            index_to_replace = x
            trans_func_of_interest = trans_functions
            break
    snippets_list = program.get_transformation_implementation_snippets()[random_key]
    if len(snippets_list) == 0:
        new_params["trans_functions"][index_to_replace] = sample_trans_function(number = random_key, input_grid_objs = input_grid_objs, output_grid_objs = output_grid_objs, action_set = action_set, properties_set = properties_set)
        new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
        return new_program
    code_snippet_to_mutate = random.choice(snippets_list)
    new_snippet = ""
    if code_snippet_to_mutate in trans_func_of_interest:
        ##figure out which transformation is in code snippet to determine which mutation to implement
        if 'flip' in code_snippet_to_mutate:
            new_snippet = indent(mutate_flip_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        elif 'translate' in code_snippet_to_mutate:
            new_snippet = indent(mutate_translate_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        elif 'extend' in code_snippet_to_mutate:
            new_snippet = indent(mutate_extend_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        elif 'change_color' in code_snippet_to_mutate:
            new_snippet = indent(mutate_change_color_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        elif 'remove_obj_portion' in code_snippet_to_mutate:
            new_snippet = indent(mutate_remove_obj_portion_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        elif 'rotate_in_place' in code_snippet_to_mutate:
            new_snippet = indent(mutate_rotate_in_place_code(code_snippet_to_mutate = code_snippet_to_mutate, input_grid_objs = input_grid_objs, properties_set = properties_set), 4)
        new_trans_function = trans_func_of_interest.replace(code_snippet_to_mutate, new_snippet)
        new_params["trans_functions"][index_to_replace] = new_trans_function
        new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
        return new_program
    else:
        return program

###takes in flip section and input grids for this problem and returns flip section with mutated parameters
def mutate_flip_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    new_code_snippet = ""
    attribute_to_mutate = random.choice(["axis", "direction"])
    if attribute_to_mutate == "direction":
        new_code_snippet = sample_flip_code(input_grid_objs = input_grid_objs, properties_set = properties_set)
        return new_code_snippet
    elif attribute_to_mutate == "axis":
        for line in code_snippet_to_mutate.splitlines():
            ##case 1. resample parameters.
            if 'axis' in line:
                cases = random.randint(1,2)
                if cases == 1:
                    direction_value = get_variable_value(line)
                    axis_section = sample_flip_axis(input_grid_objs = input_grid_objs, direction = direction_value, properties_set = properties_set)
                    new_code_snippet = """flip_params = {{}}
flip_params['direction'] = '{}'
{}
transformed_obj_grid = perform_transformation(grid = obj.get_object_grid(), transformation_category = 'flip', params = flip_params)
transformed_obj = GridObject(priority = 1, context_grid_obj = input_grid_obj, object_grid = transformed_obj_grid)
output_grid_obj.add_obj(transformed_obj)
    """.format(direction_value, axis_section)
                    return new_code_snippet
                ##case 2. increment or decrement.
                elif cases == 2:
                    new_code_snippet = ''
                    for line in code_snippet_to_mutate.splitlines():
                        ##count leading 
                        leading_spaces = count_leading_spaces(line)
                        if 'axis' in line:
                            choices = random.choice(["+ 1", "- 1"])
                            new_line = line + choices
                            new_line  = process_plus_string(new_line)
                            new_line = new_line.strip()
                            new_code_snippet += (leading_spaces * ' ') + new_line + '\n'
                        else:
                            new_code_snippet += line +'\n'
                    return new_code_snippet 


###resamples dx or dy values and returns OR increments/decrements dx, dy values and returns
def mutate_translate_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    cases = random.randint(1,2)
    ##resamples translate code
    if cases == 1:
        return sample_translate_code(input_grid_objs = input_grid_objs, properties_set = properties_set)
    ##change parameters by 1 in either direction
    elif cases == 2:
        axis_to_mutate = random.choice(['dx', 'dy'])
        new_code_snippet = ''
        for line in code_snippet_to_mutate.splitlines():
            leading_spaces = count_leading_spaces(line)
            if axis_to_mutate == 'dx':
                if 'dx' in line:
                    choices = random.choice(["+ 1", "- 1"])
                    new_line = line + choices
                    new_line  = process_plus_string(new_line)
                    new_line = new_line.strip()
                    new_code_snippet += (' ' * leading_spaces) + new_line + '\n'
                else:
                    new_code_snippet += line +'\n'
            elif axis_to_mutate == 'dy':
                if 'dy' in line:
                    choices = random.choice(["+ 1", "- 1"])
                    new_line = line + choices
                    new_line  = process_plus_string(new_line)
                    new_line = new_line.strip()
                    new_code_snippet += (' ' * leading_spaces) + new_line + '\n'
                else:
                    new_code_snippet += line +'\n' 
        return new_code_snippet                  
                

##takes in code snippet with extend code and returns mutated code
def mutate_extend_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    ##3 cases. first case is to resample the code that performs the extension. 
    cases = random.randint(1,3)
    new_code_snippet = ""
    ##resample extend code and return
    cases = 2
    if cases == 1:
        new_code_snippet = sample_extend_code(input_grid_objs = input_grid_objs, properties_set = properties_set)
        return new_code_snippet
    # Case 2: Change the object cell we are extending from
    if cases == 2:
        for lines in code_snippet_to_mutate.splitlines():
            # Counting leading spaces for lines
            leading_spaces = count_leading_spaces(lines)
            # Check for 'cell' in lines and perform mutation
            if 'cell' in lines:
                extend_params_name = extract_extend_parameter_name(code_snippet_to_mutate)[0]
                sampled_object_cell = sample_object_cell()
                new_line = "{}['cell'] = {}\n".format(extend_params_name, sampled_object_cell)
                new_code_snippet += (leading_spaces * ' ') + new_line
            else:
                # Add lines that aren’t mutated as they are
                new_line = lines.strip() + '\n'
                new_code_snippet += (leading_spaces * ' ') + new_line

    # Case 3: Adjust the length of the extension by incrementing or decrementing
    elif cases == 3:
        for lines in code_snippet_to_mutate.splitlines():
            # Counting leading spaces for lines
            leading_spaces = count_leading_spaces(lines)
            # Check for 'distance' in lines and perform mutation
            if 'distance' in lines:
                choices = random.choice(["+ 1", "- 1"])
                new_line = lines + choices
                new_line = process_plus_string(new_line)
                new_line = new_line.strip()
                new_code_snippet += (leading_spaces * ' ') + new_line
            else:
                # Add lines that aren’t mutated as they are
                new_line = lines.strip() + '\n'
                new_code_snippet += (leading_spaces * ' ') + new_line
    return new_code_snippet

##mutates change color code by resampling since there's only one parameter
def mutate_change_color_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    return sample_change_color_code(input_grid_objs = input_grid_objs, properties_set = properties_set)


##since there is only one parameter, resample the function to mutate it
def mutate_remove_obj_portion_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    return sample_remove_obj_portion_code(input_grid_objs = input_grid_objs, properties_set = properties_set)

##mutates rotating in place
def mutate_rotate_in_place_code(code_snippet_to_mutate, input_grid_objs, properties_set):
    return sample_rotate_in_place_code(input_grid_objs = input_grid_objs, properties_set = properties_set)

###removes checker trans pair based on whether a "useless" checker trans pair is found. I.E, a checker trans pair that is never executed
###If not, randomly removes one. If a useless 
def remove_checker_trans_pair(program):
    new_params = copy.deepcopy(program.get_parameters())
    if len(program.get_trans_functions()) == 0:
        return program
    random_index = random.randint(0, len(program.get_trans_functions()) - 1)
    if random_index < len(new_params["trans_functions"]):
        del new_params["trans_functions"][random_index]
    else:
        return program
    new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
    return new_program

##randomly generates a specific transform + add snippet from a transformation function
#returns a new Program object with the code snippets removed 
def remove_transformation_snippet(program):
    new_params = copy.deepcopy(program.get_parameters())
    if len(program.get_transformation_implementation_snippets()) == 0:
        return program
    random_key = random.choice(list(program.get_transformation_implementation_snippets().keys()))
    trans_func_of_interest = ""
    for x in range(len(program.get_trans_functions())):
        trans_functions = program.get_trans_functions()[x]
        if 'def transformation_{}'.format(random_key) in trans_functions:
            index_to_replace = x
            trans_func_of_interest = trans_functions
            break
    snippets_list = program.get_transformation_implementation_snippets()[random_key]
    if len(snippets_list) == 0:
        return program
    code_snippet_to_remove = random.choice(snippets_list)
    if code_snippet_to_remove in trans_func_of_interest:
        new_trans_function = trans_func_of_interest.replace(code_snippet_to_remove, 'transformed_obj_grid = obj.get_object_grid()')
        new_params["trans_functions"][index_to_replace] = new_trans_function
        new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
        return new_program
    else:
        return program

    

##takes in 2 programs and randomly mates them by swapping their obj checkers and/or transformation functions
##returns new program that is a "child" of two existing programs
def mate_programs(program_1, program_2):
    ##program 1 parameters
    program_1_params = program_1.get_parameters()
    ##program 2 parameters
    program_2_params = program_2.get_parameters()
    new_params = get_empty_program_params()
    ##randomly setting params with background color from 1 of 2 programs
    choice = random.randint(1, 2)
    if choice == 1:
        new_params["background_color"] = program_1_params["background_color"]
    elif choice == 2:
        new_params["background_color"] = program_2_params["background_color"]
    choice = random.randint(1, 2)
    if choice == 1:
        new_params["add_block"] = program_1_params["add_block"]
    elif choice == 2:
        new_params["add_block"] = program_2_params["add_block"]
    min_obj_checker_length = min(len(program_1.get_obj_checkers()), len(program_2.get_obj_checkers()))
    new_params["obj_checkers"] = [""] * min_obj_checker_length
    ##setting obj checkers in new params as one of the obj checkers from either program
    for x in range(min_obj_checker_length):
        choice = random.randint(1, 2)
        if choice == 1:
            new_params["obj_checkers"][x] = program_1_params["obj_checkers"][x]
        elif choice == 2:
            new_params["obj_checkers"][x] = program_2_params["obj_checkers"][x]
    min_trans_function_length = min(len(program_1.get_trans_functions()), len(program_2.get_trans_functions()))
    new_params["trans_functions"] = [""] * min_trans_function_length
    ##setting trans function in new params as one of the trans functions from either program
    for x in range(min_trans_function_length):
        choice = random.randint(1, 2)
        if choice == 1:
            new_params["trans_functions"][x] = program_1_params["trans_functions"][x]
        elif choice == 2:
            new_params["trans_functions"][x] = program_2_params["trans_functions"][x]
    new_program = SameSizeObjectCentricProgram(group = program_1.get_group(), problem_number = program_1.get_problem_number(), parameters = new_params)
    return new_program

###determines if a brand new object should be added to the 
##if so, samples code that adds object to grid and returns function text
def sample_add_new_object(input_grid_objs, output_grid_objs, action_set, properties_set):
    ###if we are not supposed to add a new object, as determined by the action set,
    ##no add_new_object text is created. return an empty string in its place. 
    if "add_new_object" not in action_set:
        return ""
    colors_in_input = []
    colors_in_output = []
    ##max number of additions from input to output
    max_num_additions = -1
    ##extract important information from input and output grids like what color the objects are
    ## and how many there are.
    ###looping through objects in output grid
    possible_offsets_dict = {}
    for input_grid_obj, output_grid_obj in zip(input_grid_objs, output_grid_objs):
        num_objects_input = len(input_grid_obj.get_object_list())
        num_objects_output = len(output_grid_obj.get_object_list())
        num_objects_added = num_objects_output - num_objects_input
        if num_objects_added > max_num_additions:
            max_num_additions = num_objects_added
        ##getting colors in input objects
        for input_obj in input_grid_obj.get_object_list():
            obj_color = input_obj.get_dominant_color()
            if obj_color not in colors_in_input:
                colors_in_input.append(obj_color)
        ##getting colors in output objects
        for output_obj in output_grid_obj.get_object_list():
            obj_color = output_obj.get_dominant_color()
            if obj_color not in colors_in_output:
                colors_in_output.append(obj_color)
        for output_obj in output_grid_obj.get_object_list():
            ###if the output object IS NOT in the input, this is a candidate for being a new object
            if not input_grid_obj.is_in_object_list(output_obj):
                ###looping through objects in input and getting the distance
                offset_dict = {}
                for input_obj in input_grid_obj.get_object_list():
                    offset = (int(output_obj.find_cell_pos(0)[0] - input_obj.find_cell_pos(0)[0]), int(output_obj.find_cell_pos(0)[1] - input_obj.find_cell_pos(0)[1]))
                    magnitude = math.sqrt(offset[0]**2 + offset[1]**2)
                    offset_dict[offset] = magnitude
                possible_offset = min(offset_dict, key=offset_dict.get)
                if possible_offset in possible_offsets_dict:
                    possible_offsets_dict[possible_offset] += 1
                else:
                    possible_offsets_dict[possible_offset] = 1
    ###track the offsets that are consistent across output grids
    possible_offsets = []
    for possible_offset in possible_offsets_dict:
        if possible_offsets_dict[possible_offset] == len(output_grid_objs):
            possible_offsets.append(possible_offset)

    ##isolate which colors occur in OUTPUT but not input. this is the estimated color of the new object.
    unique_colors = list(set(colors_in_output) - set(colors_in_input))
    if len(unique_colors) == 0:
        sampled_color = int(random.choice(colors_in_input))
    else:
        sampled_color = int(random.choice(unique_colors))
    function_str  = sample_complex_add_new_function(input_grid_objs = input_grid_objs, properties_set = properties_set, color_addition = sampled_color, possible_offsets = possible_offsets)
    return function_str

###samples body of a possible add function for a program based on the specific inputs for that problem
def sample_complex_add_new_function(input_grid_objs, properties_set, color_addition, possible_offsets):
    sampled_query = "None"
    chosen_reference_obj_list = random.choice(["input_grid_obj.find_obj(query)", "input_grid_obj.get_object_list()"])
    if chosen_reference_obj_list == "input_grid_obj.find_obj(query)":
        sampled_query = sample_simple_query(input_grid_objs, properties_set)
    sampled_bounds = ""
    possible_landmarks = ["(reference_obj.get_top_row() - 1, reference_obj.get_object_center()[1])",
                       "(reference_obj.get_bottom_row() + 1, reference_obj.get_object_center()[1])",
                       "(reference_obj.get_object_center()[0], reference_obj.get_left_column() - 1)",
                       "(reference_obj.get_object_center()[0], reference_obj.get_right_column() + 1)",
                       "(reference_obj.get_object_gaps()[0][0])"]
    if len(possible_offsets) > 0:
        offset = random.choice(possible_offsets)
        possible_landmarks.append("(reference_obj.find_cell_pos(0)[0] + {}, reference_obj.find_cell_pos(0)[1] + {})".format(offset[0], offset[1]))
    sampled_landmark = random.choice(possible_landmarks)
    landmark_setter = "landmark = {}".format(sampled_landmark)
    landmark_setter = indent(landmark_setter, 8)
    ##sample bounds
    function_str = """def add_new_objects(input_grid_obj, output_grid_obj):
    query = {}
    reference_obj_list = {}
    output_background_color = output_grid_obj.get_background_color()
    for reference_obj in reference_obj_list:
{}
        bounds = (landmark[0], landmark[0], landmark[1], landmark[1])
        new_obj_grid = create_new_object(grid_shape = output_grid_obj.get_shape(), background_color = output_background_color, obj_color = {}, obj_bounds = bounds)
        new_obj = GridObject(priority = 1, context_grid_obj = output_grid_obj, object_grid = new_obj_grid)
        output_grid_obj.add_obj(new_obj)
    return output_grid_obj
""".format(sampled_query, chosen_reference_obj_list, landmark_setter, color_addition)
    return function_str

###changes add function of program
def mutate_add_new_function(program, input_grid_objs, output_grid_objs, action_set, properties_set):
    old_params = program.get_parameters()
    new_params = copy.deepcopy(old_params)
    new_params["add_block"] = sample_add_new_object(input_grid_objs = input_grid_objs, output_grid_objs = output_grid_objs, action_set = action_set, properties_set = properties_set)
    new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
    return new_program



