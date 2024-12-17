from utils import indent, unindent_code, extract_code_llm
import anthropic

def get_empty_program_params():
    add_original_domain = ["output_grid_obj.add_obj(obj)", ""]
    ##randomly sample statement adding original object or not
    add_original = random.choice(add_original_domain)
    add_original = indent(add_original, 4)
    obj_checker = "obj_checker_1 = True"
    trans_function = """def transformation_1(input_grid_obj, output_grid_obj, obj):
{}
    return output_grid_obj
""".format(add_original)
    params = {"background_color": 0, "obj_checkers": [obj_checker], "trans_functions":[trans_function], "add_block":""}
    return params


##handles case where the entire function and all helpers are in a single string
def process_full_code_response(response_text):
    trans_functions = extract_transformations(response_text)
    trans_functions = reindent_transformation_code(trans_functions)
    obj_checkers = extract_obj_checkers_conditional(response_text)
    params = {"background_color": 0,  "obj_checkers": [], "trans_functions":[], "add_block":""}
    params["obj_checkers"] = obj_checkers
    params["trans_functions"] = trans_functions
    params["add_block"] = extract_add_function(response_text)
    program = SameSizeObjectCentricProgram(group = "eval", problem_number = 89, parameters = params, should_process = False)
    return program



##takes in example file and returns text/string
def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        code_string = file.read()
    return code_string

##string of necessary imports to add to top of program
def get_import_string():
    import_string = ""
    import_string += "import numpy as np\n"
    import_string += "import math\n"
    import_string += "from actions import *\n"
    import_string += "from vision import determine_direction_landmark\n"
    import_string += "from ARC_objects import Grid, GridObject, update_object_numbers\n"
    return import_string



##takes in string representation of function
def extract_transformation_number(function_string):
    match = re.search(r'transformation_(\d+)', function_string)
    if match:
        return match.group(1)  # This will return the number as a string
    return None


##determines if multi-line if, while, and for statements are in the code-block string
##if they are, we don't reindent
def should_reindent(code_block):
    # Regular expression to match if, while, and for statements that end with a colon and are multiline
    pattern = r'^\s*(if|while|for)\b.*:\s*$'
    
    # Search for matches line by line
    for line in code_block.splitlines():
        if re.search(pattern, line):
            return False
    return True

def remove_blank_lines(text):
    return "\n".join(line for line in text.splitlines() if line.strip() != "")



##uses regular expressions to extract obj checkers from text
def extract_obj_checkers(response_text):
    # Regular expression to capture `obj_checker_n = <python_boolean_expression>`
    pattern = r"d\.\s*(`?obj_checker_\d+\s*=\s*.+?`?)\s*(?=[a-f]\.|$)"
    # Find all matches in the response text
    matches = re.findall(pattern, response_text, re.DOTALL)
    # Clean up results to remove any surrounding backticks and whitespace
    obj_checkers = [match.strip(" `") for match in matches]
    return obj_checkers

import re

##extracts object checkers that may not be of format obj_checker_n from response text
def extract_obj_checkers_conditional(response_text):
    # Regular expression to capture obj_checker-style assignments
    obj_checker_pattern = r"d\.\s*(`?obj_checker_\d+\s*=\s*.+?`?)\s*(?=[a-f]\.|$)"
    # Regular expression to capture conditional statements (if, elif)
    conditional_pattern = r"(if|elif)\s+(.+?):"
    
    # Find all obj_checker-style assignments
    obj_checker_matches = re.findall(obj_checker_pattern, response_text, re.DOTALL)
    obj_checkers = [match.strip(" `") for match in obj_checker_matches]
    
    # Find all conditional statements
    conditional_matches = re.findall(conditional_pattern, response_text, re.DOTALL)
    conditional_expressions = [match[1].strip() for match in conditional_matches]
    
    # Combine obj_checker matches and conditional expressions
    obj_checkers.extend(conditional_expressions)
    return obj_checkers


##uses regular expressions to extract the transformation functions from the response text
def extract_transformations(response_text):
    # Regular expression to capture `def transformation_n(...)` function definitions with code
    pattern = r"`?def (transformation_\d+)\(input_grid_obj, output_grid_obj, obj\):(.+?return output_grid_obj)`?"
    # Find all matches in the response text
    matches = re.findall(pattern, response_text, re.DOTALL)
    # Construct function code blocks with proper formatting
    transformations = [
        f"def {name}(input_grid_obj, output_grid_obj, obj):\n{code.strip()}"
        for name, code in matches
    ]
    return transformations


##uses regular expressions to extract the transformation functions from the response text
def extract_add_function(response_text):
    # Regular expression to capture `def transformation_n(...)` function definitions with code
    pattern = r"`?def (add_new_objects)\(input_grid_obj, output_grid_obj, obj\):(.+?return output_grid_obj)`?"
    # Find all matches in the response text
    matches = re.findall(pattern, response_text, re.DOTALL)
    # Construct function code blocks and clean up any unnecessary whitespace or backticks
    transformations = [f"def {name}(input_grid_obj, output_grid_obj, obj):{code.strip()}" for name, code in matches]
    if len(transformations) > 0:
        return transformations[0]
    else:
        return ""


###takes in code snippets extracted from LLM and uses it to create the transformation block in the object centric template
def create_transformation_block(obj_checkers, trans_functions):
    block = ""
    for code in obj_checkers:
        if 'def transformation' not in code:
            if code not in block:
                code = unindent_code(code)
                block += code + '\n'
    for code in trans_functions:
        if 'def transformation' in code:
            trans_number = extract_transformation_number(code)
            if 'obj_checker_{}'.format(trans_number) not in block:
                block += 'obj_checker_{} = True\n'.format(trans_number)
    for code in trans_functions:
        if 'def transformation' in code:
            trans_number = extract_transformation_number(code)
            if 'if obj_checker' not in block:
                trans_block = """
if obj_checker_{}:
    output_grid_obj = transformation_{}(input_grid_obj = input_grid_obj, output_grid_obj = output_grid_obj, obj = obj)""".format(trans_number, trans_number)
            else:
                trans_block = """
elif obj_checker_{}:
    output_grid_obj = transformation_{}(input_grid_obj = input_grid_obj, output_grid_obj = output_grid_obj, obj = obj)""".format(trans_number, trans_number)
            block += trans_block
    block = block.rstrip()
    return block


##returns parameter dictionary based on LLM response. takes in filename as parameter. fills in values in 
##dictionary correspondingly.
def create_params_from_llm_text_sample(text):
    params = {"background_color": 0,  "obj_checkers": [], "trans_functions":[], "add_block":""}
    initial_code_snippets = extract_code_llm(text)
    for code in initial_code_snippets:
        if 'def' in code and '(' in code and ")" in code and ":" in code and "transformation" in code:
            params["trans_functions"].append(code)
        elif 'obj_checker' in code:
            params["obj_checkers"].append(code)
        elif 'def' in code and '(' in code and ")" in code and ":" in code and "add_new_objects" in code:
            params["add_block"] = code
    obj_checkers_regex = extract_obj_checkers(text)
    params["obj_checkers"] = params["obj_checkers"] + obj_checkers_regex
    ##extracts all necessary values from LLM return string
    for line in text.splitlines():
        if len(line) > 0:
            ##account for any unexpected leading or trailing whitespaces
            line_start = line[0]                
            ##extract background color from proper section
            if line_start == '0':
                for chars in line[1]:
                    if chars.isnumeric():
                        params["background_color"] = chars
    return params


##create parameters from the response the LLM gives to the mutate prompt
def create_params_from_llm_text_mutate(text):
    params = {"background_color": 0,  "obj_checkers": [], "trans_functions":[], "add_block":""}
    params["obj_checkers"] = extract_obj_checkers(text)
    params["trans_functions"] = extract_transformations(text)
    params["add_block"] = extract_add_function(text)
    return params




##extract individual function definitions from code snippets list. returns long string of all definitions appended together.
def extract_sub_functions(trans_functions):
    sub_functions_block = ""
    for code in trans_functions:
        if 'def ' in code:
            code = unindent_code(code)
            sub_functions_block += code + '\n'
    return sub_functions_block

###takes in parameters dictionary specifying background color, boolean indicating whether to add new values, and code 
###snippets, fills in template and creates entire object centric template. returns whole program.
def fill_in_same_size_object_centric_template(params):
    sampled_background_color = params["background_color"]
    extracted_sub_functions = extract_sub_functions(params["trans_functions"])
    sampled_trans_block = indent(create_transformation_block(params["obj_checkers"], params["trans_functions"]), 8)
    if len(params["add_block"]) > 1 and ("def add_new_objects" in params["add_block"]):
        run_add_block = "output_grid_obj = add_new_objects(input_grid_obj, output_grid_obj)"
        run_add_block = indent(run_add_block, 4)
        add_block = params["add_block"]
    else:
        run_add_block = ""
        add_block = ""
    program_str = """
def object_centric_rule_transformation(grid):
    ###DO NOT TOUCH THE BELOW BLOCK OF CODE
    input_grid_obj = Grid(grid)
    update_object_numbers(input_grid_obj)
    output_background_color = get_background_color(grid)
    output_grid_initial = np.ones(grid.shape, dtype=int) * output_background_color
    output_grid_obj = Grid(output_grid_initial)
    ##sample add new object block if rule involves adding new object
    for obj in input_grid_obj.get_object_list():
        obj_grid = obj.get_object_grid()
{}
{}   
    output_grid = output_grid_obj.get_grid()
    return output_grid
""".format(sampled_trans_block, run_add_block)
    program_str = add_block + program_str
    import_str = get_import_string()
    program_str = import_str + extracted_sub_functions + program_str
    program_str = program_str.replace("```python","")
    program_str = program_str.replace("```","")
    program_str = program_str.replace("`","")
    return remove_blank_lines(program_str)

###gets the number of occurrences of obj_checker_n in program_str
def get_number_object_checkers(program_str):
    pattern = r'obj_checker_\d+'  # Regex pattern to match "obj_checker" followed by digits
    return len(re.findall(pattern, program_str))

##takes in list of code blocks for transformation functions and processes the information in it
##returns dictionary with transformation information (number of adds, which transformations occur)
def process_transformations(trans_block_list):
    trans_info_dict = {"number_object_additions": 0, "transformation_instances":[], "transformation_code_snippets": {}}
    for x in range(len(trans_block_list)):
        code_block = trans_block_list[x]
        transformation_number = extract_transformation_number(code_block)
        in_trans_block = False
        trans_code_block = ""
        trans_snippets_list = []
        for line in code_block.splitlines():
            ##extracting only the specific sections where transformation parameters are defined and transformations are implemented
            ##each individual transformation block in a transformation extracted as its own section and added to a list
            if 'params' in line and in_trans_block is False:
                in_trans_block = True
                trans_code_block += line + '\n'
            ##ending transformation snippet when perform_transformation function is called
            elif (in_trans_block) and ('perform_transformation' in line):
                trans_code_block += line + '\n'
                in_trans_block = False
                trans_snippets_list.append(trans_code_block)
                trans_code_block = ""
            ##if in middle of trans block, add it to current trans block
            elif in_trans_block:
                trans_code_block += line + '\n'
            ##incrementing values in trans info dict based on what values are found
            if 'output_grid_obj.add_obj' in line:
                trans_info_dict["number_object_additions"] += 1
            if 'flip' in line:
                trans_info_dict["transformation_instances"].append('flip')
            elif 'translate' in line:
                trans_info_dict["transformation_instances"].append('translate')
            elif 'extend' in line:
                trans_info_dict["transformation_instances"].append('extend')
            elif 'change_color' in line:
                trans_info_dict["transformation_instances"].append('change_color')
            elif 'remove_obj_portion' in line:
                trans_info_dict["transformation_instances"].append('remove_obj_portion')
            elif 'rotate_in_place' in line:
                trans_info_dict["transformation_instances"].append("rotate_in_place")
        ##set transformation code snippets section with dictionary mapping function number with its list of transformation snippets
        trans_info_dict["transformation_code_snippets"][transformation_number] = trans_snippets_list
    return trans_info_dict


##takes in trans_block_list, and if the transformation function has no complex logic or structures
##and is just a series of statements, reindent all of them to make sure the indentation is correct
def reindent_transformation_code(trans_block_list):
    for x in range(len(trans_block_list)):
        new_code_block = ""
        code_block = trans_block_list[x]
        reindent = should_reindent(code_block)
        for line in code_block.splitlines():
            if reindent:
                ##strip everything so all leading spaces are 0
                line = line.strip()
                ##if it's not the first line, re-indent to level 4 and add to new line
                pattern = r'^\s*def\s+transformation(?:_\d+)?\s*\(.*\):\s*$'
                if not bool(re.search(pattern, line)):
                    new_code_block += indent(line, 4) + '\n'
                ##if it is the first line
                else:
                    new_code_block += line + '\n'
            else:
                new_code_block += line + '\n'
        trans_block_list[x] = new_code_block
        ##otherwise, add the old code block without indentation level changed back to trans list
    return trans_block_list




def process_add_block(add_block):
    add_block_info_dict = {"number_object_additions": 0}
    for line in add_block.splitlines():
        if 'x' in line and "==" in line and ":" in line:
            add_block_info_dict["number_object_additions"] += 1
    return add_block_info_dict


class SameSizeObjectCentricProgram:
    def __init__(self, group, problem_number, parameters, llm_generated = False, should_process = True):
        ##train or eval, which category
        if should_process:
            self._group = group
            ##dictionary with key parameters that create program string
            self._parameters = parameters
            ##make the first thing we do to make sure trans functions are correctly indented
            self._parameters["trans_functions"] = reindent_transformation_code(self._parameters["trans_functions"])
            ##obj checkers from parameters
            self._object_checkers = parameters["obj_checkers"]
            ##trans functions from parameters
            self._trans_functions = parameters["trans_functions"]
            ##add block from parameters
            self._add_block = parameters["add_block"]
            ##which problem number is this program for
            self._problem_number = problem_number
            ##whether program was generated by a language model or not
            self._llm_generated = llm_generated
            ##number of times an object is added to the output grid as a part of this program
            transformation_info_dict = process_transformations(self._trans_functions)
            brand_new_addition_info_dict = process_add_block(self._add_block)
            self._number_additions = transformation_info_dict["number_object_additions"] + brand_new_addition_info_dict["number_object_additions"]
            self._transformation_list = transformation_info_dict["transformation_instances"]
            self._transformation_snippets = transformation_info_dict["transformation_code_snippets"]
            ##fill in template with parameters to create text representation of program to run
            self._text = fill_in_same_size_object_centric_template(parameters)
        else:
            self._group = group
            ##dictionary with key parameters that create program string
            self._parameters = parameters
            ##obj checkers from parameters
            self._object_checkers = parameters["obj_checkers"]
            ##trans functions from parameters
            self._trans_functions = parameters["trans_functions"]
            ##add block from parameters
            self._add_block = parameters["add_block"]
            ##which problem number is this program for
            self._problem_number = problem_number
            ##whether program was generated by a language model or not
            self._llm_generated = llm_generated
            ##number of times an object is added to the output grid as a part of this program
            transformation_info_dict = process_transformations(self._trans_functions)
            brand_new_addition_info_dict = process_add_block(self._add_block)
            self._number_additions = transformation_info_dict["number_object_additions"] + brand_new_addition_info_dict["number_object_additions"]
            self._transformation_list = transformation_info_dict["transformation_instances"]
            self._transformation_snippets = transformation_info_dict["transformation_code_snippets"]
            ##fill in template with parameters to create text representation of program to run
            self._text = fill_in_same_size_object_centric_template(parameters)         
    
    def get_group(self):
        return self._group

    def get_text(self):
        return self._text


    def set_text(self, text):
        if 'import' not in text:
            text = get_import_string() + text
        self._text = text
    

    def get_display_text(self):
        return remove_blank_lines(self._text)
    
    def get_problem_number(self):
        return self._problem_number
    
    def get_parameters(self):
        return self._parameters
    
    def get_obj_checkers(self):
        return self._object_checkers
    
    def get_trans_functions(self):
        return self._trans_functions
    
    def get_number_additions(self):
        return self._number_additions
    
    def get_transformation_list(self):
        return self._transformation_list
    
    def get_transformation_implementation_snippets(self):
        return self._transformation_snippets

    #takes in an input grid and runs the program on it. returns output grid (numpy form).
    def run_string_program(self, input_grid):
        local_namespace = {}
        # Execute the code with the specified local namespace
        exec(self._text, globals(), local_namespace)
        # Bind the local functions to their correct namespace
        for name, obj in local_namespace.items():
            if callable(obj) and hasattr(obj, '__globals__'):
                # Update the globals of the function with the local namespace
                obj.__globals__.update(local_namespace)
        # Now you can call the function from the local namespace
        output_grid = local_namespace["object_centric_rule_transformation"](input_grid)
        return output_grid

    


