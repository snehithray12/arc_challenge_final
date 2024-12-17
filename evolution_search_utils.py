import random
import ast
import re


##takes in a string representing a line of code and returns the number of leading spaces before it
##to get the indentation level. 
##takes in string, returns integer
def count_leading_spaces(line):
    leading_spaces = 0
    in_leading = True
    for x in range(len(line)):
        if line[x] == ' ' and in_leading:
            leading_spaces += 1
        elif line[x] != ' ':
            in_leading = False
    return leading_spaces


##takes in a string of an integer parameter and adds up all of the attached + ints 
##ex. x + 1 + 1 + 1 + 1 is turned into x + 4. returns the processed string.
def process_plus_string(input_str):
    split_string = input_str.split("+")
    total = 0
    new_str = ""
    for items in split_string:
        items = items.strip(' ')
        if (items.lstrip('-').isdigit()):
            total += int(items)
        else:
            new_str += items + " + "
    new_str += str(total)
    return new_str

def extract_extend_parameter_name(text):
    # Regex to match either "extend_params" or "extend_params_n" (where n is a number)
    pattern = r'extend_params(?:_\d+)?'
    # Find all matches in the input text
    matches = re.findall(pattern, text)
    return matches
        
    



#gets a string of an assignment and extracts the string of the value assigned to the variable
#extracts everything after the equal 
def get_variable_value(text):
    return text.split('=', 1)[1].strip()





