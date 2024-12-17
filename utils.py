import numpy as np
from numpy.linalg import norm
import json
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import re
import ast
import asyncio
from mistralai import Mistral
import requests
import random
import google.generativeai as genai
import asyncio
from anthropic import AsyncAnthropic, Anthropic
import time


##initialize gen-ai models
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
GEMINI_MODEL_FLASH = genai.GenerativeModel('gemini-1.5-flash')
GEMINI_MODEL_PRO = genai.GenerativeModel('gemini-1.5-pro')


mistral_api_key = os.environ["MISTRAL_API_KEY"]
MISTRAL_CLIENT = Mistral(api_key=mistral_api_key)

CLAUDE_MODEL_SYNC = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def length_without_spaces(text):
    return len(text.replace(" ", "").replace("\n", ""))


##loads json file and returns data
def read_json_file(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_name} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

##gets all json files in project directory
def get_json_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filenames.append(directory + filename)
    return filenames


#gets training pairs from task
def get_train_pairs(file):
    train_list = []
    data = read_json_file(file)
    for dicts in data['train']:
        train_input = np.asarray(dicts['input'])
        train_output = np.asarray(dicts['output'])
        train_list.append((train_input, train_output))
    return train_list

##gets testing pairs from task
def get_test_pairs(file):
    test_list = []
    data = read_json_file(file)
    for dicts in data['test']:
        test_input = np.asarray(dicts['input'])
        test_output = np.asarray(dicts['output'])
        test_list.append((test_input, test_output))
    return test_list



##checks if the answer is correct
def verify(source, target):
    return np.array_equal(source, target)

##how similar is a grid with another grid (task grids)
def element_wise_similarity(source, target):
    squared_error = np.square(source - target)
    average_squared_error = np.mean(squared_error)
    return average_squared_error

def visualize_single_object(obj, title="Visualization of Single Object"):
    # Define a colormap (you can modify this as per your requirements)
    cmap = ListedColormap(['black', 'yellow', 'red', 'blue', 'cyan', 'gray', 'magenta', 'pink', 'maroon', 'white', 'orange'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(obj, cmap=cmap, vmin=0, vmax=10)
    ax.set_title(title)
    
    plt.show()


"""
Single input and output pair visualization.
"""
def visualize_single_pair(input, output, title="Visualization of Input and Output"):
    cmap = ListedColormap(['black', 'yellow', 'red', 'blue', 'cyan', 'gray', 'magenta', 'pink', 'maroon', 'white', 'orange'])
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(input, cmap=cmap, vmin=0, vmax=10)
    axs[0].set_title('Input')
    axs[1].imshow(output, cmap=cmap, vmin=0, vmax=10)
    axs[1].set_title('Output')
    fig.suptitle(title, fontsize=20)
    plt.show()
    return

"""
Visualize feature list
"""
def visualize_feature_list(input_array, feature_list):
    cmap = ListedColormap(['black', 'yellow', 'red', 'blue', 'cyan', 'gray', 'magenta', 'pink', 'maroon', 'white', 'orange'])
    num_outputs = len(feature_list)
    if num_outputs > 0:
        fig, axs = plt.subplots(1, num_outputs + 1, figsize=(30, 30))
        
        axs[0].imshow(input_array, cmap=cmap, vmin=0, vmax=10)
        axs[0].set_title('Input')
        for i, output in enumerate(feature_list):
            axs[i + 1].imshow(output, cmap=cmap, vmin=0, vmax=10)
            axs[i + 1].set_title(f'Matched feature {i + 1}')
    else:
        plt.imshow(input_array, cmap=cmap, vmin=0, vmax=10)
    plt.show()
    return




"""
displays the input/output grids for a given arc problem or for all of them. 
"""
def visualize_problem(index=None):
    filenamelist = get_json_filenames("ARC-AGI/data/training/")
    if index != None:
        train_file = filenamelist[index]
        train_pairs = get_train_pairs(train_file)
        for train_pair in train_pairs:
            train_input = train_pair[0]
            train_output = train_pair[1]
            fig, axs = plt.subplots(1, 2, figsize=(30, 30))
            visualize_single_pair(input=train_input, output=train_output)
            plt.show()
        return
    else:
        for train_files in filenamelist:
            train_pairs = get_train_pairs(train_files)
            for train_pair in train_pairs:
                train_input = train_pair[0]
                train_output = train_pair[1]
                fig, axs = plt.subplots(1, 2, figsize=(30, 30))
                visualize_single_pair(input=train_input, output=train_output)
                return

"""
displays the input/output grids for a given arc problem or for all of them. 
"""
def visualize_solution(index):
    filenamelist = get_json_filenames("ARC-AGI/data/evaluation/")
    if index != None:
        test_file = filenamelist[index]
        test_pairs = get_test_pairs(test_file)
        for test_pair in test_pairs:
            test_input = test_pair[0]
            test_output = test_pair[1]
            fig, axs = plt.subplots(1, 2, figsize=(30, 30))
            visualize_single_pair(input=test_input, output=test_output)
            plt.show()
        return



def save_string_to_file(content, filename):
    with open(filename, 'w') as file:
        file.write(content)
    file.close()



def prompt_anthropic_regular(client, prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        system=[
        {
        "type": "text",
        "text": "You are a genius puzzle solver and code writer. Your job is to describe and implement solutions to the puzzles I give you."
        }
        ],
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        ]
    )
    output_txt = message.content[0].text
    return output_txt


def prompt_anthropic_large_sync(client, context, prompt):
    message = client.beta.prompt_caching.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    temperature=1,
    system=[
    {
        "type": "text",
        "text": "You are a programmer. Your task is to write simple and correct functions that fulfill the specification. Specifically, you will write code that turns the provided inputs to the outputs."
    },
    {
        "type": "text",
        "text": context,
        "cache_control": {"type": "ephemeral"}
    }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ])
    ##print statement to track cache usage 
    output_txt = message.content[0].text
    return output_txt


async def prompt_anthropic_large_async(client, context, prompt):
    message = await client.beta.prompt_caching.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    temperature=1,
    system=[
    {
        "type": "text",
        "text": "You are a programmer. Your task is to write simple and correct functions that fulfill the specification. Specifically, you will write code that turns the provided inputs to the outputs."
    },
    {
        "type": "text",
        "text": context,
        "cache_control": {"type": "ephemeral"}
    }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ])
    ##print statement to track cache usage 
    output_txt = message.content[0].text
    return output_txt



def prompt_anthropic_small(client, prompt):
    message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=4096,
    temperature=0.7,
    system="You are a programmer. Your task is to write simple and correct functions that fulfill the specification. Specifically, you will write code that turns the provided inputs to the outputs. Some rules for you. It is very important that you do not deviate from the template given. You should not delete any lines of code whatsoever. Simply fill in the parts you are told to.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ])
    output_txt = message.content[0].text
    return output_txt

#takes in model object and string prompt and returns the text of output
async def prompt_gemini(model, prompt):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)  # Runs in a separate thread
    return response.text

def prompt_gemini_normal(model, prompt):
    response = model.generate_content(prompt)
    return response.text




def prompt_mistral_normal(client, prompt):
    model_type = "mistral-large-latest"
    chat_response = client.chat.complete(
        model= model_type,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content


def prompt_huggingface_llama(prompt):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
    headers = {"Authorization": "ADD API KEY HERE"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']


###prompt various models based on availability and free tier usage
def prompt_llm_master_sync(prompt, problem_number, save_to_file = False, directory = "", filename = None, sample_number = None):
    choice = random.randint(1,2)
    ###order of models in list, claude and gemini can change order but mistral and llama will always be backups
    models = ["claude","gemini", "mistral", "llama"]
    # if choice == 1:
    #     models = ["gemini", "claude", "mistral", "llama"]
    ##loop through models. if a model throws an error when prompted, loop through and prompt another
    ##otherwise, return response from the model
    sampled = False 
    for model in models:
        if model == "claude":
            try:
                response = prompt_anthropic_regular(CLAUDE_MODEL_SYNC, prompt)
                if save_to_file:
                    if directory == "":
                        directory = "llm_response_dump"
                    if sample_number is None:
                        sample_number = int(time.time())
                    if filename is None:
                        filename = "claude_{}_{}.txt".format(problem_number, sample_number)
                    filename  = directory + '/' + filename
                    save_string_to_file(response, filename)
                print("got response from claude model ")
                sampled = True
                return response
            except Exception as e:
                print("Errored with model {} on problem {} at time {} .\n {}".format(model, problem_number, time.time(), str(e)))
                continue
        elif model == "gemini":
            try:
                response = prompt_gemini_normal(GEMINI_MODEL_PRO, prompt)
                if save_to_file:
                    if directory == "":
                        directory = "llm_response_dump"
                    if sample_number is None:
                        sample_number = int(time.time())
                    if filename is None:
                        filename = "gemini_{}_{}.txt".format(problem_number, sample_number)
                    filename  = directory + '/' + filename
                    save_string_to_file(response, filename)
                print("got response from gemini model ")
                sampled = True
                return response
            except Exception as e:
                print("Errored with model {} on problem {} at time {} .\n {}".format(model, problem_number, time.time(), str(e)))
                continue
        elif model == "mistral":
            try:
                response = prompt_mistral_normal(MISTRAL_CLIENT, prompt)
                if save_to_file:
                    if directory == "":
                        directory = "llm_response_dump"
                    if sample_number is None:
                        sample_number = int(time.time())
                    if filename is None:
                        filename = "mistral_{}_{}.txt".format(problem_number, sample_number)
                    filename  = directory + '/' + filename
                    save_string_to_file(response, filename)
                sampled = True
                print("got response from mistral model ")
                return response
            except Exception as e:
                print("Errored with model {} on problem {} at time {} .\n {}".format(model, problem_number, time.time(), str(e)))
                continue
        elif model == "llama":
            try:
                response = prompt_huggingface_llama(prompt)
                if save_to_file:
                    if directory == "":
                        directory = "llm_response_dump"
                    if sample_number is None:
                        sample_number = int(time.time())
                    if filename is None:
                        filename = "llama_{}_{}.txt".format(problem_number, sample_number)
                    filename  = directory + '/' + filename
                    save_string_to_file(response, filename)
                sampled = True
                print("got response from llama model in huggingface ")
                return response
            except Exception as e:
                print("Errored with model {} on problem {} at time {} .\n {}".format(model, problem_number, time.time(), str(e)))
                continue
    return None







##takes string output from LLM and extracts just the code portions to run
def extract_code_llm(text):
    lines = text.splitlines()
    code_list = []
    code = ""
    start_line = '```python'
    end_line = '```'
    in_middle = False
    for line in lines:
        if start_line in line:
            in_middle = True
        elif in_middle and end_line in line:
            in_middle = False
                ##getting rid of any extraneous characters that shouldn't be in the string
            code = code.replace("```python","")
            code = code.replace("```","")
            code = code.replace("`","")
            code_list.append(code)
            code = ""
        elif in_middle:
            code += line + '\n'
    return code_list




##takes in a string representing a text, potentially multi line AND
##integer representing number of leading spaces to indent by
##returns a tabbed version of the string
def indent(text, leading_spaces):
    if text == '' or text is None:
        return '\n'
    return '\n'.join((' ' * leading_spaces) + line for line in text.split('\n'))

##takes in text and returns a version of the function with all leading spaces removed but still maintaining
##relative indentation across lines
def unindent_code(code_str):
    # Split the input string into individual lines
    lines = code_str.split('\n')
    # Find the smallest number of leading spaces in front of non-empty lines
    min_leading_spaces = float('inf')
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:  # Ignore empty lines
            leading_spaces = len(line) - len(stripped_line)
            min_leading_spaces = min(min_leading_spaces, leading_spaces)
    # If there's no indentation, just return the original code
    if min_leading_spaces == float('inf'):
        return code_str
    # Remove the smallest number of leading spaces from all lines
    unindented_lines = [line[min_leading_spaces:] for line in lines]
    # Join the unindented lines back into a single string
    return '\n'.join(unindented_lines)


##determines if two strings are equal, ignoring spaces and newlines
def compare_code_strings(str1, str2):
    # Remove extra spaces and blank lines
    def normalize_string(s):
        # Remove all leading and trailing spaces from each line
        lines = [line.strip() for line in s.splitlines() if line.strip()]
        # Join lines into a single string without spaces
        return "".join(lines)

    # Normalize both strings
    normalized_str1 = normalize_string(str1)
    normalized_str2 = normalize_string(str2)

    # Compare normalized strings
    return normalized_str1 == normalized_str2

###break up string code block into a list of functions
def split_functions(code_string):
    # Parse the code into an AST
    tree = ast.parse(code_string)
    functions = []
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):  # Check if the node is a function definition
            # Use the AST to retrieve the source code of the function
            start_line = node.lineno - 1  # line numbers in AST are 1-indexed
            end_line = node.end_lineno     # end_lineno gives the last line of the function in Python 3.8+
            function_code = "\n".join(code_string.splitlines()[start_line:end_line])
            functions.append(function_code)
    
    return functions

##parses through hypothesis search text and extracts the final rule from it
def extract_final_rule_section(text):
    lines = text.splitlines()  # Split text into individual lines
    for i, line in enumerate(lines):
        if "final rule" in line.lower():  # Look for "final rule" (case-insensitive)
            return "\n".join(lines[i:])   # Join and return from "final rule" to the end
    return ""  # Return empty string if "final rule" not found
