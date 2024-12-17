from argparse import Action
from prompts import create_context_for_caching, create_generate_code_prompt_body, create_mutate_code_prompt_body
from enum import Enum
from Program import *
import time
from datetime import datetime as dt
from ARC_objects import Grid, GridObject, update_object_numbers
from new_samplers import *
from actions import *
import numpy as np
import math
import traceback
import sys
import asyncio
from visualizer import visualize_single_index, visualize_feature_list
from anthropic import AsyncAnthropic, Anthropic
from itertools import permutations


def create_empty_program_object(group, problem_number):
    empty_params = get_empty_program_params()
    first_program = SameSizeObjectCentricProgram(group = group, problem_number = problem_number, parameters = empty_params)
    return first_program



CLAUDE_MODEL_ASYNC = AsyncAnthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

CLAUDE_MODEL_SYNC = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)



def get_rules_from_directory(directory):
    rules_dict_1 = {}
    rules_dict_2 = {}
    for filename in os.listdir(directory):
        if '.txt' in filename:
            question_number = int(filename.split('_')[5])
            vision_mode = int(filename.split('_')[3])
            text = open(directory + '/'+ filename).read()
            extracted_rule = extract_final_rule_section(text)
            if vision_mode == 1:
                if question_number in rules_dict_1:
                    rules_dict_1[question_number].append(extracted_rule)
                elif question_number not in rules_dict_1:
                    rules_dict_1[question_number] = [extracted_rule]
            elif vision_mode == 2:
                if question_number in rules_dict_2:
                    rules_dict_2[question_number].append(extracted_rule)
                elif question_number not in rules_dict_2:
                    rules_dict_2[question_number] = [extracted_rule]
    return rules_dict_1, rules_dict_2

##utility functions to access arc problems
def get_full_arc_set_train():
    filenamelist = get_json_filenames("ARC-AGI/data/training/")
    problem_set = []
    for x in range(len(filenamelist)):
        problem = get_problem_setup(index=x, which_mode = "train")
        problem_set.append(problem)
    return problem_set


##utility functions to access arc problems
def get_full_arc_set_eval():
    filenamelist = get_json_filenames("ARC-AGI/data/evaluation/")
    problem_set = []
    for x in range(len(filenamelist)):
        problem = get_problem_setup(index=x, which_mode = "eval")
        problem_set.append(problem)
    return problem_set

##gets all concept arc problems
def get_full_concept_arc_set():
    concept_arc_set = {}
    directory = 'ConceptARC/corpus/'
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for subdir in subdirectories:
        concept_arc_set[subdir] = get_problem_concept_arc(subdir=subdir)
    return concept_arc_set


##takes in problem and vision mode and creates an object centric same size prompt for those values
def create_arc_prompt(problem, problem_index, perception_mode, number_samples):
    rule_dict_1, rule_dict_2 = get_rules_from_directory('results/11_21_rule_search_results')
    rules_list = []
    if perception_mode == 1:
        rules_list = rule_dict_1[problem_index]
    elif perception_mode == 2:
        rules_list = rule_dict_2[problem_index]
    ##cycles through the rules in rules list based on which sample number we are on
    ##makes sure all rules have their turn being sampled in the prompt
    rule_text = rules_list[number_samples % (len(rules_list) - 1)]
    problem_inputs = problem['observed_inputs']
    problem_outputs = problem['observed_outputs']
    ###get prompt for generating code prompt body
    prompt = create_generate_code_prompt_body(rule_description = rule_text, input_grids = problem_inputs, output_grids = problem_outputs)
    return prompt





class SameSizeObjectCentricProgramSearchEnvironment:
    def __init__(self, group, index, perception_mode, target_population_size, time_limit, target_llm_programs):
        self._target_llm_programs = target_llm_programs
        self._finished_llm_generate = False
        self._finished_llm_mutate = False
        self._exit_early = False
        self._time_limit = time_limit
        ##training or eval are the two groups
        self._group = group
        ##index of the specific problem the program search is working on
        self._index = index
        #mode of vision that this instance of program search is working with
        self._perception_mode = perception_mode
        self._target_population_size = target_population_size
        if self._group == "eval":
            self._problem_set = get_full_arc_set_eval()
        else:
            self._problem_set = get_full_arc_set_train()
        ##the problem representation for the specific individual program we are running search on
        self._problem = self._problem_set[index]
        ##contains input Grid objects for the problem
        self._input_grid_objs = []
        ##contains output Grid objects for the problem
        self._output_grid_objs = []
        self.fill_in_obj_grids()
        ###runtime analytics values
        ##runtime analysis for llm generated programs
        self._number_llm_generated_programs = 0
        self._total_llm_generation_time = 0
        self._average_llm_generation_time = 0
        self._number_llm_query_errors = 0
        ##runtime analysis for randomly generated programs
        self._number_random_generated_programs = 0
        self._total_random_generation_time = 0
        self._average_random_generation_time = 0
        self._number_program_errors = 0
        self._error_makeup_dict = {}
        ##runtime analysis for llm mutations
        self._number_llm_mutations_programs = 0
        self._total_llm_mutations_time = 0
        self._average_llm_mutation_time = 0
        ##runtime analysis for random mutations
        self._number_random_mutations_programs = 0
        self._total_random_mutations_time = 0
        self._average_random_mutation_time = 0
        ##program population containers
        self._current_population = {}
        self._population_sorted_by_performance = []
        self._llm_generated_programs = []
        self._num_non_generated_program_errors = 0

    def get_group(self):
        return self._group

    def get_index(self):
        return self._index

    def get_perception_mode(self):
        return self._perception_mode

    def get_problem_set(self):
        return self._problem_set

    def get_problem(self):
        return self._problem

    def get_input_grid_objs(self):
        return self._input_grid_objs

    def get_output_grid_objs(self):
        return self._output_grid_objs

    def get_number_llm_generated_programs(self):
        return self._number_llm_generated_programs

    def get_total_llm_generation_time(self):
        return self._total_llm_generation_time

    def get_average_llm_generation_time(self):
        return self._average_llm_generation_time

    def get_number_random_generated_programs(self):
        return self._number_random_generated_programs

    def get_total_random_generation_time(self):
        return self._total_random_generation_time

    def get_average_random_generation_time(self):
        return self._average_random_generation_time

    def get_current_population(self):
        return self._current_population
    
    def get_number_program_errors(self):
        return self._number_program_errors
    
    def get_error_makeup_dict(self):
        return self._error_makeup_dict

    def get_number_llm_query_errors(self):
        return self._number_llm_query_errors

    def get_llm_generated_programs(self):
        return self._llm_generated_programs


    def visualize_problem(self, visualize_obj_list = False):
        for x in range(len(self._input_grid_objs)):
            input_grid_obj = self._input_grid_objs[x]
            output_grid_obj = self._output_grid_objs[x]
            input_grid = input_grid_obj.get_grid()
            output_grid = output_grid_obj.get_grid()
            if visualize_obj_list:
                print("input_grid: ")
                print(input_grid)
                print("output grid: ")
                print(output_grid)
                visualize_single_pair(input_grid, output_grid)
                feature_list_input = [obj.get_object_grid() for obj in input_grid_obj.get_object_list()]
                visualize_feature_list(input_grid, feature_list_input)
                feature_list_output = [obj.get_object_grid() for obj in output_grid_obj.get_object_list()]
                visualize_feature_list(output_grid, feature_list_output)
            else:
                print("input_grid: ")
                print(input_grid)
                print("output grid: ")
                print(output_grid)
                visualize_single_pair(input_grid, output_grid)

    ##for the problem specified for this search environment, fill the list of grid objects that make up this problem
    def fill_in_obj_grids(self):
        for x in range(len(self._problem["observed_inputs"])):
            input_grid_obj = Grid(self._problem["observed_inputs"][x], perception_mode = self._perception_mode)
            output_grid_obj = Grid(self._problem["observed_outputs"][x], perception_mode = self._perception_mode)
            self._input_grid_objs.append(input_grid_obj)
            self._output_grid_objs.append(output_grid_obj)
    


    ###uses LLM to randomly generate a new program 
    async def generate_new_program_llm(self, which_llm, abridge=False):
        while self._number_llm_generated_programs < 25:
            print("generating new program llm ", self._number_llm_generated_programs)
            start_time = time.time()
            context_prompt = create_context_for_caching()
            prompt = create_arc_prompt(problem=self._problem, problem_index = self._index, perception_mode = self._perception_mode, number_samples = self._number_llm_generated_programs)
            print("prompt is ")
            print(prompt)
            try:
                response = await prompt_anthropic_large_async(client = CLAUDE_MODEL_ASYNC, context = context_prompt, prompt = prompt)
                print("got claude response ")
            except Exception as e:
                print("Error occurred: ", str(e))
                self._handle_llm_error(e)
                return
            self._save_response_to_file(response, model_name="claude")  # Save Claude response
            self._record_generation_time(start_time)
            program, pred_output, loss = self._create_program_from_response(response)
            if pred_output not in self._current_population:
                self._add_program_to_population(pred_output, program, loss)



    def _save_response_to_file(self, response, model_name):
        filename = f"results/11_23_llm_generated_responses/llmresponse_vision_mode_{self._perception_mode}_question_{self._index}_number_{100 + self._number_llm_generated_programs}.txt"
        print("saving to file: ", filename)
        save_string_to_file(response, filename)

    def _record_generation_time(self, start_time):
        end_time = time.time()
        self._total_llm_generation_time += (end_time - start_time)
        self._number_llm_generated_programs += 1
        self._average_llm_generation_time = self._total_llm_generation_time / self._number_llm_generated_programs
        print("on problem: ", self._index, " vision mode: ", self._perception_mode)
        print("average LLM generation time: ", self._average_llm_generation_time, "generated number problems is: ", self._number_llm_generated_programs)

    def _create_program_from_response(self, response):
        params = create_params_from_llm_text_sample(response)
        program = SameSizeObjectCentricProgram(group=self._group, problem_number=self._index, parameters=params)
        pred_output, loss = self.run_program_inputs(program)
        return program, pred_output, loss

    def _add_program_to_population(self, pred_output, program, loss):
        self._current_population[pred_output] = (program, loss)
        self._llm_generated_programs.append(program)

    def _handle_llm_error(self, error):
        error_str = str(error)
        self._error_makeup_dict[error_str] = self._error_makeup_dict.get(error_str, 0) + 1
        self._number_llm_query_errors += 1


    
    ##randomly generates seed programs 
    def generate_new_program_random(self, action_set, properties_set):
        start_time = time.time()
        params = get_empty_program_params()
        if "add_new_object" in action_set:
            function_str = sample_add_new_object(input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
            params["add_block"] = function_str
        program = SameSizeObjectCentricProgram(group = self._group, problem_number = self._index, parameters = params)
        ###default generation is to call add checker trans pair twice
        program = add_checker_transformation_block(program = program, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
        end_time = time.time()
        pred_output_str, loss = self.run_program_inputs(program)
        if pred_output_str not in self._current_population:
            self._current_population[pred_output_str] = (program, loss)
    


    ##takes in program and tries to run it on the inputs for this problem
    ##returns the outputs of the program on each of the input grids all in one data structure
    def run_program_inputs(self, program):
        pred_outputs_str = ""
        pred_outputs_list = []
        has_error = False
        for i in range(len(self._input_grid_objs)):
            input_grid_obj = self._input_grid_objs[i]
            grid = input_grid_obj.get_grid()
            try:
                pred_output = program.run_string_program(grid)
                pred_outputs_str += "{}".format(pred_output) + '\n'
                pred_outputs_list.append(pred_output)
            except Exception as e:
                pred_outputs_str += "Error" + '\n'
                pred_outputs_list.append(None)
                has_error = True
                if str(e) in self._error_makeup_dict:
                    self._error_makeup_dict[str(e)] += 1
                else:
                    self._error_makeup_dict[str(e)] = 1
        loss = self.calculate_loss(pred_outputs_list)
        if has_error:
            self._number_program_errors += 1
        return pred_outputs_str, loss
    
    ##calculates loss for program based on its predicted outputs
    def calculate_loss(self, pred_outputs):
        total_loss = 0
        num_errors = 0
        num_correct = 0
        num_blanks = 0
        for x in range(len(self._output_grid_objs)):
            output_grid_obj = self._output_grid_objs[x]
            input_grid_obj = self._input_grid_objs[x]
            ##actual answer
            output_grid = output_grid_obj.get_grid()
            input_grid = input_grid_obj.get_grid()
            ##predicted output
            pred_output = pred_outputs[x]
            ##if pred_output is None (because the program errors out), treat it like it got every cell wrong
            ##on this question
            if pred_output is None:
                num_errors += 1
                total_loss += (input_grid.shape[0] * output_grid.shape[1])
                continue
            ##if there is a match between predicted output and actual output grid, it got that one correct. continue.
            if np.array_equal(pred_output, output_grid):
                num_correct += 1
                continue
            ##if there is a match between predicted output and input grid when they aren't supposed to be the same
            ##(which would be caught by the previous statement), bad program because it's not doing anything. penalize harshly.
            elif np.array_equal(pred_output, input_grid):
                total_loss += (input_grid.shape[0] * output_grid.shape[1])
                num_blanks += 1
                continue
            ##if the predicted output is all blank when it is not supposed to be (if it was, it would have been caught earlier), then
            ##penalize harshly. 
            elif np.array_equal(pred_output, np.zeros(pred_output.shape)):
                total_loss += float('inf')
            ##if the shapes don't line up
            if (pred_output.shape != output_grid.shape):
                total_loss = float('inf')
                continue
            ##increment total loss value with the hamming distance between the predicted output and actual loss
            total_loss += np.sum(pred_output != output_grid)
            ###section for object level accuracy checking. 
            obj_array_list = perform_object_detection_type(grid = pred_output, type_vision = self._perception_mode)
            in_input_and_output = 0
            not_in_input_in_output = 0
            not_in_output = 0
            ##looping through detected obj numpy array grids found in the predicted output
            ##want to look for cases where the program was able to "discover" objects in the output 
            ##grid with its rule (i.e it found an object in output grid that isnt in the input grid)
            ##OR cases where it has an object not in the output grid
            for obj_grid in obj_array_list:
                in_input = False
                in_output = False
                for obj_in_output in output_grid_obj.get_object_list():
                    obj_in_output_grid = obj_in_output.get_object_grid()
                    ##if it matches something in the output grid
                    if np.array_equal(obj_grid, obj_in_output_grid):
                        in_output = True
                        break
                for obj_in_input in input_grid_obj.get_object_list():
                    obj_in_input_grid = obj_in_input.get_object_grid()
                    ##check and see if it wasn't already in the input grid
                    if np.array_equal(obj_grid, obj_in_input_grid):
                        in_input = True
                        break
                if in_input and in_output:
                    in_input_and_output += 1
                elif not in_input and in_output:
                    not_in_input_in_output += 1
                elif not in_output:
                    not_in_output += 1
            ##reward discovering new objects.
            total_loss -= (10 * not_in_input_in_output)
            ##penalize having objects in predicted output that arent in actual output
            total_loss += (10 * not_in_output)
        ##if it errors on all of the input grids OR does not do anything to any of the inputs, it is a garbage program
        if num_errors == len(self._output_grid_objs) or num_blanks == len(self._output_grid_objs):
            return float('inf')
        ##minimum possible loss if all of the answers are correct
        if len(self._output_grid_objs) == num_correct:
            return float('-inf')
        print("total loss: ", total_loss)
        return float(total_loss / len(self._input_grid_objs))


    ##randomly chooses a program. Fifty fifty split between LLM generated programs and rest of population
    ##enforced so LLM generated programs don't get drowned out
    def choose_program(self):
        choice = random.randint(1,2)
        program = None
        if len(self._current_population) == 0:
            return
        if choice == 1:
            if len(self._llm_generated_programs) > 0:
                program = random.choice(self._llm_generated_programs)
            else:
                if len(self._current_population) >= 1:
                    random_key = random.choice(list(self._current_population.keys()))
                    program = self._current_population[random_key][0]
        elif choice == 2:
            ##make it more biased to pick a higher performing program
            if len(self._population_sorted_by_performance) > 0:
                weights = [1 / (i + 1) for i in range(len(self._population_sorted_by_performance))]
                selected_value = random.choices(self._population_sorted_by_performance, weights=weights, k=1)[0]
                program, loss, key = selected_value
            elif len(self._current_population) >= 1:
                random_key = random.choice(list(self._current_population.keys()))
                program = self._current_population[random_key][0]
        return program
    

    ##mutate programs in current population
    def run_standard_mutation_program(self, action_set, properties_set):
        program = self.choose_program()
        program_list = []
        if program is None:
            return
        start_time = time.time()
        rand_index = random.randint(1, len(self._input_grid_objs))
        rand_index -= 1
        try:
            ###run on just one input for efficiency, use it for search heuristics
            pred_output = program.run_string_program(self._input_grid_objs[rand_index].get_grid())
            ###creating a Grid object of the predicted output
            pred_output_obj = Grid(grid = pred_output, perception_mode = self._perception_mode)
        except:
            ##this program is an error causing one and is not worth mutating
            return
        ##the output we are comparing with for the search heuristics
        actual_output_obj = self._output_grid_objs[rand_index]
        program_1 = mutate_obj_checker(program, input_grid_objs = self._input_grid_objs, properties_set = properties_set)
        if not compare_code_strings(program_1.get_text(), program.get_text()):
            program_list.append(program_1)
        program_2 = add_checker_transformation_block(program, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
        if not compare_code_strings(program_2.get_text(), program.get_text()):
            program_list.append(program_2)
        ###only add a trans block section if the number of objects is fewer in the prediction than the actual
        if len(pred_output_obj.get_object_list()) < len(actual_output_obj.get_object_list()):
            program_3 = add_trans_object_block(program, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set)
            if not compare_code_strings(program_3.get_text(), program.get_text()):
                program_list.append(program_3)      
        program_4 = mutate_trans_function(program, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
        if not compare_code_strings(program_4.get_text(), program.get_text()):
            program_list.append(program_4)
        ##only mutate add block section if that section is populated/there is a function for it
        if len(program.get_parameters()["add_block"]) > 0:
            program_5 = mutate_add_new_function(program = program, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
            if not compare_code_strings(program_5.get_text(), program.get_text()):
                self._number_random_mutations_programs += 1
                program_list.append(program_5)
        program_6 = remove_checker_trans_pair(program)
        if not compare_code_strings(program_6.get_text(), program.get_text()):
            program_list.append(program_6)
        program_7 = remove_transformation_snippet(program)
        if not compare_code_strings(program_7.get_text(), program.get_text()):
            program_list.append(program_7)
        end_time = time.time()
        self._number_random_mutations_programs += 6
        self._total_random_mutations_time += (end_time - start_time)
        self._average_random_mutation_time = self._total_random_generation_time/self._number_random_mutations_programs
        for program in program_list:
            pred_output_str, loss = self.run_program_inputs(program)
            if pred_output_str not in self._current_population:
                self._current_population[pred_output_str] = (program, loss)
    
    ##mates 2 programs. if the output is NOT something we haven't already seen,
    ##add it to the population.
    def run_program_mating(self):
        program_1 = self.choose_program()
        program_2 = self.choose_program()
        if program_1 is None or program_2 is None:
            return
        new_program = mate_programs(program_1, program_2)
        pred_output_str, loss = self.run_program_inputs(new_program)
        if pred_output_str not in self._current_population:
            self._current_population[pred_output_str] = (new_program, loss)

    ###runs LLM based mutation
    def mutate_program_llm(self, program = None):
        input_list = []
        input_objects_list = []
        ##looping through input grid objects in this problem
        for input_grid_obj in self._input_grid_objs:
            input_object_list = []
            ##adding grid representation of input grid object to list
            input_list.append(input_grid_obj.get_grid())
            ##looping through objects representations of GridObjects in Grid
            for input_obj in input_grid_obj.get_object_list():
                ##adding GridObject grid to input_object_list
                input_object_list.append(input_obj.get_object_grid())
            input_objects_list.append(input_object_list)
        ##do the same with output grid objects
        actual_output_list = []
        output_objects_list = []
        for output_grid_obj in self._output_grid_objs:
            output_object_list = []
            ##adding grid representation of output grid object to list
            actual_output_list.append(output_grid_obj.get_grid())
            ##looping through objects representations of GridObjects in Grid
            for output_obj in output_grid_obj.get_object_list():
                ##adding GridObject grid to output_object_list
                output_object_list.append(output_obj.get_object_grid())
            output_objects_list.append(output_object_list)
        ###only choose the top performing programs
        sorted_performance_list = self.sort_by_performance()
        index = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        ##checks for lengths of sorted performance list
        if index < len(sorted_performance_list):
            current_program_obj = sorted_performance_list[index][0]
        elif len(sorted_performance_list) >= 1:
            current_program_obj = sorted_performance_list[0][0]
        else:
            return
        ##getting predicted outputs of chosen program
        if program is not None:
            current_program_obj = program
        pred_output_list = []
        for input_grid in input_list:
            try:
                pred_output = current_program_obj.run_string_program(input_grid)
            except Exception as e:
                pred_output = str(e)
            pred_output_list.append(pred_output)
        current_program_str = current_program_obj.get_display_text()
        mutate_prompt = create_fill_in_partial_program_code_prompt_body(input_list = input_list, actual_output_list = actual_output_list, pred_output_list = pred_output_list, current_program = current_program_str)
        context_prompt = create_context_for_caching()
        for x in range(10):
            try:
                response = prompt_anthropic_large_sync(client = CLAUDE_MODEL_SYNC, context = context_prompt, prompt = mutate_prompt)
                filename = 'results/11_26_llm_generated_responses/' + 'llmresponse_vision_mode_{}_question_{}_number_{}.txt'.format(self._perception_mode, self._index, x)
                save_string_to_file(response, filename)
                print("got claude response number {} ".format(x))
            except Exception as e:
                print("Error occurred: ", str(e))
                continue





    ##sorts current population by loss value, lowest to highest
    def sort_by_performance(self):
        ##if there are no programs in the population currently, exit. 
        if len(self._current_population) == 0:
            return
        program_loss_list = []
        for keys in self._current_population:
            program, loss = self._current_population[keys]
            program_loss_list.append((program, loss, keys))
        sorted_tuples = sorted(program_loss_list, key=lambda x: x[1])
        self._population_sorted_by_performance = sorted_tuples
        return self._population_sorted_by_performance
    

    ##flush the poor performing programs from the population.
    def flush_bad_programs(self):
        start_time = time.time()
        ##kill half of the worst performing programs each time
        threshold = self._target_population_size/2
        if len(self._current_population) < threshold:
            return
        ##start from programs with highest loss
        start_time_2 = time.time()
        sorted_programs = self.sort_by_performance()
        end_time_2 = time.time()
        new_program_list = copy.deepcopy(sorted_programs)[:int(threshold)]
        new_population_dict = {}
        ##clear out the ____ number of most poorly performing programs
        for tup in new_program_list:
            program, loss, key = tup
            if key not in new_population_dict:
                new_population_dict[key] = (program, loss)
        ##set new population to new dict with bad programs removed
        self._current_population = new_population_dict
        end_time = time.time()



    ###runs standard random mutations and standard program mating while waiting for a single 
    ##LLM call/set of LLM calls to complete
    async def run_random_changes(self, action_set, properties_set):
        last_time_pop_size_changed = time.time()
        pop_size_before = len(self._current_population)
        try:
            self.run_standard_mutation_program(action_set, properties_set)
            self.run_program_mating()
            ##this gives control back to the event manager that handles the other asynchronous functions
            if pop_size_before < len(self._current_population):
                last_time_pop_size_changed = time.time()
            current_time = time.time()
            time_elapsed = current_time - last_time_pop_size_changed
            await asyncio.sleep(0)
        except Exception as e:
            ###add error string to dictionary
            if str(e) in self._error_makeup_dict:
                self._error_makeup_dict[str(e)] += 1
            else:
                self._error_makeup_dict[str(e)] = 1
            self._num_non_generated_program_errors += 1
            await asyncio.sleep(0)

    async def run_enumerative_search(self, is_async = False, action_set = None, properties_set = None):
        if action_set is None:
            action_set = prune_action_set(self._input_grid_objs, self._output_grid_objs)
        if properties_set is None:
            properties_set = get_properties_domain(self._input_grid_objs)
        ##if action set is not the only one in the grid, remove it. It will make search space explode bc of the combanitorics. 
        if 'add_new_object' in action_set and len(action_set) > 1:
            action_set.remove('add_new_object')
        ###heuristic has a tendency to count extend when there isnt one. At the expense of not searching everything, we are focusing on 
        ##a few core operations alone to make the search space manageable. 
        if 'translate' in action_set or 'flip' in action_set or 'rotate_in_place' in action_set:
            if 'extend' in action_set:
                action_set.remove('extend')
        ##another way to constrain search space. Only represent an action as a flip OR a rotation. Not both. 
        if 'rotate_in_place' in action_set and 'flip' in action_set:
            action_set.remove('rotate_in_place')
        ###start with an empty program
        empty_params = get_empty_program_params()
        if "add_new_object" in action_set:
            function_str = sample_add_new_object(input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
            empty_params["add_block"] = function_str
        first_program = SameSizeObjectCentricProgram(group = self._group, problem_number = self._index, parameters = empty_params)
        ###gets list of all possible trans function implementations
        trans_function_domain = get_trans_function_domain(number = 1, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
        obj_checker_domain = get_total_obj_checker_domain(input_grid_objs = self._input_grid_objs, properties_set = properties_set, constrain = False)
        start_time = time.time()
        counter = 0
        total =  len(obj_checker_domain) * len(trans_function_domain)
        if total > 5000:
            obj_checker_domain = get_total_obj_checker_domain(input_grid_objs = self._input_grid_objs, properties_set = properties_set, constrain = True)
        total =  len(obj_checker_domain) * len(trans_function_domain)       
        # self.visualize_problem(visualize_obj_list = True)
        for obj_checker in obj_checker_domain:
            for trans_function in trans_function_domain:
                counter += 1
                print("on {} out of {}".format(counter, total))
                new_program = insert_checker_trans_enumerative_search(program = first_program, input_grid_objs = self._input_grid_objs, number = 1, obj_checker = obj_checker, trans_function = trans_function)
                pred_output_str, loss = self.run_program_inputs(new_program)
                if pred_output_str not in self._current_population:
                    self._current_population[pred_output_str] = (new_program, loss)
                ##if we are running enumerative search asynchronously, have it sleep so we can check on other tasks
                if is_async:
                    await asyncio.sleep(0)
        end_time = time.time()
        print("action set: ", action_set)
        print("number of programs: ", total)
        print("time taken to enumerate: ", end_time - start_time)
        print("population size: ", len(self._current_population))
        loop_limit = 1
        self.sort_by_performance()
        return self._population_sorted_by_performance[0][0]



    def local_search(self, program, action_set = None, properties_set = None):
        old_params = program.get_parameters()
        new_params = copy.deepcopy(old_params)
        if action_set is None:
            action_set = prune_action_set(self._input_grid_objs, self._output_grid_objs)
        if properties_set is None:
            properties_set = get_properties_domain(self._input_grid_objs)
        trans_function_domain = get_trans_function_domain(number = 1, input_grid_objs = self._input_grid_objs, output_grid_objs = self._output_grid_objs, action_set = action_set, properties_set = properties_set)
        obj_checker_domain = get_total_obj_checker_domain(input_grid_objs = self._input_grid_objs, properties_set = properties_set, constrain = False)
        new_obj_checker_domain = constrain_obj_checker_domain_llm(llm_obj_checkers = program.get_obj_checkers(), obj_checker_domain = obj_checker_domain)
        groupings = list(permutations(new_obj_checker_domain, len(program.get_obj_checkers())))
        counter = 0
        for grouping in groupings:
            counter += 1
            print("on {} out of {} ".format(counter, len(groupings)))
            obj_checker_list = ["obj_checker_{} = ".format(x + 1) + grouping[x].strip() for x in range(len(grouping))]
            new_params["obj_checkers"] = obj_checker_list
            new_program = SameSizeObjectCentricProgram(group = program.get_group(), problem_number = program.get_problem_number(), parameters = new_params)
            pred_output_str, loss = self.run_program_inputs(new_program)
            if pred_output_str not in self._current_population:
                self._current_population[pred_output_str] = (new_program, loss)
        self.sort_by_performance()

    ###runs the above asynchronous program generation plus mutation in a loop until population size reaches 
    ###desired amount
    async def run_single_epoch(self, first_epoch, action_set, properties_set):
        if first_epoch:
            await asyncio.gather(
                self.generate_new_program_llm(which_llm="claude", abridge = False),
                self.run_random_changes(first_epoch, action_set, properties_set)
                )
        else:
            await asyncio.gather(
                self.generate_new_program_llm(which_llm="claude", abridge = False),
                self.run_random_changes(first_epoch, action_set, properties_set)
                )


    ###run program search with only LLM prompting and storage
    def run_program_search_llm_only(self, number_prompts):
        found_answer = False
        while (self._number_llm_generated_programs < number_prompts) and not found_answer:
            self.generate_new_program_llm(which_llm = "claude")
            if len(self._current_population) > 0:
                self.sort_by_performance()
                best_program = self._population_sorted_by_performance[0][0]
                best_performance = self._population_sorted_by_performance[0][1]
                if best_performance == float('-inf'):
                    found_answer = True
                    program_text = best_program.get_text()
                    filename = 'results/11_23_program_search_results/' + 'program_vision_mode_{}_question_{}_number_{}.txt'.format(self._perception_mode, self._index, 1)
                    save_string_to_file(program_text, filename)
                    return best_program


    async def run_non_llm_search_funcs(self, action_set, properties_set):
        self.run_enumerative_search(is_async = True, action_set = action_set, properties_set = properties_set)
        await asyncio.sleep(0)
        while self._number_llm_generated_programs < 25:
            self.run_random_changes(action_set = action_set, properties_set = properties_set)
            await asyncio.sleep(0)


    async def run_program_search_async(self, action_set, properties_set):
        await asyncio.gather(self.run_non_llm_search_funcs(action_set, properties_set), self.generate_new_program_llm(which_llm = "claude", abridge = False))









    ###main loop that runs the program search. keeps generating, mutating, killing programs until correct answer is found
    ##or time runs out   
    def run_program_search(self, verbose = True):
        action_set = prune_action_set(self._input_grid_objs, self._output_grid_objs)
        properties_set = get_properties_domain(self._input_grid_objs)
        asyncio.run(self.run_program_search_async(action_set, properties_set))




    
    ##tool to help visualize how the programs are performing
    def visualize_program_performance(self):
        self.sort_by_performance()
        sorted_programs = self._population_sorted_by_performance
        for x in range(len(sorted_programs)):
            tups = sorted_programs[x]
            print("program number {} ===================================".format(x + 1))
            program, loss = tups
            print("Program string below: ")
            print(program.get_display_text())
            print("Loss value: ", loss)
            for y in range(len(self._input_grid_objs)):
                input_grid = self._input_grid_objs[y].get_grid()
                output_grid = self._output_grid_objs[y].get_grid()
                pred_output = program.run_string_program(input_grid)
                visualize_single_pair(input_grid, output_grid, "Actual Output ")
                visualize_single_pair(input_grid, pred_output, "Predicted Output for program number ".format(x + 1))
            print("End program -----------------------------------")

    ##prints all of the relevant analytics values
    def display_analytics(self):
        print("Analytics Overview")
        print("------------------")
        print(f"Number of LLM Generated Programs: {self.get_number_llm_generated_programs()}")
        print(f"Number of LLM Query Errors: {self.get_number_llm_query_errors()}")
        print(f"Total LLM Generation Time: {self.get_total_llm_generation_time()} seconds")
        print(f"Average LLM Generation Time: {self.get_average_llm_generation_time()} seconds")
        print(f"Number of Randomly Generated Programs: {self._number_random_mutations_programs}")
        print(f"Total Random Generation Time: {self._total_random_mutations_time} seconds")
        print(f"Average Random Generation Time: {self._average_random_mutation_time} seconds")
        print(f"Number of LLM Mutated Programs: {self._number_llm_mutations_programs}")
        print(f"Total LLM Mutation Time: {self._total_llm_mutations_time} seconds")
        print(f"Average LLM Mutation Time: {self._average_llm_mutation_time} seconds")
        print(f"Number of Program Errors: {self.get_number_program_errors()}")
        print(f"Error Makeup Dictionary: {self.get_error_makeup_dict()}")
    
        
env = SameSizeObjectCentricProgramSearchEnvironment(group = "eval", index = 89, perception_mode = 1, target_population_size = 10, time_limit = 10, target_llm_programs= 10)

env.run_program_search()