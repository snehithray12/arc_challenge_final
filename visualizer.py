from utils import get_json_filenames, visualize_single_pair
from vision import get_problem_setup
import random
import json


def random_visualization_arc():
    filenamelist = get_json_filenames("ARC-AGI/data/evaluation/")
    num_set = set()
    while len(num_set) <= len(filenamelist):
        rand_num = random.randrange(len(filenamelist))
        if rand_num in num_set:
            continue
        else:
            print("rand num ", rand_num)
            problem_setup = get_problem_setup(rand_num, filenamelist=filenamelist)
            input_observations = problem_setup["observed_inputs"]
            output_observations = problem_setup["observed_outputs"]
            for x in range(len(input_observations)):
                visualize_single_pair(input=input_observations[x], output=output_observations[x], title="Problem Viz")
            num_set.add(rand_num)
            print("on index ", rand_num)
            print("visualized ", len(num_set), " out of ", len(filenamelist))




def visualize_in_order(index = None):
    filenamelist = get_json_filenames("ARC-AGI/data/training/")
    if index == None:
        index = 0
    for i in range(index, len(filenamelist)):
        print("index ", i)
        problem_setup = get_problem_setup(i, filenamelist=filenamelist)
        input_observations = problem_setup["observed_inputs"]
        output_observations = problem_setup["observed_outputs"]
        for x in range(len(input_observations)):
            visualize_single_pair(input=input_observations[x], output=output_observations[x], title="Problem Viz")


def visualize_single_index(index, which_mode):
    problem_setup = get_problem_setup(index, which_mode)
    input_observations = problem_setup["observed_inputs"]
    output_observations = problem_setup["observed_outputs"]
    for x in range(len(input_observations)):
        visualize_single_pair(input=input_observations[x], output=output_observations[x], title="Problem Viz")



def visualize_multiple_pairs(index, title="Visualization of ARC Input-Output Pairs"):
    problem_setup = get_problem_setup(index, "train")
    pairs = []
    for x in range(len(problem_setup["observed_inputs"])):
        input_grid = problem_setup["observed_inputs"][x]
        output_grid = problem_setup["observed_inputs"][x]
        pairs.append((input_grid, output_grid))
    pairs.append((problem_setup["solution_input"], problem_setup["solution_output"]))
    cmap = ListedColormap(['black', 'yellow', 'red', 'blue', 'cyan', 'gray', 'magenta', 'pink', 'maroon', 'white', 'orange'])
    # Determine the grid size
    num_pairs = len(pairs)
    fig, axs = plt.subplots(num_pairs, 2, figsize=(10, 5 * num_pairs))
    # Loop through each pair
    for i, (input_data, output_data) in enumerate(pairs):
        axs[i, 0].imshow(input_data, cmap=cmap, vmin=0, vmax=10)
        axs[i, 0].set_title(f'Input {i+1}')
        axs[i, 1].imshow(output_data, cmap=cmap, vmin=0, vmax=10)
        axs[i, 1].set_title(f'Output {i+1}')
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the title
    plt.show()

