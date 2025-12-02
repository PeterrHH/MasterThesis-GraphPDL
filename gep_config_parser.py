#! Code taken from Matthijs Arnoldus' thesis (transferred to Python from Julia)

from typing import Dict, Any
import os
import toml
import pandas as pd
from pyomo.environ import ConcreteModel, SolverFactory

AUTO                 = "auto"
FROM                 = "from"
TO                   = "to"

OPTIMIZER_TABLE      = "optimizer"
INPUTS_TABLE         = "inputs"
SCALARS_TABLE        = "scalars"
DATA_TABLE           = "data"
SETS_TABLE           = "sets"
OUTPUTS_TABLE        = "outputs"

SOLVER_KEY           = "solver"
DIR_KEY              = "dir"
FILE_KEY             = "file"

DEMAND_KEY           = "demand"
GENERATION_KEY       = "generation"
GENERATION_AVAIL_KEY = "generation_availability"
TRANSMISSION_KEY     = "transmission_lines"
TIMES_KEY            = "times"
NODES_KEY            = "nodes"
GENERATORS_KEY       = "generators"
LINES_KEY            = "transmission_lines"

SOLVER_UNSUPPORTED_MESSAGE = \
    "Only Gurobi and HIGHS solvers are supported at the moment."
WRONG_SET_MESSAGE = \
    "Wrong set description. Either provide a list, or use `auto` to infer it."

# Helper functions
def prepend_dir(directory: str, path: str) -> str:
    path = os.path.join(directory, path)
    return path

def read_config(file_name: str) -> Dict[str, Any]:
    return toml.load(file_name)

def initialize_model(optimizer, attributes):
    # Initialize the Pyomo model
    model = ConcreteModel()

    # Set up the optimizer
    solver = SolverFactory(optimizer)

    # Configure solver attributes
    for key, value in attributes.items():
        solver.options[key] = value

    # Return the model and solver
    return model, solver

def read_scalars(scalars_config: Dict[str, Any], inputs_dir: str) -> Dict[str, Any]:
    file_path = prepend_dir(inputs_dir, scalars_config[FILE_KEY])
    scalars = toml.load(file_path)
    return scalars

def read_data(data_config: Dict[str, Any], inputs_dir: str) -> Dict[str, pd.DataFrame]:
    demand_data = pd.read_csv(prepend_dir(inputs_dir, data_config[DEMAND_KEY]))
    generation_data = pd.read_csv(prepend_dir(inputs_dir, data_config[GENERATION_KEY]))
    generation_availability_data = pd.read_csv(prepend_dir(inputs_dir, data_config[GENERATION_AVAIL_KEY]))
    transmission_lines_data = pd.read_csv(prepend_dir(inputs_dir, data_config[TRANSMISSION_KEY]))

    return {
        "demand_data": demand_data,
        "generation_data": generation_data,
        "generation_availability_data": generation_availability_data,
        "transmission_lines_data": transmission_lines_data,
    }

#! Unnecessary in Python
def read_sets(sets_config: Dict[str, Any]) -> Dict[str, Any]:
    return sets_config

def resolve_times(dictionary: Dict[str, Any]) -> None:
    times = dictionary.get(TIMES_KEY)
    if isinstance(times, str):
        if times != AUTO:
            raise ValueError(f"Invalid value for times: {times}. {WRONG_SET_MESSAGE}")

        from_time = min(
            dictionary["demand_data"]["Time"].min(),
            dictionary["generation_availability_data"]["Time"].min()
        )
        to_time = max(
            dictionary["demand_data"]["Time"].max(),
            dictionary["generation_availability_data"]["Time"].max()
        )
    
    elif isinstance(times, dict):
        from_time = times.get("from")
        to_time = times.get("to")
        if from_time is None or to_time is None:
            raise ValueError("Both 'from' and 'to' must be specified in the times dictionary.")
    elif isinstance(times, int):
        from_time = times
        to_time = times
    else:
        raise ValueError(f"Invalid type for times: {type(times)}. {WRONG_SET_MESSAGE}")

    dictionary[TIMES_KEY] = range(from_time, to_time + 1)

def resolve_nodes(dictionary: Dict[str, Any]) -> None:
    nodes = dictionary[NODES_KEY]
    if isinstance(nodes, str):
        if nodes != AUTO:
            raise ValueError(f"Invalid value for nodes: {nodes}. {WRONG_SET_MESSAGE}")
        
        nodes = set(dictionary["demand_data"]["Country"].unique())
        nodes.update(dictionary["generation_data"]["Country"].unique())
        nodes.update(dictionary["generation_availability_data"]["Country"].unique())
        nodes.update(dictionary["transmission_lines_data"]["CountryA"].unique())
        nodes.update(dictionary["transmission_lines_data"]["CountryB"].unique())

    elif not isinstance(nodes, list) or not all(isinstance(n, str) for n in nodes):
        raise ValueError(f"Invalid type for nodes: {type(nodes)}. {WRONG_SET_MESSAGE}")
    
    dictionary["nodes"] = list(sorted(nodes))

def resolve_generators(dictionary: Dict[str, Any]) -> None:
    generators = dictionary[GENERATORS_KEY]

    if isinstance(generators, str):
        if generators != "auto":
            raise ValueError(f"Invalid value for generators: {generators}. {WRONG_SET_MESSAGE}")

        gen_data = dictionary["generation_data"][["Country", "Technology"]]
        gen_avail_data = dictionary["generation_availability_data"][["Country", "Technology"]]

        generators = set(map(tuple, gen_data.values))
        generators.update(map(tuple, gen_avail_data.values))

    elif not (isinstance(generators, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in generators)):
        raise ValueError(f"Invalid type for generators: {type(generators)}. {WRONG_SET_MESSAGE}")

    dictionary["generators"] = list(sorted(generators))

def resolve_transmission_lines(dictionary: Dict[str, Any]) -> None:
    lines = dictionary[TRANSMISSION_KEY]

    if isinstance(lines, str):
        if lines != "auto":
            raise ValueError(f"Invalid value for transmission_lines: {lines}. {WRONG_SET_MESSAGE}")

        transmission_data = dictionary["transmission_lines_data"][["CountryA", "CountryB"]]
        lines = set(map(tuple, transmission_data.values))
    elif not (isinstance(lines, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in lines)):
        raise ValueError(f"Invalid type for transmission_lines: {type(lines)}. {WRONG_SET_MESSAGE}")

    dictionary["transmission_lines"] = list(sorted(lines))

def resolve_sets(dictionary: Dict[str, Any]) -> None:
    resolve_times(dictionary)
    resolve_nodes(dictionary)
    resolve_generators(dictionary)
    resolve_transmission_lines(dictionary)

def read_inputs(inputs_config: Dict[str, Any]) -> Dict[str, Any]:
    inputs_dir = prepend_dir(os.path.dirname(__file__), inputs_config[DIR_KEY])

    scalars_config = inputs_config[SCALARS_TABLE]
    scalars = read_scalars(scalars_config, inputs_dir)

    data_config = inputs_config[DATA_TABLE]
    data = read_data(data_config, inputs_dir)

    sets_config = inputs_config["sets"]
    sets = read_sets(sets_config)

    rounding = inputs_config["rounding"]
    crossover = inputs_config["crossover"]
    output_file = inputs_config["output_file"]
    output_log = inputs_config["output_log"]
    relaxed = inputs_config["relaxed"]
    ramping = inputs_config["ramping"]

    result = {
        **scalars,
        **data,
        **sets,
        "crossover": crossover,
        "rounding": rounding,
        "output_file": output_file,
        "output_log": output_log,
        "ramping": ramping,
        "relaxed": relaxed,
    }

    # Resolve sets if they are set to "auto"
    resolve_sets(result)
    return result


def read_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    repeats = experiment_config["repeats"]

    # Process each input configuration
    experiments = []
    for inputs_config in experiment_config["inputs"]:
        inputs = read_inputs(inputs_config)
        experiments.append(inputs)

    # Combine results
    result = {
        "repeats": repeats,
        "experiments": experiments,
    }

    return result

def parse_config(file_name: str) -> Dict[str, Any]:
    config = read_config(file_name)

    # Parse the experiment configuration
    experiment_config = config["experiment"]
    experiment = read_experiment(experiment_config)

    return {
        "experiment": experiment,
        "optimizer_config": config["optimizer"],
        "outputs_config": config["outputs"]
    }

def get_visualization_data(file_name: str) -> Dict[str, Any]:
    return toml.load(file_name)

