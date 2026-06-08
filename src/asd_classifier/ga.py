"""Genetic Algorithm (GA) hyperparameter search for the GoogLeNet FC head.

Uses DEAP to evolve a population of ``Individual`` objects that encode the
number of FC layers, epoch budget, and per-layer neuron/dropout configuration.
Fitness is the best validation accuracy achieved during training.

Key design choices:
- Elitism: top 10 % of the population is carried forward unchanged.
- Tournament selection (size 3) for the remainder.
- Crossover swaps control genes and blends FC-gene pools; mutation flips
  layer-type, perturbs dropout, and adjusts the epoch count by ±10.
- Crossover probability decays, mutation probability grows each generation.
- Checkpoint saved after every generation so interrupted runs can resume.

Entry point: ``run_evolution()``.  Call ``set_loaders()`` before invoking it.
"""

import csv
import gc
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deap import base, creator, tools
from torch.utils.data import DataLoader

from asd_classifier.config import (
    BEST_MODEL_DIR,
    CHECKPOINT_PATH,
    GA_CXPB,
    GA_MUPB,
    GA_NGEN,
    GA_POPULATION_SIZE,
    GENETIC_RESULTS_DIR,
    control_ranges,
    fc_ranges,
)
from asd_classifier.model import GoogleNet
from asd_classifier.trainer import TrainerAndEvaluation

# Module-level data loaders — set by train() in main.py before calling run_evolution
_train_loader: DataLoader | None = None
_val_loader: DataLoader | None = None
_test_loader: DataLoader | None = None


def set_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Inject data loaders into module-level globals used by ``eval_val_accuracy``.

    Must be called once before ``run_evolution()``.
    """
    global _train_loader, _val_loader, _test_loader
    _train_loader = train_loader
    _val_loader = val_loader
    _test_loader = test_loader


# DEAP type registration (idempotent)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def generate_control_genes(kwargs: dict) -> list:
    """Sample one value per entry in *kwargs* using the ``(low, high[, step])`` range spec."""
    data = []
    for _key, value in kwargs.items():
        low, high, *step = value
        step = step[0] if step else 1
        data.append(random.choice(range(low, high + 1, step)))
    return data


def generate_parametric_genes(kwargs: dict, num_layer: int) -> list:
    """Generate *num_layer* FC-gene lists from the search-space spec in *kwargs*.

    Callable values in *kwargs* are treated as neuron-pair generators; other
    values use the ``(low, high[, step])`` range spec.
    """
    if not num_layer:
        raise ValueError("num_layer must be specified to generate layers.")
    data = []
    for _ in range(num_layer):
        layer_data = []
        for _key, value in kwargs.items():
            if callable(value):
                pairs = value()
                pair = random.choice(pairs)
                high, low = pair
                layer_data.append(random.randint(low, high))
            else:
                low, high, *step = value
                step = step[0] if step else 1
                layer_data.append(random.choice(range(low, high + 1, step)))
        data.append(layer_data)
    return data


def individual_generator() -> list:
    """Create one random GA individual as ``[control_gene, [fc_gene, ...]]``."""
    control_gene = generate_control_genes(control_ranges)
    fc_gene = generate_parametric_genes(fc_ranges, control_gene[0])
    return [control_gene] + [fc_gene]


def mate(ind1: list, ind2: list) -> tuple[list, list]:
    """Crossover two individuals by swapping control genes and blending FC-gene pools.

    Each offspring's FC layers are drawn via weighted sampling from the merged
    layer pool of both parents (favouring higher-neuron layers), padded with
    blended layers if the pool runs short.  Layers are sorted descending by
    neuron count to maintain the architecture convention.
    """
    offspring1 = toolbox.clone(ind1)
    offspring2 = toolbox.clone(ind2)

    for i in range(len(offspring1[0])):
        offspring1[0][i], offspring2[0][i] = offspring2[0][i], offspring1[0][i]

    for offspring in [offspring1, offspring2]:
        target_layers = offspring[0][0]
        layer_pool = []
        for layer in ind1[1] + ind2[1]:
            if layer not in layer_pool:
                layer_pool.append(layer)
        layer_pool.sort(key=lambda x: (-x[1], x[2]))

        selected: list = []
        while len(selected) < target_layers and layer_pool:
            weights = [1 / (i + 1) for i in range(len(layer_pool))]
            chosen = random.choices(layer_pool, weights=weights, k=1)[0]
            selected.append(chosen)
            layer_pool.remove(chosen)

        while len(selected) < target_layers:
            blended = [
                random.choice([ind1[1], ind2[1]])[0][0],
                int(np.mean([random.choice(ind1[1])[1], random.choice(ind2[1])[1]])),
                int(np.clip(
                    np.mean([random.choice(ind1[1])[2], random.choice(ind2[1])[2]]),
                    fc_ranges["dropout"][0],
                    fc_ranges["dropout"][1],
                )),
            ]
            selected.append(blended)

        offspring[1] = sorted(selected[:target_layers], key=lambda x: -x[1])

    return offspring1, offspring2


def mutate(ind: list) -> tuple[list]:
    """Mutate an individual in-place (returns a 1-tuple per DEAP convention).

    Flips the number of FC layers (1↔2), perturbs epoch count by ±10,
    adds/removes layers to match the new layer count, then perturbs every
    remaining layer's type, neuron count, and dropout rate.
    """
    mutated = creator.Individual([ind[0].copy(), [layer.copy() for layer in ind[1]]])

    for i, (key, value) in enumerate(control_ranges.items()):
        if key == "fc_layers":
            mutated[0][i] = 1 if mutated[0][i] == 2 else 2
        elif key == "epochs":
            current = mutated[0][i]
            delta = random.choice([-10, 10])
            mutated[0][i] = max(value[0], min(current + delta, value[1]))

    original_layers = len(ind[1])
    new_fc_layers = mutated[0][0]

    if new_fc_layers > original_layers:
        for _ in range(new_fc_layers - original_layers):
            layer_type = random.choice(list(range(*fc_ranges["layer_type"][:2])) + [fc_ranges["layer_type"][1]])
            pair = random.choice(fc_ranges["num_neurons"]())
            dropout = random.randrange(*fc_ranges["dropout"])
            mutated[1].append([layer_type, pair[1], dropout])
    elif new_fc_layers < original_layers:
        mutated[1] = mutated[1][:new_fc_layers]

    for layer in mutated[1]:
        layer[0] = 1 if layer[0] == 2 else 2
        layer[1] = random.choice(fc_ranges["num_neurons"]())[1]
        layer[2] = max(10, min(layer[2] + random.choice([-10, 10]), 50))

    mutated[1] = sorted(mutated[1], key=lambda x: x[1], reverse=True)
    return (mutated,)


def eval_val_accuracy(individual: list) -> tuple[float]:
    """DEAP fitness function: train a ``GoogleNet`` and return (best_val_accuracy,).

    Training history is attached to *individual* as ``individual.history`` for
    later logging.  GPU cache and Python GC are flushed after training.
    """
    model = GoogleNet(individual)
    trainer = TrainerAndEvaluation(model, _train_loader, _val_loader, _test_loader)
    history = trainer.train()

    individual.history = {
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "lr": history["lr"],
    }

    acc = max(history["val_acc"])

    del model, trainer, history
    torch.cuda.empty_cache()
    gc.collect()

    return (acc,)


def get_fitness_key(ind: list) -> tuple:
    """Return the fitness values tuple for use as a DEAP stats key."""
    return ind.fitness.values


def save_best_model(halloffame: tools.HallOfFame, best_model_path: Path) -> None:
    """Persist the hall-of-fame winner's weights and training logs to *best_model_path*.

    Writes ``best_model.pth``, ``best_model_training_log.csv``, and
    ``best_individual_details.csv``.  Skips CSV writing if the individual has
    no attached ``.history``.
    """
    best_model_path.mkdir(parents=True, exist_ok=True)
    best_individual = halloffame[0]

    model = GoogleNet(best_individual)
    torch.save(model.state_dict(), best_model_path / "best_model.pth")

    if not hasattr(best_individual, "history"):
        return

    history = best_individual.history
    pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_accuracy": history["train_acc"],
        "val_accuracy": history["val_acc"],
    }).to_csv(best_model_path / "best_model_training_log.csv", index=False)

    pd.DataFrame({
        "control_genes": [best_individual[0]],
        "fc_genes": [best_individual[1]],
    }).to_csv(best_model_path / "best_individual_details.csv", index=False)


def _register_toolbox() -> None:
    """Register DEAP operators (individual, population, evaluate, mate, mutate, select)."""
    toolbox.register("individual", tools.initIterate, creator.Individual, individual_generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_val_accuracy)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=5)


_register_toolbox()


def run_evolution(
    population_size: int = GA_POPULATION_SIZE,
    ngen: int = GA_NGEN,
    best_model_path: Path = BEST_MODEL_DIR,
    genetic_results_path: Path = GENETIC_RESULTS_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    cxpb: float = GA_CXPB,
    mupb: float = GA_MUPB,
) -> tuple:
    """Run the GA for *ngen* generations and return ``(population, logbook, halloffame)``.

    Resumes from *checkpoint_path* if it exists.  After each generation the
    checkpoint is overwritten and a row is appended to ``GA_Results.csv`` in
    *genetic_results_path*.  The best individual's model is saved via
    ``save_best_model()`` on completion.

    Args:
        population_size: Number of individuals in the population.
        ngen: Total number of generations to run.
        best_model_path: Directory for the winning model artefacts.
        genetic_results_path: Directory for ``GA_Results.csv``.
        checkpoint_path: Pickle file used for mid-run checkpointing.
        cxpb: Initial crossover probability (decays by 50 % over *ngen*).
        mupb: Initial mutation probability (grows by 50 % over *ngen*).
    """
    genetic_results_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            cp = pickle.load(f)
        pop = cp["population"]
        logbook = cp["logbook"]
        hof = cp["halloffame"]
        gen = cp["generation"]
        stats = cp.get("stats", tools.Statistics(key=get_fitness_key))
    else:
        pop = toolbox.population(n=population_size)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "Best Fitness"]
        hof = tools.HallOfFame(1)
        gen = 0
        stats = tools.Statistics(key=get_fitness_key)
        stats.register("Best Fitness", np.max)

    while gen < ngen:
        gen += 1
        print(f"\n─── GENERATION {gen}/{ngen} ───")

        progress = gen / ngen
        current_cxpb = cxpb * (1 - 0.5 * progress)
        current_mupb = mupb * (1 + 0.5 * progress)

        elites = tools.selBest(pop, int(0.1 * population_size))
        offspring = tools.selTournament(pop, len(pop) - len(elites), tournsize=3)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < current_cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < current_mupb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop = elites + offspring

        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        print(logbook.stream)

        csv_file = genetic_results_path / "GA_Results.csv"
        current_row = logbook[-1]
        file_exists = csv_file.is_file()
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=current_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(current_row)

        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {"population": pop, "logbook": logbook, "halloffame": hof,
                 "generation": gen, "stats": stats},
                f,
            )

        del invalid, fitnesses, offspring
        torch.cuda.empty_cache()
        gc.collect()

    save_best_model(hof, best_model_path)
    return pop, logbook, hof
