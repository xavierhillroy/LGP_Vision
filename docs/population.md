# Population Documentation

## Overview

`population.py` manages cohorts of `Individual` instances during evolution. It tracks statistics, handles selection, applies elitism, and orchestrates generational replacement.

## `PopulationConfig`

```python
PopulationConfig(
    size: int,
    program_length: Tuple[int, int],
    elitism: int = 1,
)
```

- `size`: number of individuals in the population.
- `program_length`: inclusive range for randomly generated program lengths (`(min_len, max_len)`).
- `elitism`: count of top individuals preserved each generation.

Validation ensures positive sizes and that `elitism < size`.

## `Population`

Constructor:

```python
Population(
    config: PopulationConfig,
    instruction_set: InstructionSet,
    memory_config: MemoryConfig,
    operators: Optional[GeneticOperators] = None,
    rng: Optional[np.random.Generator] = None,
)
```

- Uses `Individual.random(...)` to initialize random individuals (`initialize_random`).
- Stores optional references to `GeneticOperators` for convenience but remains agnostic to the variation strategy.

### Core Methods

- `initialize_random(mutate_constants=True)`: rebuilds the population with freshly sampled individuals.
- `tournament_selection(tournament_size, num_winners)`: returns references to winners based on fitness.
- `select_best(n)`: fetches the top `n` individuals.
- `apply_elitism(offspring)`: merges elites from the current population into a new offspring list.
- `replace_population(new_individuals)`: swaps in the new generation and updates best-ever tracking.

### Statistics & Utilities

- `compute_statistics()` → `(min, mean, max, std)` of fitness values.
- `record_statistics()` logs min/mean/max to `fitness_history`.
- `get_diversity_metrics()` currently returns mean and std of program lengths (extensible for other measures).
- Rich iteration helpers (`__len__`, `__iter__`, `__getitem__`) make working with populations Pythonic.

### Evaluation

- `evaluate_all(evaluator, verbose=False)` delegates to each individual’s `evaluate(...)` method.
- `invalidate_all_fitness()` clears cached results prior to a new evaluation round.

## Usage Example

```python
config = PopulationConfig(size=50, program_length=(6, 12), elitism=2)
population = Population(config, instruction_set, memory_cfg, operators, rng=rng)
population.initialize_random()
winners = population.tournament_selection(tournament_size=5, num_winners=2)
```
