# Evolution Engine Documentation

## Overview

`evolution_engine.py` wires together the major components—population, genetic operators, and evaluator—to run generational improvement. The engine provides a high-level `run()` loop that can be reused across tasks.

## `EvolutionConfig`

```python
EvolutionConfig(
    max_generations: int = 100,
    mutation_threshold: float = 0.1,
    constant_mutation_rate: float = 0.0,
    verbose: bool = True,
)
```

- `max_generations`: total evolutionary iterations to execute.
- `mutation_threshold`: probability passed to `GeneticOperators.mutate_program`.
- `constant_mutation_rate`: optional probability of invoking `mutate_constants` on offspring memories.
- `verbose`: toggles per-generation summaries.

Validation ensures sensible ranges for the parameters.

## `EvolutionEngine`

Constructor:

```python
EvolutionEngine(
    population: Population,
    operators: GeneticOperators,
    evaluator: FitnessEvaluator,
    config: EvolutionConfig,
    rng: Optional[np.random.Generator] = None,
)
```

### `run()` Flow

1. Evaluate the current population via `population.evaluate_all`.
2. Optionally print statistics and record history.
3. Create offspring:
   - Select parents with tournament selection.
   - Perform crossover (`operators.crossover`).
   - Mutate the child program (`operators.mutate_program`).
   - Optionally mutate memory constants.
4. Apply elitism and replace the population.
5. Repeat until `max_generations` is reached.

The engine returns the final population, with `population.best_ever` tracking the best individual encountered.

## Custom Evaluators

`EvolutionEngine` stays problem-agnostic. Task-specific logic (e.g., reinforcement learning environments) should be implemented in a `FitnessEvaluator` subclass, which loads observations into each individual’s `MemoryBank` and computes rewards.

## Example Usage

```python
engine = EvolutionEngine(population, operators, evaluator, EvolutionConfig(max_generations=50))
final_population = engine.run()
final_population.print_summary()
best = final_population.best_ever
```
