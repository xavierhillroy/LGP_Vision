## Evaluator Documentation

## Example Implementations

### `SymbolicRegressionEvaluator`

- Pulls two scalar observations, executes the program, and penalizes absolute error.
- Useful for sanity tests.

### `CartPoleEvaluator`

- Wraps `gymnasium`'s `CartPole-v1` environment.
- Assumes four scalar observation registers and reads an action from working scalar register `output_register` (default 7).
- Converts the register value into a discrete action (`0` or `1`) and averages episode returns.
- Optional parameters: `episodes`, `max_steps`, `render_mode`.

- Requires `gymnasium`; raises an informative error if the library is missing.

### `FlappyBirdEvaluator`

```

```

### `FitnessEvaluator`

- `episodes`: Number of episodes to average over
- `rng`: Optional random number generator (for reproducibility)
- `output_registers`: Optional `List[Tuple[MemoryType, int]]` describing where the program outputs are read. If provided, the evolution engine removes introns using these registers before evaluation.
