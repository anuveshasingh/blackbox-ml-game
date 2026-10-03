# Blackbox ML Game

A command-line puzzle game about discovering hidden relationships between input values and an output `y`.

## Start

Install `uv` once.

macOS with Homebrew:

```bash
brew install uv
```

Windows and Linux: install `uv` from <https://docs.astral.sh/uv/getting-started/installation/>.

Clone the repo:

```bash
git clone https://github.com/anuveshasingh/blackbox-ml-game.git
cd blackbox-ml-game
```

Start the game. This is the only game setup command:

```bash
uv run launch.py list
```

The first run downloads the correct game executable for the computer and caches it. Players do not need Python packages or a virtual environment beyond `uv` itself.

## Puzzle Groups

| Difficulty | Puzzles |
|---|---|
| Beginner | `puzzle_01` through `puzzle_06` |
| Intermediate | `puzzle_07` through `puzzle_15` |
| Challenge | `puzzle_16` through `puzzle_25` |

## Commands

Show a puzzle and its sample data:

```bash
uv run launch.py show puzzle_08
```

Create its base plot:

```bash
uv run launch.py show puzzle_08 --plot
```

Plots and generated CSVs are saved under:

```text
outputs/puzzle_08/
```

List transformations:

```bash
uv run launch.py transforms
```

Evaluate custom input points:

```bash
uv run launch.py points puzzle_17 --input points.txt --plot
```

Fit a model and inspect residuals:

```bash
uv run launch.py residuals puzzle_08 \
  --input points.txt \
  --features identity:x sin:x \
  --model linear_regression \
  --plot
```

`points` writes `<input-name>_puzzle_XX_output.csv`. `residuals` writes `<input-name>_puzzle_XX_residuals.csv`. Each file and its optional PNG are placed in `outputs/puzzle_XX/`.

## Input Files

Put one raw input point on each line, with columns in the exact order shown by `show`. Spaces and commas are accepted. A header is optional. Lines beginning with `#` are ignored.

One input column:

```text
# x
0
1.5
3
```

Two input columns:

```text
x1,x2
0,1
2,3
-1,4
```

The two-column version may also use spaces:

```text
0 1
2 3
-1 4
```

Do not include a `y` column or transformed columns. The game calculates `y`; transformation names belong in the `--features` argument.

## Noise

The generated equations include reproducible random noise using seed `42`. The formula noise is uniformly bounded by `1.0`, and plots add visual jitter bounded by `1.0`. Random noise is not a feature and should not be treated as part of an equation.

## Leaderboard

Use this game to explore puzzles and collect the requested result. Record your puzzle ID, model, features, and metric on the leaderboard. This repository does not require player answer files.
