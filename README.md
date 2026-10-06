# Blackbox ML Game

A command-line puzzle game for discovering hidden relationships between input variables and an output `y`. Explore the generated data, try feature transformations, and record your result on the course leaderboard.

## Installation

Install `uv` once. On macOS with Homebrew:

```bash
brew install uv
```

On Windows or Linux, follow the installer at <https://docs.astral.sh/uv/getting-started/installation/>.

Clone the repository and enter it:

```bash
git clone https://github.com/anuveshasingh/blackbox-ml-game.git
cd blackbox-ml-game
```

This single command creates the environment and installs all dependencies automatically:

```bash
uv run python play.py list
```

No virtual-environment activation is needed. Later commands use the same `uv run python play.py ...` prefix.

## Puzzle Groups

Puzzles are identified publicly by number so their names do not reveal the underlying function.

| Puzzles | Kind |
|---|---|
| `puzzle_01` to `puzzle_06` | Numerical (beginner) |
| `puzzle_07` to `puzzle_10` | Numerical (physics) |
| `puzzle_11` to `puzzle_16` | Image |

List the catalogue at any time:

```bash
uv run python play.py list
```

## Playing A Puzzle

Start with a puzzle description and a 100-row sample. For numerical puzzles:

```bash
uv run python play.py show puzzle_07
```

This prints the puzzle description, input columns, sample rows, and the transformations available for that puzzle.

Create the plots for the generated sample:

```bash
uv run python play.py show puzzle_07 --plot
```

This writes one PNG per input (`y` against that input) and, for physics puzzles with two or more inputs, one 3D PLY file per pair of inputs:

```text
outputs/puzzle_07/plots/y_vs_theta.png
outputs/puzzle_07/plots/y_vs_t.png
outputs/puzzle_07/plots/3d/puzzle_07_y_vs_theta_t.ply
```

The 3D files open in VS Code. The first time you run this, the game installs the PLY viewer extension (`kleinicke.ply-visualizer`) for you. Drag with the mouse to rotate. If the `code` command is not on your PATH, the game tells you how to fix that.

List all transformation names and binary operations:

```bash
uv run python play.py transforms
```

A feature such as `square:x1` means that the model receives $x1^2$ rather than the raw `x1`. A binary feature such as `{"binary": "multiply", "a": "x1", "b": "x2"}` means $x1 \times x2$.

## Custom Input Points

Use `points` when you want to evaluate the hidden function at your own input rows:

```bash
uv run python play.py points puzzle_10 --input points.txt
```

The CSV is written under:

```text
outputs/puzzle_10/<input-name>_puzzle_10_output.csv
```

Add `--plot` to create a PNG beside the CSV:

```bash
uv run python play.py points puzzle_10 --input points.txt --plot
```

### Input File Format

Use one point per line. Values must appear in the exact order shown by `show`.
Whitespace and commas are both accepted. A header on the first line is optional, and comment lines beginning with `#` are ignored.

For a one-variable puzzle:

```text
# x values
0
1.5
3
```

For a two-variable puzzle:

```text
x1,x2
0,1
2,3
-1,4
```

The two-variable example may also be written with spaces:

```text
0 1
2 3
-1 4
```

Do not add a `y` column. The program calculates `y`. Do not add transformed columns such as `x_squared`; provide the raw input columns and use the puzzle's feature transformations when analyzing the result.

## Residuals

Use `residuals` to fit a model to your supplied points and export the observed output, model prediction, and residual:

```bash
uv run python play.py residuals puzzle_07 \
  --input points.txt \
  --features identity:t sin:theta \
  --model linear_regression
```

The residual CSV is written to:

```text
outputs/puzzle_07/<input-name>_puzzle_07_residuals.csv
```

The residual is:

```text
residual = y - prediction
```

Add `--plot` to create `..._residuals.png` beside the CSV:

```bash
uv run python play.py residuals puzzle_07 \
  --input points.txt \
  --features identity:t sin:theta \
  --model linear_regression \
  --plot
```

## Noise

Numerical puzzles are exact: their outputs are the true function values, with no formula noise. A puzzle can declare its own Gaussian noise, but no current puzzle does. The `points` and `residuals` plots add visual jitter bounded by `1.0` so overlapping points stay visible. The jitter is not in the CSV files. Random jitter is not a meaningful feature: do not try to create a feature for it.

## Leaderboard

This repository is the game engine and exploration tool. Do not create or send answer JSON files through the repository commands. After exploring a puzzle, use the separate course leaderboard to record your puzzle ID, chosen model, selected features, and the requested metric or result. Follow the leaderboard instructions for the submission link and required format.

## Common Commands

```bash
uv run python play.py list
uv run python play.py show puzzle_07
uv run python play.py show puzzle_07 --plot
uv run python play.py transforms
uv run python play.py points puzzle_10 --input points.txt --plot
uv run python play.py residuals puzzle_07 --input points.txt --features identity:t sin:theta --plot
```
