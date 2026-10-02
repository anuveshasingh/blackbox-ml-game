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

| Difficulty | Puzzles |
|---|---|
| Beginner | `puzzle_01` through `puzzle_06` |
| Intermediate | `puzzle_07` through `puzzle_15` |
| Challenge | `puzzle_16` through `puzzle_25` |

List the catalogue at any time:

```bash
uv run python play.py list
uv run python play.py list --difficulty 1
uv run python play.py list --difficulty 2
uv run python play.py list --difficulty 3
```

## Playing A Puzzle

Start with a puzzle description and a 100-row sample:

```bash
uv run python play.py show puzzle_08
```

This prints the puzzle description, input columns, sample rows, and the transformations available for that puzzle.

Create the base plot for the generated sample:

```bash
uv run python play.py show puzzle_08 --plot
```

The PNG is written to:

```text
outputs/puzzle_08/puzzle_08_base.png
```

List all transformation names and binary operations:

```bash
uv run python play.py transforms
```

A feature such as `square:x1` means that the model receives $x1^2$ rather than the raw `x1`. A binary feature such as `{"binary": "multiply", "a": "x1", "b": "x2"}` means $x1 \times x2$.

## Custom Input Points

Use `points` when you want to evaluate the hidden function at your own input rows:

```bash
uv run python play.py points puzzle_17 --input points.txt
```

The CSV is written under:

```text
outputs/puzzle_17/<input-name>_puzzle_17_output.csv
```

Add `--plot` to create a PNG beside the CSV:

```bash
uv run python play.py points puzzle_17 --input points.txt --plot
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
uv run python play.py residuals puzzle_08 \
  --input points.txt \
  --features identity:x sin:x \
  --model linear_regression
```

The residual CSV is written to:

```text
outputs/puzzle_08/<input-name>_puzzle_08_residuals.csv
```

The residual is:

```text
residual = y - prediction
```

Add `--plot` to create `..._residuals.png` beside the CSV:

```bash
uv run python play.py residuals puzzle_08 \
  --input points.txt \
  --features identity:x sin:x \
  --model linear_regression \
  --plot
```

Use `--no-noise` with `points` or `residuals` when you need the deterministic function rather than the generated noisy output:

```bash
uv run python play.py points puzzle_08 --input points.txt --no-noise
```

## Noise

Generated outputs include reproducible NumPy noise with seed `42`. Every puzzle receives uniform formula noise bounded by:

```text
Uniform(-1.0, +1.0)
```

Some puzzles also have their own configured Gaussian noise. Plot files add visual jitter bounded by `1.0` on top of the generated values. Random noise is not a meaningful feature: do not try to create a feature for it or treat it as part of the hidden equation.

## Leaderboard

This repository is the game engine and exploration tool. Do not create or send answer JSON files through the repository commands. After exploring a puzzle, use the separate course leaderboard to record your puzzle ID, chosen model, selected features, and the requested metric or result. Follow the leaderboard instructions for the submission link and required format.

## Common Commands

```bash
uv run python play.py list
uv run python play.py show puzzle_08
uv run python play.py show puzzle_08 --plot
uv run python play.py transforms
uv run python play.py points puzzle_17 --input points.txt --plot
uv run python play.py residuals puzzle_08 --input points.txt --features identity:x sin:x --plot
```
