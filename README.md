# Blackbox ML Game

A command-line puzzle game about discovering hidden relationships between input values and an output `y`.

## Start

You need `git` and `uv`. Python is not needed: `uv` fetches one by itself if the computer has none.

Install `uv` once.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(or `brew install uv` with Homebrew)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then open a new terminal so `uv` is found.

Clone the repo:

```bash
git clone https://github.com/anuveshasingh/blackbox-ml-game.git
cd blackbox-ml-game
```

Start the game. This is the only game setup command:

```bash
uv run launch.py list
```

The first run downloads the game for your computer (about 2 MB) and caches it; later runs start instantly.

## Puzzle Groups

| Puzzles | Kind |
|---|---|
| `puzzle_01` to `puzzle_06` | Numerical (beginner) |
| `puzzle_07` to `puzzle_10` | Numerical (physics) |
| `puzzle_11` to `puzzle_17` | Image |

## Commands

Show a puzzle and its sample data (25 rows for beginner puzzles, 100 for physics):

```bash
uv run launch.py show puzzle_07
```

Create its plots. Physics puzzles also get a 3D plot that opens in VS Code; the viewer extension installs itself the first time:

```bash
uv run launch.py show puzzle_07 --plot
```

For image puzzles, `show` writes each input picture with its output (`input.png`/`output.png`, or `input_1.png`/`output_1.png` and `input_2.png`/`output_2.png` when a puzzle shows two examples):

```bash
uv run launch.py show puzzle_11
```

To test a guess, use `apply` on one picture at a time: give it with `--input` (`.jpg`, `.jpeg` or `.png` only) and the transform names after `--apply`, applied left to right. The picture is cropped to a square and resized to 256×256 first, and the result goes to `outputs/apply/`. It can be a puzzle's input or any picture of your own:

```bash
uv run launch.py apply --input outputs/puzzle_11/input_1.png --apply rotate_chunks
uv run launch.py apply --input my_photo.jpg --apply invert vignette
```

Everything is saved under:

```text
outputs/puzzle_XX/
```

List feature transformations and image transform names:

```bash
uv run launch.py transforms
```

Evaluate custom input points:

```bash
uv run launch.py points puzzle_07 --input points.txt --plot
```

Fit a model and inspect residuals:

```bash
uv run launch.py residuals puzzle_07 \
  --input points.txt \
  --features square:t '{"product": ["t", "sin:theta"]}' \
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

Features in `--features` are a column (`t`), a transform of a column (`square:t`), or JSON in single quotes for combinations: `'{"product": ["t", "sin:theta"]}'`, `'{"sum": ["x", {"term": "t", "sign": -1}], "transform": "sin"}'`, `'{"binary": "multiply", "a": "x1", "b": "x2"}'`. Run `transforms` for every name.

## Exact values

Every value and every plotted point is the true function value. There is no noise and no plot jitter. Inputs are sampled with seed `42`, so every run gives the same data.

## Leaderboard

Use this game to explore puzzles and collect the requested result. Record your puzzle ID, model, features, and metric on the leaderboard. This repository does not require player answer files.
