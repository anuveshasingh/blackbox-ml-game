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

## Numerical Puzzles

Show a puzzle and its sample data (25 rows for beginner puzzles, 100 for physics):

```bash
uv run launch.py show puzzle_07
```

Create its plots. Physics puzzles also get a 3D plot that opens in VS Code; the viewer extension installs itself the first time:

```bash
uv run launch.py show puzzle_07 --plot
```

Evaluate your own input points (see Input Files below):

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

`--model` is `linear_regression` (default) or `decision_tree`. `points` writes `<input-name>_puzzle_XX_output.csv` and `residuals` writes `<input-name>_puzzle_XX_residuals.csv`, both in `outputs/puzzle_XX/` with their optional PNG plot beside them.

List every feature transform name:

```bash
uv run launch.py transforms
```

## Image Puzzles

Each image puzzle shows one or two pictures and what they became after a hidden sequence of transforms. Your job is to find that sequence: which transforms, and in what order.

**1. Look at the puzzle.**

```bash
uv run launch.py show puzzle_11
```

This writes the pictures to `outputs/puzzle_11/` and opens them in VS Code (if its `code` command is on PATH):

- one example: `input.png` and `output.png`
- two examples of the same puzzle: `input_1.png`/`output_1.png` and `input_2.png`/`output_2.png`

**2. See the transforms you can use.** The image transform names are at the end of:

```bash
uv run launch.py transforms
```

They are: `rotate_chunks`, `mirror_sum`, `circular_shift`, `swap_rgb_rbg`, `swap_rgb_bgr`, `invert`, `solarise`, `posterise`, `opacity`, `vignette`, `ghost_echo`, `stretch_horizontal`.

**3. Test a guess with `apply`.** Give one picture with `--input` and one or more transform names after `--apply`; they are applied left to right.

```bash
uv run launch.py apply --input outputs/puzzle_11/input_1.png --apply rotate_chunks
uv run launch.py apply --input outputs/puzzle_15/input.png --apply invert vignette
```

The result is written to `outputs/apply/`, named after the picture and the transforms (for example `input_1_rotate_chunks.png`). Compare it with the puzzle's output picture: if they are identical, your guess is right. Order can matter: `invert vignette` and `vignette invert` may give different pictures.

**4. Try your own pictures.** `apply` works on any `.jpg`, `.jpeg` or `.png`, which helps to see what a transform does:

```bash
uv run launch.py apply --input my_photo.jpg --apply mirror_sum
```

Every picture is first centre-cropped to a square and resized to 256×256, like the puzzle pictures; that version is saved too, as `outputs/apply/my_photo.png`.

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
