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

| Puzzles | Kind |
|---|---|
| `puzzle_01` to `puzzle_06` | Numerical (beginner) |
| `puzzle_07` to `puzzle_10` | Numerical (physics) |
| `puzzle_11` to `puzzle_20` | Image |

## Commands

Show a puzzle and its sample data (25 rows for beginner puzzles, 100 for physics):

```bash
uv run launch.py show puzzle_07
```

Create its plots. Physics puzzles also get a 3D plot that opens in VS Code; the viewer extension installs itself the first time:

```bash
uv run launch.py show puzzle_07 --plot
```

For image puzzles, `show` writes `input.png` and `output.png`. Try your own transforms on the input picture; they are applied left to right:

```bash
uv run launch.py show puzzle_18
uv run launch.py apply puzzle_18 --apply invert rotate_chunks
```

Try transforms on any picture of your own (`.jpg`, `.jpeg` or `.png` only). It is cropped to a square and resized to 256×256 first, and results go to `outputs/custom/`:

```bash
uv run launch.py apply --image my_photo.jpg --apply invert vignette
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
