# Blackbox ML Game

A command-line puzzle game for discovering hidden relationships between input variables and an output `y`. Explore the generated data, try feature transformations, and record your result on the course leaderboard.

## Installation

This branch is the game written in Rust. Install Rust once from <https://rustup.rs>, then:

```bash
git clone -b rust https://github.com/anuveshasingh/blackbox-ml-game.git
cd blackbox-ml-game
cargo build --release
```

The game is one self-contained file, `target/release/blackbox-ml-game` (`.exe` on Windows), with the puzzle pictures and font built in. Nothing else is needed to run it. The examples below use `cargo run --release --`, which builds if needed and then runs it. You can call the binary directly instead.

```bash
cargo run --release -- list
```

Outputs are written under `outputs/` in the current directory.

## Puzzle Groups

Puzzles are identified publicly by number so their names do not reveal the underlying function.

| Puzzles | Kind |
|---|---|
| `puzzle_01` to `puzzle_06` | Numerical (beginner) |
| `puzzle_07` to `puzzle_10` | Numerical (physics) |
| `puzzle_11` to `puzzle_17` | Image |

List the catalogue at any time:

```bash
cargo run --release -- list
```

## Numerical Puzzles

Show a puzzle and its sample data: 25 rows for beginner puzzles, 100 rows for physics puzzles.

```bash
cargo run --release -- show puzzle_07
```

This prints the puzzle ID, the input columns, and the sample rows.

Create the plots for the sample:

```bash
cargo run --release -- show puzzle_07 --plot
```

This writes one PNG per input (`y` against that input) and, for physics puzzles, one 3D PLY file per pair of inputs:

```text
outputs/puzzle_07/plots/y_vs_theta.png
outputs/puzzle_07/plots/y_vs_t.png
outputs/puzzle_07/plots/3d/puzzle_07_y_vs_theta_t.ply
```

The 3D files open in VS Code. The first time you run this, the game installs the PLY viewer extension (`kleinicke.ply-visualizer`) for you. Drag with the mouse to rotate. If the `code` command is not on your PATH, the game tells you how to fix that.

## Image Puzzles

For an image puzzle, `show` writes each input picture with its output and opens them in VS Code. Some puzzles show two examples of the same change:

```bash
cargo run --release -- show puzzle_11
```

```text
outputs/puzzle_11/input_1.png   outputs/puzzle_11/output_1.png
outputs/puzzle_11/input_2.png   outputs/puzzle_11/output_2.png
```

A puzzle with one example writes `input.png` and `output.png`. Work out what was done to each input to make its output. To test a guess, use `apply` on one picture at a time: give the picture with `--input` (`.jpg`, `.jpeg` or `.png` only) and one or more transform names after `--apply`. The transforms are applied left to right. The picture is cropped to a square and resized to 256×256 first, and the result is saved under `outputs/apply/` as `<picture name>_<transform names>.png`.

The picture can be one of the puzzle's inputs or any picture of your own:

```bash
cargo run --release -- apply --input outputs/puzzle_11/input_1.png --apply rotate_chunks
cargo run --release -- apply --input my_photo.jpg --apply invert vignette
```

Run `cargo run --release -- transforms` to see every image transform name.

## Features

List all feature transformations, binary operations, and image transform names:

```bash
cargo run --release -- transforms
```

A feature is one of:

| Feature | Meaning |
|---|---|
| `t` | the raw column |
| `square:t` | a transform of a column, here $t^2$ |
| `'{"binary": "multiply", "a": "x1", "b": "x2"}'` | two columns combined, here $x_1 x_2$ |
| `'{"product": ["t", "sin:theta"]}'` | a product of features, here $t\sin\theta$ |
| `'{"sum": ["x", {"term": "t", "sign": -1}], "transform": "sin"}'` | a sum of features (with signs), optionally transformed, here $\sin(x - t)$ |

Features written with `{...}` are JSON. Put them in single quotes on the command line so the shell passes them through unchanged.

## Custom Input Points

Use `points` to compute `y` for a numerical puzzle at your own input rows:

```bash
cargo run --release -- points puzzle_10 --input points.txt
```

The CSV is written under:

```text
outputs/puzzle_10/<input-name>_puzzle_10_output.csv
```

Add `--plot` to create a PNG beside the CSV:

```bash
cargo run --release -- points puzzle_10 --input points.txt --plot
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
cargo run --release -- residuals puzzle_07 \
  --input points.txt \
  --features square:t '{"product": ["t", "sin:theta"]}' \
  --model linear_regression
```

`--model` is `linear_regression` (default) or `decision_tree`.

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
cargo run --release -- residuals puzzle_07 \
  --input points.txt \
  --features square:t '{"product": ["t", "sin:theta"]}' \
  --model linear_regression \
  --plot
```

## Exact Values

Numerical puzzles are exact: every value in the output and every point on a plot is the true function value. There is no noise and no plot jitter. Inputs are sampled with seed `42`, so every run gives the same data.

## Leaderboard

This repository is the game engine and exploration tool. Do not create or send answer JSON files through the repository commands. After exploring a puzzle, use the separate course leaderboard to record your puzzle ID, chosen model, selected features, and the requested metric or result. Follow the leaderboard instructions for the submission link and required format.

## Common Commands

```bash
cargo run --release -- list
cargo run --release -- show puzzle_07
cargo run --release -- show puzzle_07 --plot
cargo run --release -- show puzzle_11
cargo run --release -- apply --input my_photo.jpg --apply invert vignette
cargo run --release -- transforms
cargo run --release -- points puzzle_10 --input points.txt --plot
cargo run --release -- residuals puzzle_07 --input points.txt --features square:t '{"product": ["t", "sin:theta"]}' --plot
```
