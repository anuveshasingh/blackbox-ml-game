# Ground Truth and Design Decisions

This file is the answer key and design record for the Blackbox ML Game. The code on the `preesha` branch is the ground truth; this file describes it. If they ever disagree, the code wins and this file should be updated.

## Catalogue at a glance

There are **21 puzzles**, numbered `puzzle_01` to `puzzle_21` with no gaps. Numbers follow catalogue order. Public IDs never reveal the puzzle's name, title, function, or tier.

| Puzzles | Round | Kind |
|---|---|---|
| `puzzle_01` – `puzzle_06` | 1 — Beginner | Numerical, one obvious feature |
| `puzzle_07` – `puzzle_10` | 2 — Physics | Numerical, two inputs, fixed constants |
| `puzzle_11` – `puzzle_16` | 3 — Image, single transform | Curated picture + one transform |
| `puzzle_17` – `puzzle_21` | 3 — Image, combinations | Curated picture + several transforms |

## Text shown to players

Puzzle IDs, titles and tiers are never shown. The only text is each puzzle's `description`:

- **Beginner:** none. The data is the whole puzzle.
- **Physics:** the setup and the fixed values, worded like an exam problem. It must state every fixed value and nothing about the answer.
- **Image:** the same neutral line for every image puzzle — "Work out what was done to input.png to make output.png." It must not hint at the transform, the number of steps, or whether order matters.

## Code layout — where to make changes

The engine is done; later work should only touch puzzle definitions and images.

| To change | Edit |
|---|---|
| Which puzzles exist, their order (= public numbers), descriptions, ranges, fixed values | `src/blackbox_game/puzzles.py` (definitions + `PUZZLE_REGISTRY`) |
| A physics formula or constant | `src/blackbox_game/physics.py` (`FORMULAS`) |
| A beginner function type | `src/blackbox_game/generator.py` (`_SAMPLERS`, `_hidden_function`) |
| Feature transforms players can use | `src/blackbox_game/transforms.py` |
| An image transform or its parameters | `src/blackbox_game/images.py` (`IMAGE_TRANSFORMS` and the constants at the top) |
| A combination puzzle's steps or picture | `src/blackbox_game/images.py` (`PIPELINES`) |
| The curated pictures | `src/blackbox_game/curated/*.png`, 256×256 RGB |
| Sample sizes | `src/blackbox_game/generator.py` (`SAMPLE_COUNTS`) |

Then update this file to match.

## Player commands

```bash
uv run python play.py list                          # IDs only, no titles or tiers
uv run python play.py show puzzle_07 [--plot]       # description + data, or two pictures
uv run python play.py apply puzzle_17 --apply invert rotate solarise
uv run python play.py transforms                    # every feature and image transform name
uv run python play.py points puzzle_07 --input points.txt [--plot]
uv run python play.py residuals puzzle_07 --input points.txt --features square:t '{"product": ["t", "sin:theta"]}' [--plot]
```

- `show` prints `ID: puzzle_NN`, the puzzle description (if any), the input columns and the sample rows. It does not print a title, a difficulty tier, or a list of models.
- There is **no submission or scoring code**. Players record results on the separate leaderboard.
- `apply` works on image puzzles only. `points` and `residuals` work on numerical puzzles only.

## Output folder

Everything is written under `outputs/` (next to `play.py`, or in the current directory when run from the packaged binary).

```text
outputs/
├── puzzle_01/
│   └── plots/
│       └── y_vs_x.png
├── puzzle_07/
│   ├── plots/
│   │   ├── y_vs_theta.png
│   │   ├── y_vs_t.png
│   │   └── 3d/
│   │       └── puzzle_07_y_vs_theta_t.ply
│   ├── <input-name>_puzzle_07_output.csv          (from points)
│   └── <input-name>_puzzle_07_residuals.csv       (from residuals)
└── puzzle_17/
    ├── input.png
    ├── output.png
    └── input_invert_rotate_solarise.png           (from apply)
```

- `show` does not write a CSV; the data is printed in the terminal. Plots are written only with `--plot`.
- `points --plot` and `residuals --plot` write a PNG beside their CSV.
- `apply` names its result `input_<transform names joined by _>.png`.

## Numerical data and plots

- **Exact values.** There is no noise of any kind and no plot jitter. Every printed value and every plotted point is the true function value.
- **Sample size.** Beginner puzzles show **25** rows. Physics puzzles show **100** rows, because their 3D plots need more points to show a surface.
- **Seed.** Inputs are sampled uniformly from each puzzle's ranges with NumPy seed `42`, so every run gives identical data.
- **Univariate plots.** With `--plot`, every numerical puzzle gets one scatter of `y` against each input: `plots/y_vs_<input>.png`.
- **3D plots.** Physics puzzles also get one 3D plot per pair of inputs (all four have exactly two inputs, so one plot each): `plots/3d/puzzle_NN_y_vs_<a>_<b>.ply`.
  - Format: ASCII PLY point cloud, no 3D library needed.
  - Axes: first input on x (red line), output `y` vertical (green line), second input on z (blue line).
  - Each axis is normalised to [−1, 1] for display; the real ranges are written in the PLY header as comments.
  - Each data point is a small sphere of 120 points, coloured by `y` (viridis).
  - Axis names and end values are drawn as text made of points (PLY has no text).
  - Viewer: the VS Code extension `kleinicke.ply-visualizer`. `show --plot` installs it automatically if it is missing and opens the files; drag to rotate. If the `code` command is not on PATH, the game prints how to fix that.

## Round 1 — Beginner (`puzzle_01` – `puzzle_06`)

| Puzzle | Inputs (range) | Ground-truth function | Solution feature |
|---|---|---|---|
| `puzzle_01` | $x \in [-5, 5]$ | $y = 3x + 15$ | `identity:x` |
| `puzzle_02` | $x \in [-5, 5]$ | $y = -2x + 20$ | `identity:x` |
| `puzzle_03` | $x \in [-5, 5]$ | $y = x^2$ | `square:x` |
| `puzzle_04` | $x \in [0.5, 25]$ | $y = 2\sqrt{x}$ | `sqrt:x` |
| `puzzle_05` | $x \in [1, 100]$ | $y = 3\ln(x) + 1$ | `log:x` |
| `puzzle_06` | $x_1 \in [-5, 5]$; $x_2, x_3, x_4 \in [-5, 5]$ | $y = 3x_1$ | `identity:x1` ($x_2$–$x_4$ are distractors) |

## Round 2 — Physics (`puzzle_07` – `puzzle_10`)

Each puzzle has exactly two input columns. Every other quantity is **fixed**, and the fixed values are stated in the puzzle's description the way an exam problem would state them. Input names keep the standard physics symbols as hints. The solution features listed below are the answer key; each reaches $R^2 = 1$.

| Puzzle | Inputs (range) | Fixed values | Ground-truth function | Solution features |
|---|---|---|---|---|
| `puzzle_07` | $\theta \in [0.1, 1.4]$ rad, $t \in [0.5, 5]$ s | $v_0 = 10$ m/s | $y = v_0\sin(\theta)\,t - \tfrac{1}{2}gt^2$ | $t\sin\theta$, $t^2$ |
| `puzzle_08` | $x \in [0.1, 2]$ m, $v \in [0.5, 5]$ m/s | $k = 4$ N/m, $m = 2$ kg | $y = \tfrac{1}{2}kx^2 + \tfrac{1}{2}mv^2$ | $x^2$, $v^2$ |
| `puzzle_09` | $x \in [0, 5]$ m, $t \in [0, 5]$ s | $A = 5$, $k = 1$ rad/m, $\omega = 1$ rad/s, $\phi = 0$ | $y = A\sin(kx - \omega t + \phi)$ | $\sin(x - t)$ |
| `puzzle_10` | $r_1, r_2 \in [1, 5]$ m | $q_1 = 2\,\mu$C, $q_2 = -3\,\mu$C | $y = k_e\left(\dfrac{q_1}{r_1} + \dfrac{q_2}{r_2}\right)$ | $1/r_1$, $1/r_2$ |

In `puzzle_08`, $k$ is the spring constant. In `puzzle_09`, $k$ is the wavenumber. Coulomb's constant is written $k_e$.

### Physical constants

| Constant | Value |
|---|---|
| $g$ | $9.80665\ \text{m/s}^2$ |
| $k_e$ | $8.9875517923 \times 10^{9}\ \text{N}\,\text{m}^2\,\text{C}^{-2}$ |

### Feature grammar (numerical puzzles)

Features passed to `residuals --features` can be (dict forms are written as JSON in single quotes on the command line):

- a column: `"t"`
- a unary transform of a column: `"sin:theta"`
- a binary transform: `{"binary": "multiply", "a": "x1", "b": "x2"}`
- a product of features: `{"product": ["t", "sin:theta"]}`
- a sum of features, with optional sign: `{"sum": ["x", {"term": "t", "sign": -1}]}`
- any product or sum with a transform on the result: `{"sum": [...], "transform": "sin"}`

Unary transforms: `identity`, `square`, `cube`, `sqrt`, `abs`, `log`, `log2`, `reciprocal`, `sin`, `cos`, `exp`, `exp_neg`.
Binary transforms: `multiply`, `divide`, `add`, `subtract`, `distance`.

`residuals` fits `linear_regression` (default) or a depth-4 `decision_tree` (`src/blackbox_game/fitting.py`). Linear regression scales each feature column before fitting; this does not change the fit, but stops a very large column from swamping the others.

## Round 3 — Image Processing (`puzzle_11` – `puzzle_21`)

No data points or plots. `show` writes `input.png` and `output.png` (both 256×256 RGB) into the puzzle's folder and opens them in VS Code. Players work out what was done to the first picture to get the second, and can test guesses with `apply`.

### Curated images

The source pictures were centre-cropped to a square and resized to 256×256 PNGs, stored in `src/blackbox_game/curated/`. The game reads only these PNGs.

| Name | Source file |
|---|---|
| `chessboard` | `black-white-checkered-chessboard-pattern-background_1017-60365.jpg.avif` |
| `doctor_strange` | `doctor-strange91.jpg.webp` |
| `istockphoto` | `istockphoto-1125768166-612x612.jpg` |
| `matrix` | `matrix.avif` |
| `pexels` | `pexels-photo-27966277.avif` |

### All image transforms

These 15 names are accepted by `apply`. Each takes and returns a 256×256 RGB image, so any sequence is valid. All work on a fixed canvas, pivot about the image centre, and are deterministic.

| Name | What it does | Parameters |
|---|---|---|
| `solarise` | Inverts every channel value at or above the threshold; darker values are unchanged | threshold 128 |
| `posterise` | Reduces each channel to 4 levels: 0, 85, 170, 255 | step 64 |
| `repeated_overlay` | Blends the image with copies of itself moved down and right by 4, 8 and 12 px, each step at 50% opacity. No wrap-around: where a copy does not reach, the image underneath stays | 3 overlays, 4 px offset, opacity 0.5 |
| `channel_shuffle` | $(R, G, B) \to (B, R, G)$ | — |
| `swap_rgb_bgr` | $(R, G, B) \to (B, G, R)$ | — |
| `invert` | $255 - \text{value}$ on every channel | — |
| `ghost_echo` | Blends the image with a copy moved 16 px right. No wrap-around: the leftmost 16 columns keep the original image | shift 16 px, weights 0.5137 image / 0.4863 echo |
| `flip_horizontal` | Mirrors left–right | — |
| `stretch_horizontal` | Horizontal magnification about the centre, nearest-neighbour (pixels sampled from outside the image would be black) | factor 1.6 |
| `rotate` | Clockwise rotation about the centre, nearest-neighbour, corners filled black | 15° |
| `gaussian_blur` | Separable 5-tap Gaussian, wrapping at the edges | σ = 1 |
| `circular_shift` | Rolls the image 32 px right, wrapping around | (0, 32) |
| `vignette` | Darkens towards the corners: factor $1 - 0.6\,(r/\sqrt{2})^2$, $r$ = normalised distance from centre | strength 0.6 |
| `fft_encode` | 2D Fourier transform of the grey image, zero frequency centred. Red = $\log(1 + \lvert F\rvert)$ scaled by 255/17; green = phase mapped from $[-\pi, \pi)$ to [0, 255]; blue = 0 | log cap 17 |
| `fft_decode` | Inverse of `fft_encode`: rebuilds the grey image (as RGB) from an encoded spectrum | — |

`channel_shuffle` and `swap_rgb_bgr` are different swaps: the first rotates the channels, the second exchanges red and blue.

### Single-transform puzzles

| Puzzle | Image | Transform | Notes |
|---|---|---|---|
| `puzzle_11` | `doctor_strange` | `solarise` | |
| `puzzle_12` | `istockphoto` | `posterise` | |
| `puzzle_13` | `matrix` | `repeated_overlay` | |
| `puzzle_14` | `pexels` | `channel_shuffle` | RGB → BRG |
| `puzzle_15` | `chessboard` | `fft_encode` | |
| `puzzle_16` | `istockphoto` | `fft_decode` | `input.png` is the encoded spectrum of the image; `output.png` is the decoded grey image. Round-trip PSNR ≈ 44 dB |

### Combination puzzles

Steps are applied left to right as listed. Four pipelines are **commutative**: every ordering of their steps gives a pixel-identical result. The last is **the trap**: order matters.

| Puzzle | Name | Image | Steps | Commutes? |
|---|---|---|---|---|
| `puzzle_17` | Psychedelic ghost | `doctor_strange` | `ghost_echo`, `swap_rgb_bgr`, `invert` | Yes |
| `puzzle_18` | Funhouse pop-art | `doctor_strange` | `stretch_horizontal`, `flip_horizontal`, `posterise` | Yes |
| `puzzle_19` | Tilted acid trip | `matrix` | `rotate`, `solarise`, `swap_rgb_bgr` | Yes |
| `puzzle_20` | Dream negative | `istockphoto` | `gaussian_blur`, `invert`, `circular_shift` | Yes |
| `puzzle_21` | The trap | `matrix` | `ghost_echo`, `vignette` | **No** |

Why the trap does not commute: the vignette darkens towards the corners. Echo then vignette darkens the echo along with everything else. Vignette then echo copies an already-darkened edge 16 px into the picture, so the dark falloff is smeared sideways. The two orders are visibly different.

### Rules that keep the commutative pipelines exact

- **Fixed canvas.** Every transform keeps the 256×256 size and pivots about the centre.
- **Black fill only where it is safe.** `rotate` fills its corners black. Black stays black under `solarise` and under the channel swaps, so `puzzle_19` commutes. `invert` is never paired with `rotate`, because it would turn the black corners white in some orders.
- **No wrap where the image should show through.** `ghost_echo` and `repeated_overlay` paste shifted copies without wrapping; the original image shows where a copy does not reach.
- **Wrap where commuting needs it.** `gaussian_blur` wraps at the edges so it commutes exactly with `circular_shift` in `puzzle_20`.
- **No rounding ties.** An exact 50/50 blend or a binomial blur kernel produces values ending in .5, which round differently depending on order. `ghost_echo` uses weight 0.5137 and `gaussian_blur` uses a σ = 1 Gaussian kernel to avoid this.
- **Checked every time.** Whenever a commutative pipeline is rendered, the game computes every ordering of its steps and asserts the results are identical (`verify_pipeline`). If a new pipeline fails this, nudge a parameter (angle, scale, weight) to break the rounding tie.

## Reproducibility

- **Numerical puzzles:** exact function values, NumPy seed `42`. Every run prints identical data and draws identical plots.
- **Image puzzles:** the curated PNGs and every transform parameter are fixed in code, so `input.png` and `output.png` are byte-identical on every run.

## Packaging

- Runtime dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`. The curated PNGs are declared as package data.
- The player binaries are built from the tip of `preesha` by the workflow on `preesha-binaries`. Its PyInstaller step uses `--collect-data blackbox_game`, which bundles `curated/`. Any new file the game reads at runtime must live inside the `blackbox_game` package and match `package-data`, or the binaries will not include it.
- There is no test suite. Check changes by running `show`, `apply`, `points` and `residuals` on the affected puzzles.
