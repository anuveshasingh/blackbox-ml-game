# Ground Truth and Design Decisions

This file is the answer key and design record for the Blackbox ML Game. The code on the `code` branch is the ground truth; this file describes it. If they ever disagree, the code wins and this file should be updated.

## Catalogue at a glance

There are **17 puzzles**, numbered `puzzle_01` to `puzzle_17` with no gaps. Numbers follow catalogue order. Public IDs never reveal the puzzle's name, title, function, or tier.

| Puzzles | Round | Kind |
|---|---|---|
| `puzzle_01` – `puzzle_06` | 1 — Beginner | Numerical, one input, one or two obvious features |
| `puzzle_07` – `puzzle_10` | 2 — Physics | Numerical, two inputs, fixed constants |
| `puzzle_11` – `puzzle_14` | 3 — Image, single transform | One transform, shown on one or two example pictures |
| `puzzle_15` – `puzzle_17` | 3 — Image, combinations | One picture + several transforms |

## Text shown to players

Puzzle IDs, titles and tiers are never shown, and no puzzle has a description. `show` prints only `ID: puzzle_NN`, the input columns and the data (or, for image puzzles, the paths of the pictures).

## Code layout — where to make changes

The engine is done; later work should only touch puzzle definitions and images.

| To change | Edit |
|---|---|
| Which puzzles exist, their order (= public numbers), ranges, fixed values | `src/puzzles.rs` (definitions in `build()`) |
| A physics formula or constant | `src/generator.rs` (`physics`, constants at the top) |
| A beginner function type | `src/puzzles.rs` (`Function`) and `src/generator.rs` (`hidden_function`) |
| Feature transforms players can use | `src/transforms.rs` |
| An image transform or its parameters | `src/images.rs` (`IMAGE_TRANSFORMS` and the constants at the top) |
| A combination puzzle's steps or picture | `src/images.rs` (`PIPELINES`) |
| Which pictures an image puzzle shows (one, or several examples) | `src/puzzles.rs` (the pictures in `image(...)`) |
| The curated pictures | `assets/curated/*.png` (built into the binary; list new ones in `CURATED` in `src/images.rs`), 256×256 RGB |
| Sample sizes | `src/generator.rs` (`sample_count`) |

Then update this file to match.

## Player commands

```bash
blackbox-ml-game list                          # IDs only, no titles or tiers
blackbox-ml-game show puzzle_07 [--plot]       # data, or the input/output pictures
blackbox-ml-game apply --input photo.jpg --apply invert vignette   # one picture per command
blackbox-ml-game transforms                    # every feature and image transform name
blackbox-ml-game points puzzle_07 --input points.txt [--plot]
blackbox-ml-game residuals puzzle_07 --input points.txt --features square:t '{"product": ["t", "sin:theta"]}' [--plot]
```

- `show` prints `ID: puzzle_NN`, the input columns and the sample rows. It does not print a title, a difficulty tier, or a list of models.
- There is **no submission or scoring code**. Players record results on the separate leaderboard.
- `apply` takes exactly one picture per command, given with `--input`, like `points --input`. It is not tied to a puzzle: the picture can be a puzzle's `input.png` or any picture the player has. Only `.jpg`, `.jpeg` and `.png` are accepted (checked by extension and by the file's real format). The picture is centre-cropped to a square and resized to 256×256 first, so it behaves exactly like a puzzle picture; applying a puzzle's transform to its `input.png` reproduces its `output.png` exactly. `points` and `residuals` work on numerical puzzles only.

## Output folder

Everything is written under `outputs/` (in the current directory).

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
├── puzzle_11/                                     (two example pictures)
│   ├── input_1.png, output_1.png
│   └── input_2.png, output_2.png
├── puzzle_15/                                     (one picture)
│   ├── input.png
│   └── output.png
└── apply/                                         (from apply --input)
    ├── photo.png                                  (the 256×256 version of photo.jpg)
    └── photo_invert_vignette.png
```

- `show` does not write a CSV; the data is printed in the terminal. Plots are written only with `--plot`.
- `points --plot` and `residuals --plot` write a PNG beside their CSV.
- An image puzzle with one picture writes `input.png` and `output.png`; one with two example pictures writes `input_1.png`/`output_1.png` and `input_2.png`/`output_2.png`.
- `apply` writes `<picture name>_<transform names joined by _>.png` under `apply/`, plus the 256×256 version of the picture it worked on.

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
| `puzzle_06` | $x \in [-10, 10]$ | $y = x + 2\sin x$ | `identity:x`, `sin:x` (two features) |

## Round 2 — Physics (`puzzle_07` – `puzzle_10`)

Each puzzle has exactly two input columns. Every other quantity is **fixed** (and not shown to players). Input names keep the standard physics symbols as hints. The solution features listed below are the answer key; each reaches $R^2 = 1$.

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

`residuals` fits `linear_regression` (default) or a depth-4 `decision_tree` (`src/fitting.rs`, matching scikit-learn's `LinearRegression` and `DecisionTreeRegressor`). Linear regression scales each feature column before fitting; this does not change the fit, but stops a very large column from swamping the others.

## Round 3 — Image Processing (`puzzle_11` – `puzzle_17`)

No data points or plots. `show` writes each input/output pair (all 256×256 RGB) into the puzzle's folder and opens them in VS Code. A single-transform puzzle may show **two example pictures of the same transform**, one easier and one harder; they are one puzzle, not two. Players work out what was done to each input to get its output, and can test guesses with `apply`.

### Curated images

Each source picture is converted once to a 256×256 PNG in `assets/curated/`; the game reads only these PNGs. Most are centre-cropped to a square and resized. Two are placed so that `mirror_sum` lines up:

- `moon`: the moon's disc is centred in the square, so its flip lands on itself.
- `molecule` (from `smile.png`): the centre of the drawing's square ring is put exactly on the centre of the picture, so after the flip the ring and the OH groups coincide; only the parts that genuinely differ (Cl, H labels) double up.

| Name | Source file | Used by |
|---|---|---|
| `lsd` | `lsd.jpg` | `puzzle_11` (example 1) |
| `checkmate` | `checkmate.jpeg` (chess endgame) | `puzzle_11` (example 2) |
| `moon` | `moon.jpg` | `puzzle_12` (example 1) |
| `molecule` | `smile.png` (achiral molecule) | `puzzle_12` (example 2) |
| `matrix` | `matrix.png` | `puzzle_13`, `puzzle_15` |
| `marbles` | `marbles.png` (blue and green marbles) | `puzzle_14` (example 1) |
| `monet` | `monet.jpg` (blue Monet painting) | `puzzle_14` (example 2) |
| `doctor_strange` | `doctor_strange.png` | `puzzle_16` |
| `pexels` | `pexels.png` | `puzzle_17` |

### All image transforms

These 12 names are accepted by `apply`. Each takes and returns a 256×256 RGB image, so any sequence is valid. All keep the canvas fixed and are deterministic.

| Name | What it does | Parameters |
|---|---|---|
| `rotate_chunks` | Cuts the image into square tiles and rotates each tile 90° clockwise in place | 64 px tiles (4 × 4 grid) |
| `mirror_sum` | Pixel-wise sum of the image and its flip about the central horizontal line, halved to stay in range | — |
| `circular_shift` | Moves the image 32 px right; what leaves the right edge comes back on the left | (0, 32) |
| `swap_rgb_rbg` | $(R, G, B) \to (R, B, G)$: green and blue exchange | — |
| `swap_rgb_bgr` | $(R, G, B) \to (B, G, R)$: red and blue exchange | — |
| `invert` | $255 - \text{value}$ on every channel | — |
| `solarise` | Inverts every channel value at or above the threshold; darker values are unchanged | threshold 128 |
| `posterise` | Reduces each channel to 4 levels: 0, 85, 170, 255 | step 64 |
| `opacity` | Fades towards white: $0.6 \times \text{image} + 0.4 \times 255$ | 0.6 |
| `vignette` | Darkens towards the corners: factor $1 - 0.6\,(r/\sqrt{2})^2$, $r$ = normalised distance from centre | strength 0.6 |
| `ghost_echo` | Blends the image with a copy moved 16 px right. No wrap-around: the leftmost 16 columns keep the original | shift 16 px, weights 0.5137 image / 0.4863 echo |
| `stretch_horizontal` | Horizontal magnification about the centre, nearest-neighbour | factor 1.6 |

`swap_rgb_rbg` and `swap_rgb_bgr` are different swaps: the first exchanges green and blue, the second red and blue.

### Single-transform puzzles

Each row is one puzzle. Where two pictures are listed, both are examples of the same transform, shown together.

| Puzzle | Transform | Example 1 | Example 2 | Notes |
|---|---|---|---|---|
| `puzzle_11` | `rotate_chunks` | `lsd` | `checkmate` | The busy pattern hides the tiles; the chess endgame shows them through rotated pieces and a broken board |
| `puzzle_12` | `mirror_sum` | `moon` | `molecule` | The achiral molecule's flipped copy coincides with the original |
| `puzzle_13` | `circular_shift` | `matrix` | — | |
| `puzzle_14` | `swap_rgb_rbg` | `marbles` | `monet` | Hint picture: blue and green marbles trade colours. Tough picture: a mostly blue painting turns green |

### Combination puzzles

Steps are applied left to right as listed. "Commutes" means every ordering of the steps gives a pixel-identical picture.

| Puzzle | Image | Steps | Commutes? |
|---|---|---|---|
| `puzzle_15` | `matrix` | `ghost_echo`, `solarise` | No |
| `puzzle_16` | `doctor_strange` | `stretch_horizontal`, `posterise`, `swap_rgb_bgr` | Yes |
| `puzzle_17` | `pexels` | `invert`, `circular_shift`, `rotate_chunks` | No |

Why they behave this way:

- **15:** solarise is not linear, so solarising then blending gives different values from blending then solarising.
- **16:** the stretch only moves pixels, and posterise and the swap act on each pixel's own values, so the order never matters.
- **17:** invert commutes with both, but the shift (32 px) is not a whole number of tiles (64 px), so shifting before or after rotating the tiles moves different pixels.

### Rules for image transforms

- **Fixed canvas.** Every transform keeps the 256×256 size.
- **No rounding ties in blends.** An exact 50/50 blend gives values ending in .5 that round differently in different orders; `ghost_echo` uses 0.5137 to avoid this.
- **Edges.** `circular_shift` wraps around. `ghost_echo` does not: the original shows where the echo does not reach. `stretch_horizontal` only magnifies, so it never samples outside the image.

## Reproducibility

- **Numerical puzzles:** exact function values, seed `42` with NumPy's generator (PCG64 + SeedSequence, ported in `src/rng.rs`, so the rows are the same as the Python game's). Every run prints identical data and draws identical plots.
- **Image puzzles:** the curated PNGs and every transform parameter are fixed in code, so `input.png` and `output.png` are byte-identical on every run.
- **Fourier puzzles were removed.** There are no Fourier-transform puzzles or transforms.

## Packaging

- The game is a single Rust binary (`cargo build --release`, about 2.3 MB). The curated PNGs and the plot font (`assets/DejaVuSans-subset.ttf`, matplotlib's DejaVu Sans cut down to the characters the plots use) are compiled into it with `include_bytes!`, so nothing else ships with it. Any new file the game reads must be added the same way.
- Dependencies: `clap` (command line), `serde_json` (feature specs), `image` (PNG), `mozjpeg` (libjpeg-turbo, the JPEG decoder Pillow uses, so player photos give the same pixels), `ab_glyph` (font outlines). The JPEG decoder is C, built with the `cc` crate: building needs a C compiler but not cmake or nasm.
- Output matches the Python game (the `code` branch): `show` text, generated data, `points` CSVs, every image puzzle and `apply` result, and the 3D PLY files are byte-identical. `residuals` predictions agree to floating-point rounding (about 1e-16 relative). PNG plots copy matplotlib's default style (figure size, DPI, font, tick placement), but they are drawn by `src/plots.rs`, so they are not pixel-identical to matplotlib's.
- Players get the game from the `binaries` branch (and `main`, its copy): `launch.py` downloads the binary for their computer from the GitHub release named in its `RELEASE_TAG` and caches it under `~/.cache/blackbox-ml-game/<tag>/`. That branch's workflow builds this `rust` branch when a `v*` tag is pushed: musl static on Linux, static C runtime on Windows, macOS 11+, one zip per platform holding just the executable named after the asset.

### Releasing a new version

1. Commit and push the change on `rust`.
2. On `binaries`, set `RELEASE_TAG` in `launch.py` to the new version, commit, push.
3. Tag that commit and push the tag (`git tag vX.Y.Z && git push origin vX.Y.Z`). The workflow builds the tip of `rust`, smoke-tests each binary and publishes the release.
4. Check the release has all four zips; `launch.py` fails instead of falling back if they are missing.
5. Fast-forward `main`: `git push origin binaries:main`. A plain `git clone` gets `main`, so skipping this leaves players on the old release.

Asset names in `launch.py`'s `_ASSET_NAMES` must match the workflow's `asset:` values exactly; only `launch.py` adds `.exe`.
- Player pictures for `apply --input` must be `.jpg`, `.jpeg` or `.png` (checked by extension and by the file's real format). All source pictures are kept as JPEG or PNG.
- `cargo test` covers number formatting only. Check changes by running `show`, `apply`, `points` and `residuals` on the affected puzzles.
