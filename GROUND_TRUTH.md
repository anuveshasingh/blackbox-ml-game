# Ground Truth Functions

This file is the answer key for the 25 public puzzle IDs. The public IDs are assigned in catalogue order and intentionally do not reveal the puzzle names.

Every generated output is the deterministic function plus reproducible uniform formula noise $\epsilon \sim U(-1.0, 1.0)$. For puzzles with configured noise, `eta` is also independent Gaussian noise with the stated standard deviation. The deterministic function is shown first.

## Beginner

| Puzzle | Ground-truth function | Notes |
|---|---|---|
| `puzzle_01` | $y = 3x + 15$ | Linear regression |
| `puzzle_02` | $y = -2x + 20$ | Linear regression |
| `puzzle_03` | $y = x^2$ | Use the feature $x^2$ |
| `puzzle_04` | $y = 2\sqrt{x}$ | Use the feature $\sqrt{x}$ |
| `puzzle_05` | $y = 3\ln(x) + 1$ | Natural logarithm |
| `puzzle_06` | $y = 3x_1$ | $x_2$, $x_3$, and $x_4$ are distractors |

## Intermediate

| Puzzle | Ground-truth function | Notes |
|---|---|---|
| `puzzle_07` | $y = x + 0.5\sin(x)$ | Use both $x$ and $\sin(x)$ |
| `puzzle_08` | $y = 2x + \sin(x) + \eta$, $\eta \sim N(0, 0.1^2)$ | Trend plus periodic noise |
| `puzzle_09` | $y = 10/x$ | Use $1/x$ |
| `puzzle_10` | $y = 2|x|$ | Absolute-value relationship |
| `puzzle_11` | $y = \cos(x)$ | Cosine wave |
| `puzzle_12` | $y = \sin(x)$ | Period $2\pi$ |
| `puzzle_13` | $y = \sin(2\pi x/7)$ | Period 7 |
| `puzzle_14` | $y = x_1x_2$ | Product interaction |
| `puzzle_15` | $y = x_1/x_2$ | Ratio interaction |

## Challenge

| Puzzle | Ground-truth function | Notes |
|---|---|---|
| `puzzle_16` | $y = 1$ if $x_1^2 + x_2^2 < 16$, otherwise $y = 0$ | Circular classification boundary |
| `puzzle_17` | $y = \sqrt{x_1^2 + x_2^2}$ | Euclidean distance |
| `puzzle_18` | $y = x^3$ | Cubic relationship |
| `puzzle_19` | $y = \begin{cases}2x & x < 3 \\ x + 3 & 3 \le x < 7 \\ 0.5x + 6.5 & x \ge 7\end{cases}$ | Three-piece function |
| `puzzle_20` | $y = x_1x_2 + 2x_3$ | Product plus linear feature |
| `puzzle_21` | $y = \sin(x) + 0.5\sin(3x)$ | Two frequencies |
| `puzzle_22` | $y = \sin(x_1) + x_2^2 + 3x_3$ | Three transformed features |
| `puzzle_23` | $y = \sin(2\pi x/7) + \eta$, $\eta \sim N(0, 0.15^2)$ | `x2` is a distractor |
| `puzzle_24` | $y = x^2 - 3x$ | Requires both $x^2$ and $x$ |
| `puzzle_25` | $y = \sin(x) + \cos(x)$ | Phase-shifted sinusoid |

## Plot Epsilon

Plots add bounded uniform visual jitter to plotted values only:

```text
noise ~ Uniform(-epsilon, +epsilon), epsilon = 1.0
```

The fixed bound is `epsilon = 1.0`, and NumPy uses a fixed random seed of `42`, so repeated generated datasets and plots are reproducible. CSV outputs include formula noise; residual calculations use those generated values, while plot jitter is applied only to the rendered PNG.
