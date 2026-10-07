//! Reproducible datasets for the numerical puzzles, and the exact physics formulas.
//!
//! Outputs are exact function values: there is no noise of any kind. Generated
//! data and player points both go through `hidden_function`, so they can never disagree.

use crate::puzzles::{Formula, Function, Puzzle};
use crate::rng::Rng;

/// Standard values, fixed and never input columns.
pub const G_ACCEL: f64 = 9.80665; // m/s^2 — gravitational acceleration (g)
pub const K_COULOMB: f64 = 8.9875517923e9; // N m^2 C^-2 — Coulomb's constant (k_e)

/// Seed for every generated dataset.
pub const SEED: u32 = 42;

/// Input columns, in the puzzle's input order.
pub type Columns = Vec<(&'static str, Vec<f64>)>;

pub struct Dataset {
    pub columns: Columns,
    pub y: Vec<f64>,
}

/// Rows shown by `show`: 25 for beginner puzzles, 100 for physics
/// (their 3D plots need more points to show a surface).
pub fn sample_count(puzzle: &Puzzle) -> usize {
    if puzzle.difficulty == 1 { 25 } else { 100 }
}

fn column<'a>(columns: &'a Columns, name: &str) -> &'a [f64] {
    &columns.iter().find(|(n, _)| *n == name).expect("input column").1
}

fn physics(formula: Formula, fixed: &[(&str, f64)], columns: &Columns, i: usize) -> f64 {
    let c = |name: &str| fixed.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap_or_else(|| column(columns, name)[i]);
    match formula {
        Formula::ProjectileY => {
            let t = c("t");
            c("v0") * c("theta").sin() * t - 0.5 * G_ACCEL * (t * t)
        }
        Formula::ShmEnergy => {
            let (x, v) = (c("x"), c("v"));
            0.5 * c("k") * (x * x) + 0.5 * c("m") * (v * v)
        }
        Formula::TravellingWave => c("A") * (c("k") * c("x") - c("omega") * c("t") + c("phi")).sin(),
        Formula::Coulomb2 => K_COULOMB * (c("q1") / c("r1") + c("q2") / c("r2")),
    }
}

/// The hidden function at every row of `columns`.
pub fn hidden_function(function: &Function, columns: &Columns) -> Vec<f64> {
    let n = columns.first().map_or(0, |(_, v)| v.len());
    (0..n)
        .map(|i| {
            let x = || column(columns, "x")[i];
            match *function {
                Function::Linear { slope, intercept } => slope * x() + intercept,
                Function::Quadratic { a, b, c } => a * (x() * x()) + b * x() + c,
                Function::Sqrt { a } => a * x().sqrt(),
                Function::Log { a, b } => a * x().ln() + b,
                Function::LinearPlusSin { slope, amplitude } => slope * x() + amplitude * x().sin(),
                Function::Physics { formula, fixed } => physics(formula, fixed, columns, i),
                Function::Image { .. } => unreachable!("image puzzles have no numeric inputs"),
            }
        })
        .collect()
}

/// The sample `show` prints: each input drawn uniformly from its range with
/// `np.random.default_rng(42)`, one column after another.
pub fn generate_dataset(puzzle: &Puzzle) -> Dataset {
    let n = sample_count(puzzle);
    let mut rng = Rng::new(SEED);
    let columns: Columns = puzzle.inputs.iter().map(|&(name, lo, hi)| (name, rng.uniform(lo, hi, n))).collect();
    let y = hidden_function(&puzzle.function, &columns);
    Dataset { columns, y }
}

/// Evaluate a puzzle's hidden function at player-supplied rows (one value per input, in order).
pub fn evaluate_points(puzzle: &Puzzle, rows: &[Vec<f64>]) -> Result<Dataset, String> {
    if rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err("Input points must contain only finite numbers.".into());
    }
    let columns: Columns =
        puzzle.input_features().into_iter().enumerate().map(|(j, name)| (name, rows.iter().map(|r| r[j]).collect())).collect();
    let y = hidden_function(&puzzle.function, &columns);
    Ok(Dataset { columns, y })
}
