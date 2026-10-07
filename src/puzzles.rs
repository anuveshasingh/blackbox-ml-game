//! Puzzle catalogue.
//!
//! Public numbers (puzzle_NN) are catalogue positions, 1 to N in `catalogue()` order.
//! To change the game, edit the puzzle definitions below.
//!
//! Beginner — numerical, one obvious feature
//! Physics  — numerical, two inputs, fixed physical constants
//!
//! No puzzle has any description: `show` prints only the ID and the data.
//! Images   — curated picture + one transform or a pipeline of transforms
//!
//! Physical formulas live in `generator.rs`. Image transforms, pipelines and
//! curated pictures live in `images.rs` and `assets/curated/`.

use std::sync::OnceLock;

use crate::images::PIPELINES;

/// The hidden function inside a puzzle.
#[derive(Debug, Clone)]
pub enum Function {
    Linear { slope: f64, intercept: f64 },
    Quadratic { a: f64, b: f64, c: f64 },
    Sqrt { a: f64 },
    Log { a: f64, b: f64 },
    LinearPlusSin { slope: f64, amplitude: f64 },
    Physics { formula: Formula, fixed: &'static [(&'static str, f64)] },
    /// `images` are curated picture names: one picture, or several examples of the
    /// same transform. `transform` is a name in `IMAGE_TRANSFORMS` or `PIPELINES`.
    Image { images: &'static [&'static str], transform: &'static str },
}

#[derive(Debug, Clone, Copy)]
pub enum Formula {
    ProjectileY,
    ShmEnergy,
    TravellingWave,
    Coulomb2,
}

/// One game puzzle.
#[derive(Debug, Clone)]
pub struct Puzzle {
    /// Internal name (never shown; players see puzzle_NN).
    pub id: &'static str,
    /// 1 beginner, 2 physics, 3 image.
    pub difficulty: u8,
    /// Input column names and their sampling ranges (empty for image puzzles).
    pub inputs: Vec<(&'static str, f64, f64)>,
    pub function: Function,
}

impl Puzzle {
    pub fn input_features(&self) -> Vec<&'static str> {
        self.inputs.iter().map(|(name, _, _)| *name).collect()
    }

    pub fn is_image(&self) -> bool {
        matches!(self.function, Function::Image { .. })
    }
}

// ===========================================================================
// BEGINNER (difficulty = 1)
// ===========================================================================

fn beginner(id: &'static str, x_min: f64, x_max: f64, function: Function) -> Puzzle {
    Puzzle { id, difficulty: 1, inputs: vec![("x", x_min, x_max)], function }
}

// ===========================================================================
// PHYSICS (difficulty = 2)
// Exact outputs. Every quantity that is not an input is fixed.
// ===========================================================================

fn physics(
    id: &'static str,
    ranges: [(&'static str, f64, f64); 2],
    fixed: &'static [(&'static str, f64)],
    formula: Formula,
) -> Puzzle {
    Puzzle { id, difficulty: 2, inputs: ranges.to_vec(), function: Function::Physics { formula, fixed } }
}

// ===========================================================================
// IMAGES (difficulty = 3)
// ===========================================================================

fn image(id: &'static str, images: &'static [&'static str], transform: &'static str) -> Puzzle {
    Puzzle { id, difficulty: 3, inputs: vec![], function: Function::Image { images, transform } }
}

// ===========================================================================
// Catalogue — order here sets the public puzzle numbers
// ===========================================================================

fn build() -> Vec<Puzzle> {
    use Function::*;
    let mut puzzles = vec![
        beginner("line_01", -5.0, 5.0, Linear { slope: 3.0, intercept: 15.0 }),
        beginner("line_02", -5.0, 5.0, Linear { slope: -2.0, intercept: 20.0 }),
        beginner("square_01", -5.0, 5.0, Quadratic { a: 1.0, b: 0.0, c: 0.0 }),
        beginner("sqrt_01", 0.5, 25.0, Sqrt { a: 2.0 }),
        beginner("log_01", 1.0, 100.0, Log { a: 3.0, b: 1.0 }),
        beginner("line_sin_01", -10.0, 10.0, LinearPlusSin { slope: 1.0, amplitude: 2.0 }),
        physics(
            "projectile_y",
            [("theta", 0.1, 1.4), ("t", 0.5, 5.0)],
            &[("v0", 10.0)],
            Formula::ProjectileY,
        ),
        physics(
            "shm_energy",
            [("x", 0.1, 2.0), ("v", 0.5, 5.0)],
            &[("k", 4.0), ("m", 2.0)],
            Formula::ShmEnergy,
        ),
        physics(
            "travelling_wave",
            [("x", 0.0, 5.0), ("t", 0.0, 5.0)],
            &[("A", 5.0), ("k", 1.0), ("omega", 1.0), ("phi", 0.0)],
            Formula::TravellingWave,
        ),
        physics(
            "coulomb_2",
            [("r1", 1.0, 5.0), ("r2", 1.0, 5.0)],
            &[("q1", 2e-6), ("q2", -3e-6)],
            Formula::Coulomb2,
        ),
        // Single transforms. Two pictures = two examples of the same puzzle.
        image("rotate_chunks", &["lsd", "checkmate"], "rotate_chunks"),
        image("mirror_sum", &["moon", "molecule"], "mirror_sum"),
        image("circular_shift", &["matrix"], "circular_shift"),
        image("swap_rgb_rbg", &["marbles", "monet"], "swap_rgb_rbg"),
    ];
    // Pipelines, in images::PIPELINES order.
    for (name, picture, _) in PIPELINES {
        puzzles.push(image(name, std::slice::from_ref(picture), name));
    }
    puzzles
}

pub fn catalogue() -> &'static [Puzzle] {
    static CATALOGUE: OnceLock<Vec<Puzzle>> = OnceLock::new();
    CATALOGUE.get_or_init(build)
}

/// The public number (the NN in puzzle_NN): the 1-based catalogue position.
pub fn puzzle_number(puzzle_id: &str) -> Option<usize> {
    catalogue().iter().position(|p| p.id == puzzle_id).map(|i| i + 1)
}

/// The puzzle ID at a public number, or None if out of range.
pub fn puzzle_id_from_number(number: i64) -> Option<&'static str> {
    let index = usize::try_from(number).ok()?.checked_sub(1)?;
    catalogue().get(index).map(|p| p.id)
}

pub fn get_puzzle(puzzle_id: &str) -> Result<&'static Puzzle, String> {
    catalogue().iter().find(|p| p.id == puzzle_id).ok_or_else(|| format!("Puzzle '{puzzle_id}' not found."))
}
