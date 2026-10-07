//! Puzzle catalogue.
//!
//! Public numbers (puzzle_NN) are catalogue positions, 1 to N in `catalogue()` order.
//! To change the game, edit the puzzle definitions below.
//!
//! Beginner — numerical, one obvious feature, no description
//! Physics  — numerical, two inputs, fixed values stated in the description
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
    /// Text shown by `show` (may be empty).
    pub description: &'static str,
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
// BEGINNER (difficulty = 1) — no description: the data is the whole puzzle
// ===========================================================================

fn beginner(id: &'static str, x_min: f64, x_max: f64, function: Function) -> Puzzle {
    Puzzle { id, difficulty: 1, description: "", inputs: vec![("x", x_min, x_max)], function }
}

// ===========================================================================
// PHYSICS (difficulty = 2)
// Exact outputs. Every quantity that is not an input is fixed, and the fixed
// values are stated in the description the way an exam problem would.
// ===========================================================================

fn physics(
    id: &'static str,
    description: &'static str,
    ranges: [(&'static str, f64, f64); 2],
    fixed: &'static [(&'static str, f64)],
    formula: Formula,
) -> Puzzle {
    Puzzle { id, difficulty: 2, description, inputs: ranges.to_vec(), function: Function::Physics { formula, fixed } }
}

// ===========================================================================
// IMAGES (difficulty = 3)
// ===========================================================================

/// The only text shown for any image puzzle. Kept neutral on purpose: it must
/// not hint at the transform, the number of steps, or whether order matters.
const IMAGE_DESCRIPTION: &str = "Work out what was done to each input picture to make its output picture.";

fn image(id: &'static str, images: &'static [&'static str], transform: &'static str) -> Puzzle {
    Puzzle { id, difficulty: 3, description: IMAGE_DESCRIPTION, inputs: vec![], function: Function::Image { images, transform } }
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
            "A ball is thrown from the ground at a fixed speed of 10 m/s, at an angle theta \
             to the horizontal. Record its height above the ground after time t.",
            [("theta", 0.1, 1.4), ("t", 0.5, 5.0)],
            &[("v0", 10.0)],
            Formula::ProjectileY,
        ),
        physics(
            "shm_energy",
            "A 2 kg mass is on a spring with stiffness 4 N/m. The mass is displaced by x \
             metres and moves with speed v m/s. Record the total mechanical energy.",
            [("x", 0.1, 2.0), ("v", 0.5, 5.0)],
            &[("k", 4.0), ("m", 2.0)],
            Formula::ShmEnergy,
        ),
        physics(
            "travelling_wave",
            "A wave has amplitude 5 m, wavenumber 1 rad/m, angular frequency 1 rad/s and zero \
             phase. Record its displacement at position x metres and time t seconds.",
            [("x", 0.0, 5.0), ("t", 0.0, 5.0)],
            &[("A", 5.0), ("k", 1.0), ("omega", 1.0), ("phi", 0.0)],
            Formula::TravellingWave,
        ),
        physics(
            "coulomb_2",
            "Two point charges sit on a line: q1 = 2 μC and q2 = −3 μC. They are r1 and r2 \
             metres from a measuring point. Record the electric potential at that point.",
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
