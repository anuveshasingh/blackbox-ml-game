//! Blackbox ML Game — discover hidden functions through experimentation.
//!
//! COMMANDS
//!   list                                   List all puzzle IDs.
//!   show <puzzle_id> [--plot]              A puzzle's description and data; image
//!                                          puzzles write their input/output pictures.
//!   apply --input IMAGE --apply NAME...    Image transforms, left to right, on one picture.
//!   transforms                             Every transformation key.
//!   points <puzzle_id> --input FILE [--plot]
//!                                          y for a numerical puzzle at your own points.
//!   residuals <puzzle_id> --input FILE --features SPEC... [--plot]
//!                                          Fit a model to your points and export residuals.
//!
//! Everything is written under `outputs/` in the current directory.

mod canvas;
mod fitting;
mod fmt;
mod generator;
mod images;
mod plots;
mod puzzles;
mod rng;
mod transforms;
mod viewer;
mod viridis;

use std::path::{Path, PathBuf};
use std::process::exit;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};

use crate::fmt::{g_right, repr};
use crate::generator::Dataset;
use crate::plots::{safe_name, Panel};
use crate::puzzles::{catalogue, get_puzzle, puzzle_id_from_number, puzzle_number, Puzzle};
use crate::viewer::{ensure_viewer_extension, open_in_vscode, ViewerStatus};

// ── ANSI colours (VS Code terminal supports these) ────────────────────────

fn b(s: &str) -> String {
    format!("\x1b[1m{s}\x1b[0m")
}
fn green(s: &str) -> String {
    format!("\x1b[92m{s}\x1b[0m")
}
fn cyan(s: &str) -> String {
    format!("\x1b[96m{s}\x1b[0m")
}
fn red(s: &str) -> String {
    format!("\x1b[91m{s}\x1b[0m")
}
fn dim(s: &str) -> String {
    format!("\x1b[2m{s}\x1b[0m")
}

fn sep() -> String {
    dim(&"─".repeat(64))
}
fn sep2() -> String {
    dim(&"═".repeat(64))
}

/// Print a red message and exit with status 1.
fn fail(message: &str) -> ! {
    println!("{}", red(message));
    exit(1)
}

/// An I/O error worded like Python's: "[Errno 2] No such file or directory: 'x'".
pub fn io_error(e: &std::io::Error, path: &Path) -> String {
    match e.raw_os_error() {
        Some(code) => {
            let text = e.to_string();
            let text = text.split(" (os error").next().unwrap_or(&text);
            format!("[Errno {code}] {text}: '{}'", path.display())
        }
        None => format!("{e}: '{}'", path.display()),
    }
}

// ── Command line ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "blackbox-ml-game",
    version,
    about = "Blackbox ML Game — discover hidden functions through experimentation.",
    infer_long_args = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all puzzles
    List,
    /// Show a puzzle and its data
    Show {
        /// Puzzle ID (e.g. puzzle_03)
        puzzle_id: String,
        /// Save plots under outputs/; 3D plots open in VS Code
        #[arg(long)]
        plot: bool,
    },
    /// List all available transforms
    Transforms,
    /// Apply image transforms to one picture
    Apply {
        /// The picture to transform (.jpg, .jpeg or .png)
        #[arg(long, value_name = "IMAGE")]
        input: PathBuf,
        /// Transform names, applied left to right (see `transforms`)
        #[arg(long, num_args = 1.., required = true, value_name = "TRANSFORM")]
        apply: Vec<String>,
    },
    /// Evaluate a puzzle at user-supplied input points
    Points {
        #[command(flatten)]
        points: PointArgs,
    },
    /// Write residuals for a model fitted to supplied points
    Residuals {
        #[command(flatten)]
        points: PointArgs,
        /// Feature specs, e.g. square:t '{"product": ["t", "sin:theta"]}'
        #[arg(long, num_args = 1.., required = true)]
        features: Vec<String>,
        #[arg(long, value_enum, default_value_t = Model::LinearRegression)]
        model: Model,
    },
}

#[derive(clap::Args)]
struct PointArgs {
    /// Puzzle ID (e.g. puzzle_03)
    puzzle_id: String,
    /// Text file with one input point per line
    #[arg(long)]
    input: PathBuf,
    /// CSV path (default: input stem plus puzzle ID and command)
    #[arg(long)]
    output: Option<PathBuf>,
    /// Also save a PNG plot beside the CSV
    #[arg(long)]
    plot: bool,
}

#[derive(Clone, Copy, ValueEnum)]
#[value(rename_all = "snake_case")]
enum Model {
    LinearRegression,
    DecisionTree,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Model::LinearRegression => "linear_regression",
            Model::DecisionTree => "decision_tree",
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            let _ = Cli::command().print_help();
            println!();
        }
        Some(Cmd::List) => cmd_list(),
        Some(Cmd::Show { puzzle_id, plot }) => cmd_show(&puzzle_id, plot),
        Some(Cmd::Transforms) => cmd_transforms(),
        Some(Cmd::Apply { input, apply }) => cmd_apply(&input, &apply),
        Some(Cmd::Points { points }) => cmd_points(&points),
        Some(Cmd::Residuals { points, features, model }) => cmd_residuals(&points, &features, model),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn output_root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join("outputs")
}

fn puzzle_label(puzzle_id: &str) -> String {
    format!("puzzle_{:02}", puzzle_number(puzzle_id).unwrap_or(0))
}

/// "puzzle_07" → the internal ID at number 7; anything else is returned as given.
fn resolve_puzzle_id(identifier: &str) -> String {
    if let Some(rest) = identifier.strip_prefix("puzzle_") {
        if let Ok(number) = rest.trim().parse::<i64>() {
            return puzzle_id_from_number(number).map_or_else(|| identifier.to_string(), str::to_string);
        }
    }
    identifier.to_string()
}

/// Create and return outputs/puzzle_NN for this puzzle only.
fn ensure_puzzle_dir(puzzle_id: &str) -> Result<PathBuf, String> {
    let dir = output_root().join(puzzle_label(puzzle_id));
    std::fs::create_dir_all(&dir).map_err(|e| io_error(&e, &dir))?;
    Ok(dir)
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![];
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
            line = word.to_string();
        } else {
            line = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

// ── Command: list ─────────────────────────────────────────────────────────

fn cmd_list() {
    let puzzles = catalogue();
    println!("\n{} — {} puzzles\n", b("Blackbox ML Game"), puzzles.len());
    println!("{}", sep());
    for p in puzzles {
        println!("  {}", cyan(&puzzle_label(p.id)));
    }
    println!("{}", sep());
    println!("{}", dim("  Run: show <puzzle_id>"));
    println!();
}

// ── Command: show ─────────────────────────────────────────────────────────

fn cmd_show(identifier: &str, plot: bool) {
    let puzzle_id = resolve_puzzle_id(identifier);
    let puzzle = get_puzzle(&puzzle_id).unwrap_or_else(|e| fail(&e));
    if puzzle.is_image() {
        show_image_puzzle(puzzle);
        return;
    }

    let data = generator::generate_dataset(puzzle);
    let (plot_files, plot_note) = if plot {
        write_show_plots(puzzle, &data).unwrap_or_else(|e| fail(&format!("Error: {e}")))
    } else {
        (vec![], None)
    };

    println!("\n{}", sep2());
    println!("  ID: {}", cyan(&puzzle_label(puzzle.id)));
    println!("{}", sep2());
    println!();
    if !puzzle.description.is_empty() {
        for line in wrap(puzzle.description, 62) {
            println!("  {line}");
        }
        println!();
    }
    println!("{}{}", b("  Input features: "), cyan(&puzzle.input_features().join(", ")));
    println!();

    println!("{}", b(&format!("  Sample data ({} rows):", data.y.len())));
    println!();
    let header = format!(
        "  {}  y",
        data.columns.iter().map(|(name, _)| format!("{name:<12}")).collect::<Vec<_>>().join("  ")
    );
    println!("{}", dim(&header));
    println!("{}", dim(&format!("  {}", "─".repeat(header.chars().count() - 2))));
    for i in 0..data.y.len() {
        let row: Vec<String> = data.columns.iter().map(|(_, v)| g_right(v[i], 12, 5)).collect();
        println!("  {}  {}", row.join("  "), g_right(data.y[i], 12, 5));
    }
    println!();
    println!("{}", sep());
    println!("{}", dim("  Record your result on the leaderboard."));
    if let Some(first) = plot_files.first() {
        println!("  Plots written to {}:", b(&first.parent().unwrap_or(Path::new("")).display().to_string()));
        for path in &plot_files {
            println!("    {}", path.file_name().unwrap_or_default().to_string_lossy());
        }
    }
    if let Some(note) = plot_note {
        println!("  {note}");
    }
    println!();
}

/// Write each input/output pair for an image puzzle and show where they are.
fn show_image_puzzle(puzzle: &Puzzle) {
    let pairs = ensure_puzzle_dir(puzzle.id)
        .and_then(|dir| images::render_image_puzzle(puzzle, &dir))
        .unwrap_or_else(|e| fail(&format!("Error: {e}")));

    println!("\n{}", sep2());
    println!("  ID: {}", cyan(&puzzle_label(puzzle.id)));
    println!("{}", sep2());
    println!();
    for line in wrap(puzzle.description, 62) {
        println!("  {line}");
    }
    println!();
    for (input, output) in &pairs {
        println!("  {}  {}", b("Input:"), input.display());
        println!("  {} {}", b("Output:"), output.display());
        println!();
    }
    let files: Vec<PathBuf> = pairs.into_iter().flat_map(|(i, o)| [i, o]).collect();
    if open_in_vscode(&files) {
        println!("  Opened {} pictures in VS Code.", files.len());
    }
    println!("{}", sep());
    println!("{}", dim("  Record your result on the leaderboard."));
    println!();
}

/// Plots for `show --plot`: one scatter of y against each input; physics
/// puzzles also get 3D PLY files, opened in VS Code.
/// Returns (files written, a note for the player or None).
fn write_show_plots(puzzle: &Puzzle, data: &Dataset) -> Result<(Vec<PathBuf>, Option<String>), String> {
    let label = puzzle_label(puzzle.id);
    let plots_dir = ensure_puzzle_dir(puzzle.id)?.join("plots");
    let mut files = vec![];
    for (name, values) in &data.columns {
        let panel = Panel {
            x: values,
            y: &data.y,
            x_label: name.to_string(),
            y_label: "y".into(),
            title: format!("{name} vs y"),
            zero_line: false,
        };
        files.push(plots::save_scatter(&plots_dir.join(format!("y_vs_{}.png", safe_name(name))), &[panel], 7.0, 4.0)?);
    }

    let mut three_d = vec![];
    if puzzle.difficulty >= 2 && data.columns.len() >= 2 {
        three_d = plots::save_3d_plots(&data.columns, &data.y, &plots_dir.join("3d"), &label)?;
        files.extend(three_d.iter().cloned());
    }

    let note = (!three_d.is_empty()).then(|| match ensure_viewer_extension() {
        ViewerStatus::Ready | ViewerStatus::Installed => {
            if open_in_vscode(&three_d) {
                "Opened the 3D plots in VS Code. Drag with the mouse to rotate.".to_string()
            } else {
                "Could not open VS Code. Open the .ply files above by hand.".to_string()
            }
        }
        ViewerStatus::NoVsCode => "VS Code's `code` command is not on PATH, so the 3D plots were not opened. \
             In VS Code run 'Shell Command: Install code command in PATH', then run this again."
            .to_string(),
        ViewerStatus::Failed => "Could not install the PLY viewer automatically. \
             In VS Code, install 'kleinicke.ply-visualizer', then run this again."
            .to_string(),
    });
    Ok((files, note))
}

// ── Command: apply ────────────────────────────────────────────────────────

fn cmd_apply(input: &Path, steps: &[String]) {
    let unknown: Vec<&str> = steps.iter().map(String::as_str).filter(|s| !images::is_image_transform(s)).collect();
    if !unknown.is_empty() {
        println!("{}", red(&format!("Unknown transform(s): {}", unknown.join(", "))));
        let valid: Vec<&str> = images::IMAGE_TRANSFORMS.iter().map(|(n, _)| *n).collect();
        println!("{}", dim(&format!("  Valid names: {}", valid.join(", "))));
        exit(1);
    }
    let source = images::load_user_image(input)
        .unwrap_or_else(|e| fail(&format!("Could not read the image {}: {e}", input.display())));

    let name = input.file_name().unwrap_or_default().to_string_lossy();
    let stem = match name.split('.').next() {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => "image".to_string(),
    };
    let out_dir = output_root().join("apply");
    let steps: Vec<&str> = steps.iter().map(String::as_str).collect();
    let written = images::save_png(&source, &out_dir.join(format!("{stem}.png"))).and_then(|resized| {
        let output = images::apply_pipeline(&source, &steps);
        Ok((resized, images::save_png(&output, &out_dir.join(format!("{stem}_{}.png", steps.join("_"))))?))
    });
    let (resized, output) = written.unwrap_or_else(|e| fail(&format!("Error: {e}")));

    println!("\n{} Applied left to right: {}", green("✓"), b(&steps.join(" → ")));
    println!("  Input (256×256): {}", b(&resized.display().to_string()));
    println!("  Written:         {}", b(&output.display().to_string()));
    if open_in_vscode(std::slice::from_ref(&output)) {
        println!("  Opened in VS Code.");
    }
    println!();
}

// ── Command: transforms ───────────────────────────────────────────────────

fn cmd_transforms() {
    println!("\n{} — use as \"transform:column\" in your features list\n", b("Unary transforms"));
    println!("  {:<18} Description", "Key");
    println!("{}", sep());
    for (key, description, _) in transforms::UNARY {
        println!("  {:<27} {description}", cyan(key));
    }

    println!("\n{} — use as dict in your features list\n", b("Binary transforms"));
    println!("  {:<18} Description", "Key");
    println!("{}", sep());
    for (key, description, _) in transforms::BINARY {
        println!("  {:<27} {description}", cyan(key));
    }

    println!();
    println!("{} — use as --apply names for `apply`, applied left to right\n", b("  Image transforms"));
    for (name, _) in images::IMAGE_TRANSFORMS {
        println!("  {}", cyan(name));
    }
    println!();
    println!("{}", b("  Feature spec formats:"));
    println!("    \"identity:x\"                          → x (unary)");
    println!("    \"square:x\"                            → x²  (unary)");
    println!("    {{\"binary\": \"multiply\", \"a\": \"x1\", \"b\": \"x2\"}}  → x1 × x2");
    println!("    {{\"product\": [\"t\", \"sin:theta\"]}}             → t · sin(theta)");
    println!("    {{\"sum\": [\"x\", {{\"term\": \"t\", \"sign\": -1}}], \"transform\": \"sin\"}}  → sin(x − t)");
    println!();
}

// ── Commands: points and residuals ────────────────────────────────────────

/// Python's `float()` for one value: also accepts underscores between digits.
fn parse_float(s: &str) -> Option<f64> {
    s.parse().ok().or_else(|| {
        let b = s.as_bytes();
        let ok = s.contains('_')
            && b.iter().enumerate().all(|(i, &c)| {
                c != b'_' || (i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit())
            });
        if ok { s.replace('_', "").parse().ok() } else { None }
    })
}

/// Read one whitespace- or comma-separated input row per line.
fn read_points(path: &Path, puzzle: &Puzzle) -> Result<Vec<Vec<f64>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| io_error(&e, path))?;
    let features = puzzle.input_features();
    let mut rows: Vec<Vec<f64>> = vec![];
    for (i, raw) in text.lines().enumerate() {
        let line_number = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let replaced = line.replace(',', " ");
        let values: Vec<&str> = replaced.split_whitespace().collect();
        let row: Option<Vec<f64>> = values.iter().map(|v| parse_float(v)).collect();
        let Some(row) = row else {
            let lower: Vec<String> = values.iter().map(|v| v.to_lowercase()).collect();
            let header: Vec<String> = features.iter().map(|f| f.to_lowercase()).collect();
            if rows.is_empty() && lower == header {
                continue;
            }
            if rows.is_empty() {
                return Err(format!(
                    "Line {line_number} is not numbers and is not this puzzle's header ({}).",
                    features.join(", ")
                ));
            }
            return Err(format!("Line {line_number} contains a non-numeric value."));
        };
        if row.len() != features.len() {
            return Err(format!(
                "Line {line_number} has {} values; expected {} ({}).",
                row.len(),
                features.len(),
                features.join(", ")
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("Input file contains no data rows.".into());
    }
    Ok(rows)
}

/// The numerical puzzle named on the command line, with the player's points evaluated.
fn evaluate(args: &PointArgs) -> Result<(&'static Puzzle, Dataset), String> {
    // Worded like the Python game's KeyError, quotes included.
    let puzzle = get_puzzle(&resolve_puzzle_id(&args.puzzle_id)).map_err(|e| format!("\"{e}\""))?;
    if puzzle.is_image() {
        return Err("image puzzles have no numeric inputs; use show instead".into());
    }
    let rows = read_points(&args.input, puzzle)?;
    Ok((puzzle, generator::evaluate_points(puzzle, &rows)?))
}

fn output_path(args: &PointArgs, puzzle: &Puzzle, suffix: &str) -> Result<PathBuf, String> {
    if let Some(output) = &args.output {
        return Ok(output.clone());
    }
    let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
    Ok(ensure_puzzle_dir(puzzle.id)?.join(format!("{stem}_{}_{suffix}.csv", puzzle_label(puzzle.id))))
}

/// Write a CSV like pandas' `to_csv(index=False)`.
fn write_csv(path: &Path, columns: &[(&str, &[f64])]) -> Result<(), String> {
    let newline = if cfg!(windows) { "\r\n" } else { "\n" };
    let mut out = columns.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(",") + newline;
    for i in 0..columns.first().map_or(0, |(_, v)| v.len()) {
        out += &columns.iter().map(|(_, v)| repr(v[i])).collect::<Vec<_>>().join(",");
        out += newline;
    }
    std::fs::write(path, out).map_err(|e| io_error(&e, path))
}

fn cmd_points(args: &PointArgs) {
    let result = (|| -> Result<_, String> {
        let (puzzle, data) = evaluate(args)?;
        let output = output_path(args, puzzle, "output")?;
        let mut columns: Vec<(&str, &[f64])> = data.columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        columns.push(("y", &data.y));
        write_csv(&output, &columns)?;
        let plot = if args.plot {
            let panels: Vec<Panel> = data
                .columns
                .iter()
                .map(|(name, values)| Panel {
                    x: values,
                    y: &data.y,
                    x_label: name.to_string(),
                    y_label: "y".into(),
                    title: format!("{name} vs y"),
                    zero_line: false,
                })
                .collect();
            let height = (3.5 * panels.len() as f64).max(4.0);
            Some(plots::save_scatter(&output.with_extension("png"), &panels, 7.0, height)?)
        } else {
            None
        };
        Ok((data, output, plot))
    })();
    let (data, output, plot) = result.unwrap_or_else(|e| fail(&format!("Error: {e}")));

    let names: Vec<&str> = data.columns.iter().map(|(n, _)| *n).collect();
    println!("\n{} Points evaluated: {}", green("✓"), b(&output.display().to_string()));
    println!("  {} rows, columns: {} + y\n", data.y.len(), names.join(", "));
    if let Some(plot) = plot {
        println!("  Plot written: {}\n", b(&plot.display().to_string()));
    }
}

fn cmd_residuals(args: &PointArgs, features: &[String], model: Model) {
    let result = (|| -> Result<_, String> {
        let (puzzle, data) = evaluate(args)?;
        let specs = fitting::parse_features(features)?;
        let matrix = fitting::build_feature_matrix(&data.columns, &specs)?;
        let predictions = fitting::predict_model(&matrix, &data.y, model.name())?;
        let residuals: Vec<f64> = data.y.iter().zip(&predictions).map(|(y, p)| y - p).collect();
        let output = output_path(args, puzzle, "residuals")?;
        let mut columns: Vec<(&str, &[f64])> = data.columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        columns.extend([("y", data.y.as_slice()), ("prediction", &predictions), ("residual", &residuals)]);
        write_csv(&output, &columns)?;
        let plot = if args.plot {
            let panel = Panel {
                x: &predictions,
                y: &residuals,
                x_label: "prediction".into(),
                y_label: "residual (y - prediction)".into(),
                title: "Residuals vs prediction".into(),
                zero_line: true,
            };
            Some(plots::save_scatter(&output.with_extension("png"), &[panel], 7.0, 4.0)?)
        } else {
            None
        };
        Ok((data.y.len(), output, plot))
    })();
    let (rows, output, plot) = result.unwrap_or_else(|e| fail(&format!("Error: {e}")));

    println!("\n{} Residuals written: {}", green("✓"), b(&output.display().to_string()));
    println!("  {rows} rows, model: {}\n", model.name());
    if let Some(plot) = plot {
        println!("  Plot written: {}\n", b(&plot.display().to_string()));
    }
}
