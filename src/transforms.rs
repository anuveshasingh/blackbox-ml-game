//! Feature transformations for `residuals`.
//!
//! Invalid domains give NaN instead of an error; model fitting then rejects
//! features containing NaN values.

/// (key, description, function). Order is the order `transforms` lists them.
pub const UNARY: &[(&str, &str, fn(f64) -> f64)] = &[
    ("identity", "x  (no change)", |x| x),
    ("square", "x²", |x| x * x),
    ("cube", "x³", |x| x.powf(3.0)),
    ("sqrt", "√x", |x| if x >= 0.0 { x.sqrt() } else { f64::NAN }),
    ("abs", "|x|", f64::abs),
    ("log", "log(x)", |x| if x > 0.0 { x.ln() } else { f64::NAN }),
    ("log2", "log₂(x)", |x| if x > 0.0 { x.log2() } else { f64::NAN }),
    ("reciprocal", "1/x", |x| if x.abs() > 1e-10 { 1.0 / x } else { f64::NAN }),
    ("sin", "sin(x)", f64::sin),
    ("cos", "cos(x)", f64::cos),
    ("exp", "eˣ", |x| x.clamp(-100.0, 100.0).exp()),
    ("exp_neg", "e⁻ˣ", |x| (-x.clamp(-100.0, 100.0)).exp()),
];

pub const BINARY: &[(&str, &str, fn(f64, f64) -> f64)] = &[
    ("multiply", "x1 × x2", |a, b| a * b),
    ("divide", "x1 / x2", |a, b| if b.abs() > 1e-10 { a / b } else { f64::NAN }),
    ("add", "x1 + x2", |a, b| a + b),
    ("subtract", "x1 − x2", |a, b| a - b),
    ("distance", "√(x1² + x2²)", |a, b| (a * a + b * b).sqrt()),
];

fn sorted_keys<T>(table: &[(&str, &str, T)]) -> String {
    let mut keys: Vec<String> = table.iter().map(|(k, _, _)| format!("'{k}'")).collect();
    keys.sort();
    format!("[{}]", keys.join(", "))
}

pub fn apply_transform(key: &str, x: &[f64]) -> Result<Vec<f64>, String> {
    let (_, _, f) = UNARY
        .iter()
        .find(|(k, _, _)| *k == key)
        .ok_or_else(|| format!("Unknown transform '{key}'. Available: {}", sorted_keys(UNARY)))?;
    Ok(x.iter().map(|&v| f(v)).collect())
}

pub fn apply_binary_transform(key: &str, a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    let (_, _, f) = BINARY
        .iter()
        .find(|(k, _, _)| *k == key)
        .ok_or_else(|| format!("Unknown binary transform '{key}'. Available: {}", sorted_keys(BINARY)))?;
    Ok(a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect())
}
