//! Feature building and model fitting for the `residuals` command.
//!
//! `build_feature_matrix` turns feature specifications (columns, transforms,
//! products, sums) into feature columns. `predict_model` fits linear
//! regression or a shallow decision tree and returns predictions for the same
//! rows. Both models follow scikit-learn's `LinearRegression` and
//! `DecisionTreeRegressor(max_depth=4)`, which the Python game used.

use serde_json::Value;

use crate::generator::Columns;
use crate::transforms::{apply_binary_transform, apply_transform};

/// Decision-tree depth kept shallow so it stays interpretable.
const DT_MAX_DEPTH: usize = 4;

/// A feature specification: a plain string, or JSON for products, sums and binaries.
pub enum Spec {
    Text(String),
    Json(Value),
}

/// Plain specs stay strings; specs starting with '{' are JSON (products, sums, binaries).
pub fn parse_features(specs: &[String]) -> Result<Vec<Spec>, String> {
    specs
        .iter()
        .map(|spec| {
            if spec.trim_start().starts_with('{') {
                serde_json::from_str(spec).map(Spec::Json).map_err(|_| format!("Feature is not valid JSON: {spec}"))
            } else {
                Ok(Spec::Text(spec.clone()))
            }
        })
        .collect()
}

/// One column per feature specification.
///
/// * `"x"` — a column as-is
/// * `"square:x"` — a unary transform of a column
/// * `{"binary": "multiply", "a": "x1", "b": "x2"}` — a binary transform of two columns
/// * `{"product": [spec, ...]}` — element-wise product of nested specs
/// * `{"sum": [spec, {"term": spec, "sign": -1}, ...]}` — sum of nested specs, with signs
///
/// A `product` or `sum` may also carry `"transform": key` to transform its result.
pub fn build_feature_matrix(x: &Columns, features: &[Spec]) -> Result<Vec<Vec<f64>>, String> {
    if features.is_empty() {
        return Err("features list must not be empty.".into());
    }
    features
        .iter()
        .map(|spec| match spec {
            Spec::Text(s) => eval_text(x, s),
            Spec::Json(v) => eval_json(x, v),
        })
        .collect()
}

fn column(x: &Columns, name: &str) -> Result<Vec<f64>, String> {
    x.iter().find(|(n, _)| *n == name).map(|(_, v)| v.clone()).ok_or_else(|| format!("Column '{name}' not in dataset."))
}

fn eval_text(x: &Columns, spec: &str) -> Result<Vec<f64>, String> {
    match spec.split_once(':') {
        Some((key, name)) => {
            let values = column(x, name.trim())?;
            apply_transform(key.trim(), &values)
        }
        None => column(x, spec),
    }
}

/// A JSON value written the way Python prints the parsed value (`{'a': 1}`), for error messages.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => crate::fmt::repr(n.as_f64().unwrap_or(f64::NAN)),
        Value::Number(n) => n.to_string(),
        Value::String(s) if s.contains('\'') && !s.contains('"') => format!("\"{s}\""),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => {
            let items: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

fn eval_json(x: &Columns, spec: &Value) -> Result<Vec<f64>, String> {
    let Value::Object(map) = spec else {
        return match spec {
            Value::String(s) => eval_text(x, s),
            other => Err(format!("Feature must be a string or a JSON object: {}", py_repr(other))),
        };
    };
    let nested = |key: &str| -> Result<&Vec<Value>, String> {
        map[key].as_array().ok_or_else(|| format!("'{key}' must be a list of features: {}", py_repr(spec)))
    };
    let result = if map.contains_key("product") {
        reduce(nested("product")?.iter().map(|item| eval_json(x, item)), |a, b| a * b)?
    } else if map.contains_key("sum") {
        let terms = nested("sum")?.iter().map(|item| {
            let (sign, term) = match item {
                Value::Object(m) if m.contains_key("term") => {
                    let sign = match m.get("sign") {
                        None => 1.0,
                        Some(s) => s.as_f64().ok_or_else(|| format!("'sign' must be a number: {}", py_repr(item)))?,
                    };
                    (sign, &m["term"])
                }
                _ => (1.0, item),
            };
            Ok(eval_json(x, term)?.into_iter().map(|v| sign * v).collect())
        });
        reduce(terms, |a, b| a + b)?
    } else {
        let field = |key: &str| map.get(key).and_then(Value::as_str);
        let (Some(binary), Some(a), Some(b)) = (field("binary"), field("a"), field("b")) else {
            return Err(format!("Feature dict must have 'product', 'sum', or 'binary'/'a'/'b': {}", py_repr(spec)));
        };
        let (a, b) = (column(x, a)?, column(x, b)?);
        return apply_binary_transform(binary, &a, &b);
    };
    match map.get("transform") {
        None => Ok(result),
        Some(Value::String(key)) => apply_transform(key, &result),
        Some(other) => Err(format!("'transform' must be a transform name: {}", py_repr(other))),
    }
}

/// Combine nested feature columns element-wise, first to last.
fn reduce(
    mut items: impl Iterator<Item = Result<Vec<f64>, String>>,
    op: fn(f64, f64) -> f64,
) -> Result<Vec<f64>, String> {
    let mut acc = items.next().ok_or("A product or sum needs at least one feature.")??;
    for item in items {
        acc.iter_mut().zip(item?).for_each(|(a, b)| *a = op(*a, b));
    }
    Ok(acc)
}

/// Fit a supported model and return predictions for the same rows.
pub fn predict_model(features: &[Vec<f64>], y: &[f64], model: &str) -> Result<Vec<f64>, String> {
    if features.iter().flatten().any(|v| !v.is_finite()) {
        return Err("A feature contains invalid values (NaN). Check the transform domain.".into());
    }
    match model {
        "linear_regression" => Ok(linear_regression(features, y)),
        "decision_tree" => decision_tree(features, y),
        _ => Err(format!("Unknown model '{model}'. Use 'linear_regression' or 'decision_tree'.")),
    }
}

// ---------------------------------------------------------------------------
// Linear regression — ordinary least squares with an intercept
// ---------------------------------------------------------------------------

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// `LinearRegression().fit(X, y).predict(X)` after scaling each column to unit
/// max-magnitude. Predictions are unchanged by the scaling (least squares is
/// scale-invariant), but without it a column near 1e10 would swamp O(1) columns.
/// Like scipy's `lstsq`, singular values below machine epsilon × the largest are
/// treated as zero, so duplicate features give the minimum-norm solution.
fn linear_regression(features: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let x: Vec<Vec<f64>> = features
        .iter()
        .map(|col| {
            let scale = col.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let scale = if scale == 0.0 { 1.0 } else { scale };
            col.iter().map(|v| v / scale).collect()
        })
        .collect();
    let x_offset: Vec<f64> = x.iter().map(|col| mean(col)).collect();
    let y_offset = mean(y);
    let centred: Vec<Vec<f64>> = x.iter().zip(&x_offset).map(|(col, m)| col.iter().map(|v| v - m).collect()).collect();
    let y_centred: Vec<f64> = y.iter().map(|v| v - y_offset).collect();

    let coef = least_squares(centred, &y_centred);
    let intercept = y_offset - x_offset.iter().zip(&coef).map(|(m, c)| m * c).sum::<f64>();
    (0..n).map(|i| x.iter().zip(&coef).map(|(col, c)| col[i] * c).sum::<f64>() + intercept).collect()
}

/// Minimum-norm least-squares solution of `a · coef = b` (columns of `a` given),
/// by one-sided Jacobi SVD.
fn least_squares(mut a: Vec<Vec<f64>>, b: &[f64]) -> Vec<f64> {
    let p = a.len();
    let mut v: Vec<Vec<f64>> = (0..p).map(|j| (0..p).map(|k| if j == k { 1.0 } else { 0.0 }).collect()).collect();
    let dot = |u: &[f64], w: &[f64]| u.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();

    for _sweep in 0..60 {
        let mut rotated = false;
        for j in 0..p {
            for k in j + 1..p {
                let alpha = dot(&a[j], &a[j]);
                let beta = dot(&a[k], &a[k]);
                let gamma = dot(&a[j], &a[k]);
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for m in [&mut a, &mut v] {
                    let (left, right) = m.split_at_mut(k);
                    for (x, y) in left[j].iter_mut().zip(right[0].iter_mut()) {
                        let (xj, xk) = (*x, *y);
                        *x = c * xj - s * xk;
                        *y = s * xj + c * xk;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // a = U Σ now (columns), so coef = V Σ⁺ Uᵀ b = Σ_j v_j (a_j · b) / σ_j².
    let sigma: Vec<f64> = a.iter().map(|col| dot(col, col).sqrt()).collect();
    let cutoff = f64::EPSILON * sigma.iter().cloned().fold(0.0, f64::max);
    let mut coef = vec![0.0; p];
    for j in 0..p {
        if sigma[j] > cutoff && sigma[j] > 0.0 {
            let weight = dot(&a[j], b) / (sigma[j] * sigma[j]);
            for (c, vj) in coef.iter_mut().zip(&v[j]) {
                *c += weight * vj;
            }
        }
    }
    coef
}

// ---------------------------------------------------------------------------
// Decision tree — scikit-learn's CART regressor with the "best" splitter
// ---------------------------------------------------------------------------

/// scikit-learn compares feature values in float32 and treats values closer
/// than this as equal.
const FEATURE_THRESHOLD: f32 = 1e-7;

struct Tree<'a> {
    x: Vec<Vec<f32>>,
    y: &'a [f64],
    samples: Vec<usize>,
    predictions: Vec<f64>,
}

/// Running sums for one node, updated like scikit-learn's `RegressionCriterion`.
struct Criterion {
    start: usize,
    end: usize,
    pos: usize,
    sum_total: f64,
    sq_sum_total: f64,
    sum_left: f64,
}

impl Tree<'_> {
    fn criterion(&self, start: usize, end: usize) -> Criterion {
        let (mut sum, mut sq) = (0.0, 0.0);
        for &i in &self.samples[start..end] {
            sum += self.y[i];
            sq += self.y[i] * self.y[i];
        }
        Criterion { start, end, pos: start, sum_total: sum, sq_sum_total: sq, sum_left: 0.0 }
    }

    fn update(&self, c: &mut Criterion, new_pos: usize) {
        if new_pos - c.pos <= c.end - new_pos {
            for &i in &self.samples[c.pos..new_pos] {
                c.sum_left += self.y[i];
            }
        } else {
            c.sum_left = c.sum_total;
            for &i in self.samples[new_pos..c.end].iter().rev() {
                c.sum_left -= self.y[i];
            }
        }
        c.pos = new_pos;
    }

    fn build(&mut self, start: usize, end: usize, depth: usize, impurity: f64) {
        let mut c = self.criterion(start, end);
        let n_node = end - start;
        let mut is_leaf = depth >= DT_MAX_DEPTH || n_node < 2 || impurity <= f64::EPSILON;

        let mut split = None;
        if !is_leaf {
            split = self.best_split(&mut c, impurity);
            is_leaf = split.is_none();
        }
        if let Some((pos, impurity_left, impurity_right, improvement)) = split {
            if improvement + f64::EPSILON < 0.0 {
                is_leaf = true;
            } else {
                self.build(start, pos, depth + 1, impurity_left);
                self.build(pos, end, depth + 1, impurity_right);
            }
        }
        if is_leaf {
            let value = c.sum_total / n_node as f64;
            for p in start..end {
                self.predictions[self.samples[p]] = value;
            }
        }
    }

    /// Returns (pos, impurity_left, impurity_right, improvement) of the best split.
    fn best_split(&mut self, c: &mut Criterion, impurity: f64) -> Option<(usize, f64, f64, f64)> {
        let (start, end) = (c.start, c.end);
        let mut best: Option<(usize, usize, f64)> = None; // (feature, pos, threshold)
        let mut best_proxy = f64::NEG_INFINITY;

        for f in 0..self.x.len() {
            let x = &self.x[f];
            self.samples[start..end].sort_by(|&a, &b| x[a].total_cmp(&x[b]));
            let values: Vec<f32> = self.samples[start..end].iter().map(|&i| x[i]).collect();
            if values[values.len() - 1] <= values[0] + FEATURE_THRESHOLD {
                continue; // constant feature
            }
            c.sum_left = 0.0;
            c.pos = start;
            let mut p = 0;
            loop {
                while p + 1 < values.len() && values[p + 1] <= values[p] + FEATURE_THRESHOLD {
                    p += 1;
                }
                let p_prev = p;
                p += 1;
                if p >= values.len() {
                    break;
                }
                self.update(c, start + p);
                let n_left = p as f64;
                let n_right = (values.len() - p) as f64;
                let sum_right = c.sum_total - c.sum_left;
                let proxy = c.sum_left * c.sum_left / n_left + sum_right * sum_right / n_right;
                if proxy > best_proxy {
                    best_proxy = proxy;
                    let mut threshold = values[p_prev] as f64 / 2.0 + values[p] as f64 / 2.0;
                    if threshold == values[p] as f64 || !threshold.is_finite() {
                        threshold = values[p_prev] as f64;
                    }
                    best = Some((f, start + p, threshold));
                }
            }
        }

        let (feature, pos, threshold) = best?;
        // Partition exactly like scikit-learn: values <= threshold to the left.
        let x = &self.x[feature];
        let (mut p, mut partition_end) = (start, end);
        while p < partition_end {
            if (x[self.samples[p]] as f64) <= threshold {
                p += 1;
            } else {
                partition_end -= 1;
                self.samples.swap(p, partition_end);
            }
        }

        c.sum_left = 0.0;
        c.pos = start;
        self.update(c, pos);
        let sq_left: f64 = self.samples[start..pos].iter().map(|&i| self.y[i] * self.y[i]).sum();
        let (n_left, n_right) = ((pos - start) as f64, (end - pos) as f64);
        let sum_right = c.sum_total - c.sum_left;
        let impurity_left = sq_left / n_left - (c.sum_left / n_left).powi(2);
        let impurity_right = (c.sq_sum_total - sq_left) / n_right - (sum_right / n_right).powi(2);
        let n_node = (end - start) as f64;
        let n_total = self.y.len() as f64;
        let improvement =
            (n_node / n_total) * (impurity - (n_right / n_node * impurity_right) - (n_left / n_node * impurity_left));
        Some((pos, impurity_left, impurity_right, improvement))
    }
}

fn decision_tree(features: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, String> {
    if features.iter().flatten().any(|v| v.abs() > f32::MAX as f64) {
        return Err("Input X contains infinity or a value too large for dtype('float32').".into());
    }
    let n = y.len();
    let mut tree = Tree {
        x: features.iter().map(|col| col.iter().map(|&v| v as f32).collect()).collect(),
        y,
        samples: (0..n).collect(),
        predictions: vec![0.0; n],
    };
    let root = tree.criterion(0, n);
    let impurity = root.sq_sum_total / n as f64 - (root.sum_total / n as f64).powi(2);
    tree.build(0, n, 0, impurity);
    Ok(tree.predictions)
}
