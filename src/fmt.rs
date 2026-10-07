//! Python's number formatting, so printed values and CSV files match the Python game.

/// Split Rust's `{:e}` output ("1.2345e-3") into its digits ("12345") and exponent (-3).
fn sci_parts(s: &str) -> (String, i32) {
    let (mantissa, exp) = s.split_once('e').expect("exponent");
    (mantissa.replace('.', ""), exp.parse().expect("exponent"))
}

fn exponent_suffix(exp: i32) -> String {
    format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

fn special(v: f64) -> Option<String> {
    if v.is_nan() {
        Some("nan".into())
    } else if v.is_infinite() {
        Some(if v > 0.0 { "inf" } else { "-inf" }.into())
    } else {
        None
    }
}

/// Python's `format(v, ".{precision}g")`.
pub fn g(v: f64, precision: usize) -> String {
    if let Some(s) = special(v) {
        return s;
    }
    let p = precision.max(1);
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let (digits, exp) = sci_parts(&format!("{:.*e}", p - 1, v.abs()));
    if exp < -4 || exp >= p as i32 {
        let digits = digits.trim_end_matches('0');
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits.to_string()
        };
        format!("{sign}{mantissa}{}", exponent_suffix(exp))
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        let fixed = format!("{:.*}", decimals, v.abs());
        let fixed = if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            fixed
        };
        format!("{sign}{fixed}")
    }
}

/// Python's `format(v, f"{width}.{precision}g")` with right alignment.
pub fn g_right(v: f64, width: usize, precision: usize) -> String {
    format!("{:>width$}", g(v, precision))
}

/// Python's `repr(float)`: the shortest string that reads back to the same value.
pub fn repr(v: f64) -> String {
    if let Some(s) = special(v) {
        return s;
    }
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let (digits, exp) = sci_parts(&format!("{:e}", v.abs()));
    let n = digits.len() as i32;
    if (-4..16).contains(&exp) {
        if exp < 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        } else if exp + 1 >= n {
            format!("{sign}{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
        } else {
            let point = (exp + 1) as usize;
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        }
    } else {
        let mantissa = if n > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        format!("{sign}{mantissa}{}", exponent_suffix(exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_format() {
        assert_eq!(g(0.0, 5), "0");
        assert_eq!(g(-0.0, 5), "-0");
        assert_eq!(g(123456.0, 5), "1.2346e+05");
        assert_eq!(g(12345.0, 5), "12345");
        assert_eq!(g(0.0001234567, 5), "0.00012346");
        assert_eq!(g(0.00001234567, 5), "1.2346e-05");
        assert_eq!(g(1.5, 5), "1.5");
        assert_eq!(g(99999.5, 5), "1e+05");
        assert_eq!(g(-2.0, 3), "-2");
        assert_eq!(g(8.9875517923e9, 6), "8.98755e+09");
    }

    #[test]
    fn python_repr() {
        assert_eq!(repr(0.0), "0.0");
        assert_eq!(repr(-0.0), "-0.0");
        assert_eq!(repr(3.0), "3.0");
        assert_eq!(repr(0.1), "0.1");
        assert_eq!(repr(1e-5), "1e-05");
        assert_eq!(repr(0.0001), "0.0001");
        assert_eq!(repr(1e16), "1e+16");
        assert_eq!(repr(1234567890123456.0), "1234567890123456.0");
        assert_eq!(repr(1.5e-7), "1.5e-07");
        assert_eq!(repr(123.456), "123.456");
        assert_eq!(repr(-2.5e20), "-2.5e+20");
    }
}
