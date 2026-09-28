//! Python's float formatting and parsing, at the places the reference prints
//! or reads numbers: `repr` (f-strings, json.dumps), `json` (json.dumps, which
//! differs only for NaN/inf), and `parse` (`float(str)` on score option keys).

/// Python's `repr(x)`.
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (negative, digits, decpt) = shortest_digits(x);
    let n = digits.len() as i32;
    let mut out = String::with_capacity(24);
    if negative {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if decpt >= n {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (decpt - n) as usize));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// What `json.dumps` prints for a float.
pub fn json(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else {
        repr(x)
    }
}

/// Shortest round-trip digits of finite nonzero `x`:
/// (negative, digits without leading/trailing zeros, decpt) with |x| = 0.d1d2…dn × 10^decpt.
fn shortest_digits(x: f64) -> (bool, String, i32) {
    let mut buf = ryu::Buffer::new();
    let s = buf.format_finite(x.abs());
    let (mantissa, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i32>().unwrap()),
        None => (s, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = format!("{int_part}{frac_part}");
    let mut decpt = int_part.len() as i32 + exp;
    let lead = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..lead);
    decpt -= lead as i32;
    let keep = digits.trim_end_matches('0').len();
    digits.truncate(keep);
    (x < 0.0, digits, decpt)
}

/// Python's `float(s)` for a str; `None` where Python raises ValueError.
pub fn parse(s: &str) -> Option<f64> {
    // float() maps non-ASCII whitespace to ' ' and then strips C-locale
    // whitespace, so U+001C..U+001F stay (str.isspace() counts them).
    let t = s.trim_matches(|c| crate::pystr::is_space(c) && !matches!(c, '\u{1c}'..='\u{1f}'));
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    let (sign, unsigned) = match lower.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, lower.strip_prefix('+').unwrap_or(&lower)),
    };
    match unsigned {
        "inf" | "infinity" => return Some(sign * f64::INFINITY),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    let b = t.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' {
            let prev_digit = i > 0 && b[i - 1].is_ascii_digit();
            let next_digit = b.get(i + 1).is_some_and(|c| c.is_ascii_digit());
            if !(prev_digit && next_digit) {
                return None;
            }
        }
    }
    let cleaned: String = t.chars().filter(|&c| c != '_').collect();
    is_decimal_literal(&cleaned).then(|| cleaned.parse::<f64>().ok()).flatten()
}

/// `[+-]? (digits [. digits*] | . digits) ([eE] [+-]? digits)?`, ASCII only.
fn is_decimal_literal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let digits = |i: &mut usize| {
        let start = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        *i - start
    };
    let int_digits = digits(&mut i);
    let mut frac_digits = 0;
    if b.get(i) == Some(&b'.') {
        i += 1;
        frac_digits = digits(&mut i);
    }
    if int_digits + frac_digits == 0 {
        return false;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return false;
        }
    }
    i == b.len()
}
