//! `json.dumps(value, ensure_ascii=False, indent=N)` over serde_json values
//! parsed with key order preserved (`preserve_order`) and number literals kept
//! as written (`arbitrary_precision`), so ints and floats are classified the
//! way `json.loads` would classify them.

use serde_json::{Number, Value};

pub enum PyNum {
    Int(String),
    Float(f64),
}

/// What `json.loads` turns a JSON number literal into.
pub fn classify(n: &Number) -> PyNum {
    let text = n.to_string();
    if text.contains(['.', 'e', 'E']) {
        PyNum::Float(text.parse::<f64>().expect("serde_json validated the literal"))
    } else if text.trim_start_matches('-').bytes().all(|b| b == b'0') {
        PyNum::Int("0".into())
    } else {
        PyNum::Int(text)
    }
}

/// Python 3.12's default `sys.get_int_max_str_digits()`.
const INT_MAX_STR_DIGITS: usize = 4300;

/// The ValueError `json.loads(text)` raises for the first int literal of more
/// than 4300 digits (sign excluded). It fails wherever the literal is, even
/// under a key the reference ignores or a later duplicate replaces, so this
/// scans the text, not the parsed value. `text` must be valid JSON.
pub fn int_digits_error(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < b.len() && matches!(b[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    i += 1;
                }
                let literal = &text[start..i];
                let digits = literal.trim_start_matches('-').len();
                if digits > INT_MAX_STR_DIGITS && !literal.contains(['.', 'e', 'E']) {
                    return Some(format!(
                        "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: \
                         value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
                    ));
                }
            }
            _ => i += 1,
        }
    }
    None
}

pub fn dumps(v: &Value, indent: usize) -> String {
    let mut out = String::new();
    write_value(&mut out, v, indent, 0);
    out
}

fn write_value(out: &mut String, v: &Value, indent: usize, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match classify(n) {
            PyNum::Int(t) => out.push_str(&t),
            PyNum::Float(x) => out.push_str(&crate::pyfloat::json(x)),
        },
        Value::String(s) => write_str(out, s),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, indent, level + 1);
                write_value(out, item, indent, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, indent, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write_value(out, item, indent, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', indent * level));
}

/// A JSON string with `ensure_ascii=False`: only `"`, `\` and C0 controls are escaped.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
