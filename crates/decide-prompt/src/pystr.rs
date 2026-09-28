//! Python `str()` of JSON scalars and `str.strip()`, as the reference uses them.

use serde_json::Value;

/// Python's `str.isspace()` for one character.
pub fn is_space(c: char) -> bool {
    matches!(c,
        '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r' | '\u{1c}'..='\u{1f}' | ' ' | '\u{85}' | '\u{a0}'
        | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}'
        | '\u{3000}')
}

/// Python's `s.strip()` (and Jinja's `|trim`, which calls it).
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// Python's `str(v)` for a value as `json.loads` produced it. Lists and dicts
/// are refused (documented divergence): the reference would print their repr.
pub fn str_of(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Bool(true) => Ok("True".into()),
        Value::Bool(false) => Ok("False".into()),
        Value::Null => Ok("None".into()),
        Value::Number(n) => Ok(match crate::pyjson::classify(n) {
            crate::pyjson::PyNum::Int(t) => t,
            crate::pyjson::PyNum::Float(x) => crate::pyfloat::repr(x),
        }),
        Value::Array(_) | Value::Object(_) => Err(
            "decide accepts only strings, numbers, booleans or null as instructions and option \
             descriptions (the reference would print a Python repr)".into(),
        ),
    }
}
