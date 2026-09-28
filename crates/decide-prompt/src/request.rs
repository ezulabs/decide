//! InternLM's `validate_request` and `_options`, with the same messages.

use crate::pystr;
use serde_json::{Map, Value};

pub const ANSWER_SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
pub const MAX_OPTIONS: usize = 62;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Score,
    Noul,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub name: String,
    pub kind: Kind,
    /// `None` when absent: the reference then prints `''`, not `None`.
    pub instructions: Option<Value>,
    pub criteria: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub state: Value,
    pub questions: Vec<Question>,
}

/// A request the reference rejects (same message), or one decide refuses
/// where the reference would not (the message says so).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError(pub String);

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RequestError {}

pub(crate) fn fail<T>(msg: impl Into<String>) -> Result<T, RequestError> {
    Err(RequestError(msg.into()))
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0), // None: overflows to inf, truthy
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

pub fn validate_request(request: &Value) -> Result<Row, RequestError> {
    let obj = match request {
        Value::Object(m) if m.contains_key("state") => m,
        _ => return fail("Supply a JSON object containing state and questions."),
    };
    // The reference's image checks, with its messages; images that pass them
    // are refused only once the rest of the request has passed too.
    let images = obj.get("images").filter(|v| truthy(v));
    if let Some(images) = images {
        let Value::Array(list) = images else { return fail("Upload 1–8 images.") };
        if list.len() > 8 {
            // truthy, so already nonempty
            return fail("Upload 1–8 images.");
        }
        if !list.iter().all(|i| i.as_str().is_some_and(|s| !s.is_empty())) {
            return fail("Images must be nonempty local file paths.");
        }
    }
    let questions = match obj.get("questions") {
        Some(Value::Object(q)) if (1..=16).contains(&q.len()) => q,
        _ => return fail("Supply 1–16 questions."),
    };
    let mut out = Vec::with_capacity(questions.len());
    for (name, question) in questions {
        if name.is_empty() {
            return fail("Question names must be nonempty strings.");
        }
        let Value::Object(q) = question else { return fail("Question type must be choice, score, or noul.") };
        let kind = match q.get("type").and_then(Value::as_str) {
            Some("choice") => Kind::Choice,
            Some("score") => Kind::Score,
            Some("noul") => Kind::Noul,
            _ => return fail("Question type must be choice, score, or noul."),
        };
        let criteria = q.get("criteria");
        if kind == Kind::Choice && !matches!(criteria, Some(Value::Object(_))) {
            return fail("Choice criteria must be an object.");
        }
        if kind == Kind::Score {
            match criteria {
                Some(Value::Array(_)) => {}
                Some(Value::Object(m)) => {
                    if !m.keys().all(|k| crate::pyfloat::parse(k).is_some_and(f64::is_finite)) {
                        return fail("Score option keys must be finite numbers.");
                    }
                }
                _ => return fail("Score criteria must be a list or object."),
            }
        }
        if kind != Kind::Noul {
            let n = match criteria {
                Some(Value::Array(a)) => a.len(),
                Some(Value::Object(m)) => m.len(),
                _ => 0,
            };
            if !(1..=MAX_OPTIONS).contains(&n) {
                return fail(format!("Supply 1–{MAX_OPTIONS} options."));
            }
        }
        out.push(Question {
            name: name.clone(),
            kind,
            instructions: q.get("instructions").cloned(),
            criteria: criteria.cloned(),
        });
    }
    let row = Row { state: obj["state"].clone(), questions: out };
    crate::compile::compile_row(&row)?; // the reference validates by compiling
    if images.is_some() {
        return fail("decide is text-only: remove `images` (the reference would accept them).");
    }
    Ok(row)
}

fn describe(v: &Value) -> Result<String, RequestError> {
    pystr::str_of(v).map_err(RequestError)
}

/// `_options(question)`: (value, description) in order.
pub fn options(q: &Question) -> Result<Vec<(String, String)>, RequestError> {
    match (q.kind, &q.criteria) {
        (Kind::Choice, Some(Value::Object(m))) | (Kind::Score, Some(Value::Object(m))) => {
            m.iter().map(|(k, v)| Ok((k.clone(), describe(v)?))).collect()
        }
        (Kind::Score, Some(Value::Array(items))) => {
            items.iter().enumerate().map(|(i, v)| Ok((i.to_string(), describe(v)?))).collect()
        }
        (Kind::Noul, criteria) => {
            let empty = Map::new();
            let d = match criteria {
                Some(Value::Object(m)) => m,
                _ => &empty,
            };
            let pick = |keys: [&str; 3], default: &str| -> Result<String, RequestError> {
                for (k, v) in d {
                    if keys.contains(&k.to_lowercase().as_str()) {
                        return describe(v);
                    }
                }
                Ok(default.to_string())
            };
            let yes = pick(["yes", "true", "1"], "The answer is yes (affirmative, or align with the claim).")?;
            let no = pick(["no", "false", "0"], "The answer is no (negative, or disagree with the claim).")?;
            Ok(vec![("no".into(), no), ("yes".into(), yes)])
        }
        (Kind::Choice, _) => fail("choice criteria must be an object"),
        (Kind::Score, _) => fail("score criteria must be a list or object"),
    }
}
