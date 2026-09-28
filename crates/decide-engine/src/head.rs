//! The decision head: InternLM's `DecisionEngine.predict` from the logits on,
//! line for line (`inference.py`: `predict`, `scale_result`,
//! `scale_probabilities`, `argmax`). Per field, a softmax over the candidate
//! logits in f32, temperature calibration in f64, then the Jev-envelope
//! response in Python's dict insertion order.
//!
//! The Python semantics it depends on: `sum()` of floats is 3.12's Neumaier
//! sum; `min(key=(-p, label))` keeps the first minimum and compares labels by
//! code point; `round(x, 2)` is correctly rounded; `float(label)` is
//! [`pyfloat::parse`]; `math.log`/`math.exp` are the libm calls Rust's
//! `ln`/`exp` make. Floats are stored as their Python `repr`.

use crate::config::TEMPERATURE;
use decide_prompt::pyfloat;
use decide_prompt::request::Kind;
use decide_prompt::tokenize::Prepared;
use serde_json::{Map, Number, Value};
use std::str::FromStr;

/// `MODEL_NAME` in inference.py.
pub const MODEL: &str = "Intern-Decision-4B";
/// The response's `backend` (InternLM's is `"hf"`).
pub const BACKEND: &str = "decide-metal";

/// An error the reference raises after its forward pass (same message), or
/// logits that do not fit the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadError(pub String);

impl std::fmt::Display for HeadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HeadError {}

fn fail<T>(msg: impl Into<String>) -> Result<T, HeadError> {
    Err(HeadError(msg.into()))
}

/// `torch.softmax(logits.float(), dim=-1)` in f32: max, `exp(x − max)`, sum,
/// divide. Torch's CPU kernel multiplies by `1/sum` and uses Sleef's `expf`, so
/// its values can differ from these by an ulp or two.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum = exps.iter().fold(0.0f32, |s, &e| s + e);
    exps.iter().map(|&e| e / sum).collect()
}

/// Refuses labels and probabilities that cannot be one dict.
fn check_pairs<S>(labels: &[S], probs: &[f64]) -> Result<(), HeadError> {
    if labels.len() != probs.len() {
        return fail(format!("{} labels for {} probabilities", labels.len(), probs.len()));
    }
    Ok(())
}

/// `argmax(probs)`, i.e. `min(labels, key=lambda l: (-probs[l], l))`: the most
/// probable label, ties to the smaller label. `str` order is byte order, which
/// for UTF-8 is Python's code-point order.
pub fn argmax<S: AsRef<str>>(labels: &[S], probs: &[f64]) -> Result<usize, HeadError> {
    check_pairs(labels, probs)?;
    if labels.is_empty() {
        return fail("min() iterable argument is empty");
    }
    let mut best = 0;
    for i in 1..labels.len() {
        let (key, best_key) = (-probs[i], -probs[best]);
        let less = if key == best_key { labels[i].as_ref() < labels[best].as_ref() } else { key < best_key };
        if less {
            best = i;
        }
    }
    Ok(best)
}

/// Python 3.12's `sum()` of floats (start `0`): Neumaier-compensated.
pub fn py_sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut xs = xs.into_iter();
    // The int start meets the first float as `0.0 + x`, which turns -0.0 into 0.0.
    let Some(first) = xs.next() else { return 0.0 };
    let (mut sum, mut c) = (0.0 + first, 0.0f64);
    for x in xs {
        let t = sum + x;
        c += if sum.abs() >= x.abs() { (sum - t) + x } else { (x - t) + sum };
        sum = t;
    }
    // CPython skips a zero or non-finite compensation.
    if c != 0.0 && c.is_finite() {
        sum += c;
    }
    sum
}

/// Python's `round(x, 2)`. Rust's fixed-precision formatting is exact with
/// ties to even, as is CPython's `_Py_dg_dtoa` mode 3.
fn round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.2}").parse().unwrap_or(x) // always parses
}

/// `validate_temperature`.
fn validate_temperature(t: f64) -> Result<f64, HeadError> {
    if !t.is_finite() || t <= 0.0 {
        return fail("Temperature must be finite and positive");
    }
    Ok(t)
}

/// `scale_probabilities(probs, temperature)`, with `probs` given as its labels
/// and values in dict order.
pub fn scale_probabilities<S: AsRef<str>>(
    labels: &[S],
    probs: &[f64],
    temperature: f64,
) -> Result<Vec<f64>, HeadError> {
    check_pairs(labels, probs)?;
    let temperature = validate_temperature(temperature)?;
    if probs.is_empty() || probs.iter().any(|&p| !p.is_finite() || !(0.0..=1.0).contains(&p)) {
        return fail("Invalid probability distribution");
    }
    if (py_sum(probs.iter().copied()) - 1.0).abs() > 0.001 {
        return fail("Probabilities must sum to one");
    }
    if temperature == 1.0 {
        return Ok(probs.to_vec());
    }
    let logs: Vec<f64> = probs.iter().map(|&p| if p != 0.0 { p.ln() } else { f64::NEG_INFINITY }).collect();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logs.iter().map(|&l| ((l - maximum) / temperature).exp()).collect();
    let total = py_sum(weights.iter().copied());
    let result: Vec<f64> = weights.iter().map(|&w| w / total).collect();
    if argmax(labels, &result)? != argmax(labels, probs)? {
        return fail("Floating-point temperature scaling changed argmax");
    }
    Ok(result)
}

/// One field after the head.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldHead {
    /// Calibrated probabilities in option order: the response's `probabilities`.
    pub probabilities: Vec<f64>,
    /// Index, in option order, of the response's `decision`.
    pub decision: usize,
}

/// Softmax and calibration for every field; fails where `scale_result` would.
/// `logits` holds each field's candidate logits in option order.
pub fn fields(prepared: &Prepared, logits: &[Vec<f32>]) -> Result<Vec<FieldHead>, HeadError> {
    let options = &prepared.compiled.options;
    if logits.len() != options.len() {
        return fail(format!("{} logit rows for {} fields", logits.len(), options.len()));
    }
    let mut heads = Vec::with_capacity(options.len());
    for (field, (opts, row)) in prepared.compiled.fields.iter().zip(options.iter().zip(logits)) {
        if row.len() != opts.len() {
            return fail(format!("{} logits for the {} options of {field:?}", row.len(), opts.len()));
        }
        let labels: Vec<&str> = opts.iter().map(|(_, value, _)| value.as_str()).collect();
        // `.tolist()` widens each f32 to a Python float.
        let probs: Vec<f64> = softmax(row).into_iter().map(f64::from).collect();
        let probabilities = scale_probabilities(&labels, &probs, TEMPERATURE)?;
        let decision = argmax(&labels, &probabilities)?;
        heads.push(FieldHead { probabilities, decision });
    }
    Ok(heads)
}

/// The top-2 margin of one field's candidate logits: largest minus second
/// largest. Infinite with a single candidate; NaN with none, or if any logit
/// is NaN.
pub fn top2_margin(logits: &[f32]) -> f64 {
    if logits.is_empty() || logits.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    if logits.len() == 1 {
        return f64::INFINITY;
    }
    let mut sorted: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[0] - sorted[1]
}

/// A Python float as JSON, written as `json.dumps` writes it. inf and NaN,
/// which Python would print as `Infinity`/`NaN`, are refused.
fn float(x: f64) -> Result<Value, HeadError> {
    let repr = pyfloat::repr(x);
    Number::from_str(&repr)
        .map(Value::Number)
        .map_err(|_| HeadError(format!("the response would carry {repr}, which JSON cannot (the reference emits it)")))
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Choice => "choice",
        Kind::Score => "score",
        Kind::Noul => "noul",
    }
}

/// A JSON object with its keys in the order given.
fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Object(entries.into_iter().map(|(key, value)| (key.to_owned(), value)).collect())
}

/// InternLM's `DecisionEngine.predict` from the logits on. `logits` holds
/// each field's candidate logits in option order; `input_tokens` is the full
/// prompt length (`prepared.ids.len()`, although the forward pass stops at
/// the last answer row); `inference_ms` is the forward time.
pub fn respond(
    prepared: &Prepared,
    logits: &[Vec<f32>],
    input_tokens: usize,
    inference_ms: f64,
) -> Result<Value, HeadError> {
    respond_from(prepared, &fields(prepared, logits)?, input_tokens, inference_ms)
}

/// [`respond`] from the output of [`fields`], for a caller that also needs
/// the heads themselves.
pub fn respond_from(
    prepared: &Prepared,
    heads: &[FieldHead],
    input_tokens: usize,
    inference_ms: f64,
) -> Result<Value, HeadError> {
    if heads.len() != prepared.compiled.options.len() {
        return fail(format!("{} heads for {} fields", heads.len(), prepared.compiled.options.len()));
    }
    let mut answers = Map::new();
    for ((question, options), head) in prepared.row.questions.iter().zip(&prepared.compiled.options).zip(heads) {
        let (p, best) = (&head.probabilities, head.decision);
        if p.len() != options.len() || best >= p.len() {
            return fail(format!(
                "a head of {} probabilities deciding #{best} for the {} options of {:?}",
                p.len(),
                options.len(),
                question.name
            ));
        }
        let label = options[best].1.as_str();
        let mut probabilities = Map::new();
        for ((_, value, _), &pi) in options.iter().zip(p) {
            probabilities.insert(value.clone(), float(pi)?);
        }
        // Key order is the reference's: type, probabilities, confidence, the
        // kind's own keys, then `scale_result`'s and `predict`'s additions.
        let mut answer = Map::new();
        answer.insert("type".into(), kind_name(question.kind).into());
        answer.insert("probabilities".into(), Value::Object(probabilities));
        answer.insert("confidence".into(), float(p[best])?);
        match question.kind {
            Kind::Noul => {
                let yes = options
                    .iter()
                    .position(|(_, value, _)| value == "yes")
                    .ok_or_else(|| HeadError(format!("noul field {:?} has no \"yes\" option", question.name)))?;
                answer.insert("noul".into(), float(p[yes])?);
            }
            Kind::Score => {
                let mut terms = Vec::with_capacity(p.len());
                for ((_, value, _), &pi) in options.iter().zip(p) {
                    let x = pyfloat::parse(value)
                        .ok_or_else(|| HeadError(format!("could not convert string to float: {value:?}")))?;
                    terms.push(x * pi);
                }
                answer.insert("score".into(), float(py_sum(terms))?);
                let legend = options.iter().map(|(_, value, text)| (value.clone(), Value::String(text.clone())));
                answer.insert("legend".into(), Value::Object(legend.collect()));
            }
            Kind::Choice => {
                answer.insert("choice".into(), label.into());
            }
        }
        answer.insert("source".into(), "local".into());
        answer.insert("decision".into(), label.into());
        answers.insert(question.name.clone(), Value::Object(answer));
    }
    let count = answers.len();
    Ok(object([
        ("answers", Value::Object(answers)),
        ("usage", object([
            ("input_tokens", input_tokens.into()),
            ("output_tokens", count.into()),
            ("decision_count", count.into()),
        ])),
        ("timing", object([("inference_ms", float(round2(inference_ms))?)])),
        ("calibration", object([("method", "temperature-scaling".into()), ("temperature", float(TEMPERATURE)?)])),
        ("model", MODEL.into()),
        ("backend", BACKEND.into()),
    ]))
}
