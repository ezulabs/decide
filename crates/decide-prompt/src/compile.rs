//! InternLM's `compile_row`: the user message and the JSON answer skeleton.

use crate::request::{fail, options, RequestError, Row, ANSWER_SYMBOLS, MAX_OPTIONS};
use crate::{pyjson, pystr};
use serde_json::Value;

pub const SYSTEM_PROMPT: &str = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.";
pub const DECISION_TOKEN: &str = "<decision>";
pub const USER_PREAMBLE: &str = "Return one answer for every field using the supplied answer symbols.\n\n## State\n";

#[derive(Debug, Clone)]
pub struct Compiled {
    pub user_text: String,
    pub skeleton: String,
    pub fields: Vec<String>,
    /// Per field, in order: (symbol, option value, description).
    pub options: Vec<Vec<(char, String, String)>>,
}

pub fn compile_row(row: &Row) -> Result<Compiled, RequestError> {
    let mut schema = Vec::new();
    let mut fields = Vec::new();
    let mut per_field = Vec::new();
    for q in &row.questions {
        let opts = options(q)?;
        if opts.len() > MAX_OPTIONS {
            return fail(format!("At most {MAX_OPTIONS} single-token answer symbols are supported"));
        }
        let instructions = match &q.instructions {
            None => String::new(),
            Some(v) => pystr::str_of(v).map_err(RequestError)?,
        };
        schema.push(format!("{}: {}", q.name, instructions));
        let mut entries = Vec::with_capacity(opts.len());
        for (symbol, (value, description)) in ANSWER_SYMBOLS.chars().zip(opts) {
            schema.push(format!("    {symbol} = {value}: {description}"));
            entries.push((symbol, value, description));
        }
        fields.push(q.name.clone());
        per_field.push(entries);
    }
    let state = pyjson::dumps(&row.state, 2);
    let user_text = format!("{USER_PREAMBLE}{state}\n## Decision schema\n{}", schema.join("\n"));
    if user_text.contains(DECISION_TOKEN) {
        return fail("Reserved decision marker appears in input evidence");
    }
    let skeleton: serde_json::Map<String, Value> =
        fields.iter().map(|f| (f.clone(), Value::String(DECISION_TOKEN.into()))).collect();
    Ok(Compiled { user_text, skeleton: pyjson::dumps(&Value::Object(skeleton), 4), fields, options: per_field })
}
