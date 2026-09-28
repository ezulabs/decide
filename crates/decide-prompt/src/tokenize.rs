//! Request JSON → token ids, logit rows and candidate ids, as InternLM does it.

use crate::compile::{compile_row, Compiled, DECISION_TOKEN, SYSTEM_PROMPT};
use crate::request::{validate_request, RequestError, Row, ANSWER_SYMBOLS};
use crate::{pyjson, template};
use anyhow::{anyhow, ensure, Result};
use std::path::Path;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{PreTokenizerWrapper, SplitDelimiterBehavior};

#[derive(Debug)]
pub enum PrepareError {
    Request(RequestError),
    Other(String),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::Request(e) => f.write_str(&e.0),
            PrepareError::Other(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for PrepareError {}

#[derive(Debug, Clone)]
pub struct Prepared {
    pub text: String,
    pub ids: Vec<u32>,
    /// Logit row (index of the token before each `<decision>`), one per field.
    pub positions: Vec<usize>,
    /// Candidate symbol token ids, one list per field, in option order.
    pub candidates: Vec<Vec<u32>>,
    pub row: Row,
    pub compiled: Compiled,
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    marker: u32,
    symbol_ids: Vec<u32>,
}

/// The split regex the reference actually runs. transformers 5.14.1's
/// `Qwen2Tokenizer.__init__` (models/qwen2/tokenization_qwen2.py) replaces
/// tokenizer.json's pre-tokenizer with this original Qwen2 regex, which has no
/// `\p{M}`: text with combining marks (Indic and Thai vowel signs, U+0303, emoji
/// VS16) splits, and so tokenizes, differently. Vocab, merges, the NFC normalizer
/// and the added tokens are tokenizer.json's.
const REFERENCE_PRETOKENIZE_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn reference_pre_tokenizer() -> Result<PreTokenizerWrapper> {
    let split = Split::new(
        SplitPattern::Regex(REFERENCE_PRETOKENIZE_REGEX.into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let byte_level = ByteLevel::default().add_prefix_space(false).trim_offsets(true).use_regex(false);
    Ok(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ])))
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("{e}"))?;
        inner.with_pre_tokenizer(Some(reference_pre_tokenizer()?));
        let marker = inner.token_to_id(DECISION_TOKEN).ok_or_else(|| anyhow!("tokenizer lacks {DECISION_TOKEN}"))?;
        let mut symbol_ids = Vec::with_capacity(ANSWER_SYMBOLS.len());
        for c in ANSWER_SYMBOLS.chars() {
            let enc = inner.encode(c.to_string(), false).map_err(|e| anyhow!("{e}"))?;
            ensure!(enc.get_ids().len() == 1, "Candidate symbols must each encode to a single token");
            symbol_ids.push(enc.get_ids()[0]);
        }
        Ok(Self { inner, marker, symbol_ids })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    pub fn prefix_ids(&self) -> Result<Vec<u32>> {
        self.encode(&template::prefix_text())
    }

    pub fn prepare(&self, request_json: &str, max_length: usize) -> Result<Prepared, PrepareError> {
        let request: serde_json::Value = serde_json::from_str(request_json)
            .map_err(|e| PrepareError::Other(format!("request is not valid JSON: {e}")))?;
        if let Some(e) = pyjson::int_digits_error(request_json) {
            return Err(PrepareError::Request(RequestError(e)));
        }
        let row = validate_request(&request).map_err(PrepareError::Request)?;
        let compiled = compile_row(&row).map_err(PrepareError::Request)?;
        let text = template::render(SYSTEM_PROMPT, &compiled.user_text, &compiled.skeleton);
        let ids = self.encode(&text).map_err(|e| PrepareError::Other(e.to_string()))?;
        if ids.len() > max_length {
            return Err(PrepareError::Other(format!(
                "Example has {} tokens, above {max_length}; truncation is forbidden",
                ids.len()
            )));
        }
        let positions: Vec<usize> =
            ids.iter().enumerate().filter(|(_, &t)| t == self.marker).map(|(i, _)| i.wrapping_sub(1)).collect();
        if positions.len() != compiled.fields.len() || positions.contains(&usize::MAX) {
            return Err(PrepareError::Other("Decision marker count or position mismatch".into()));
        }
        let candidates = compiled.options.iter().map(|o| self.symbol_ids[..o.len()].to_vec()).collect();
        Ok(Prepared { text, ids, positions, candidates, row, compiled })
    }
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;
    use serde_json::Value;
    use std::path::PathBuf;

    /// Every part of the pipeline that affects ids is what the reference runs.
    /// The fixture is its `backend_tokenizer.to_str()` less vocab and merges.
    /// The decoder is not compared: decide never decodes.
    #[test]
    fn pipeline_is_the_references() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tok = Tokenizer::from_file(&root.join("models/Intern-Decision-4B/tokenizer.json")).unwrap();
        let ours: Value = serde_json::from_str(&tok.inner.to_string(false).unwrap()).unwrap();
        let path = root.join("tests/fixtures/tokenizer_pipeline.json");
        let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for key in ["normalizer", "pre_tokenizer", "added_tokens", "post_processor", "truncation", "padding"] {
            assert_eq!(ours[key], reference[key], "{key}");
        }
        for (key, value) in reference["model"].as_object().unwrap() {
            assert_eq!(&ours["model"][key], value, "model.{key}");
        }
    }
}
