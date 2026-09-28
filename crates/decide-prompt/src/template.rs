//! The checkpoint's chat template for exactly the messages decide sends:
//! [system, user (text), assistant (skeleton)], no generation prompt, thinking
//! off. Every content is `|trim`med (Python strip) as the Jinja template does.

use crate::compile::{SYSTEM_PROMPT, USER_PREAMBLE};
use crate::pystr::strip;

pub fn render(system: &str, user: &str, assistant: &str) -> String {
    let mut content = strip(assistant).to_string();
    let mut reasoning = String::new();
    if content.contains("</think>") {
        let before = content.split("</think>").next().unwrap().trim_end_matches('\n');
        reasoning = before.rsplit("<think>").next().unwrap().trim_start_matches('\n').to_string();
        content = content.rsplit("</think>").next().unwrap().trim_start_matches('\n').to_string();
    }
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n{}\n</think>\n\n{}<|im_end|>\n",
        strip(system),
        strip(user),
        strip(&reasoning),
        content
    )
}

/// The constant head of every prompt: everything through "## State\n".
pub fn prefix_text() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{USER_PREAMBLE}")
}
