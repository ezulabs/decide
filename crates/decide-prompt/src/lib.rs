//! Request JSON → prompt → token ids, exactly as InternLM's inference.py does it.
pub mod compile;
pub mod pyfloat;
pub mod pyjson;
pub mod pystr;
pub mod request;
pub mod template;
pub mod tokenize;
