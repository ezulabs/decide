//! Intern-Decision-4B text-model constants. The engine is specialised to them;
//! `check_config` refuses a checkpoint whose config differs.

pub const HIDDEN: usize = 2560;
pub const N_LAYERS: usize = 32;
pub const NK: usize = 16;
pub const DK: usize = 128;
pub const NV: usize = 32;
pub const DV: usize = 128;
pub const KEY_DIM: usize = NK * DK; // 2048
pub const VALUE_DIM: usize = NV * DV; // 4096
pub const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM; // 8192
pub const CONV_K: usize = 4;
pub const LIN_IN_N: usize = CONV_DIM + VALUE_DIM + 2 * NV; // 12352
pub const LIN_Z: usize = CONV_DIM; // column of z in P
pub const LIN_B: usize = CONV_DIM + VALUE_DIM; // 12288
pub const LIN_A: usize = LIN_B + NV; // 12320
pub const NQ: usize = 16;
pub const NKV: usize = 4;
pub const HEAD_DIM: usize = 256;
pub const Q_DIM: usize = NQ * HEAD_DIM; // 4096
pub const KV_DIM: usize = NKV * HEAD_DIM; // 1024
pub const ATT_IN_N: usize = 2 * Q_DIM + 2 * KV_DIM; // 10240
pub const ATT_GATE: usize = Q_DIM; // 4096
pub const ATT_K: usize = 2 * Q_DIM; // 8192
pub const ATT_V: usize = 2 * Q_DIM + KV_DIM; // 9216
pub const ROT_DIM: usize = 64;
pub const ROPE_THETA: f32 = 1e7;
pub const INTER: usize = 9216;
pub const MLP_IN_N: usize = 2 * INTER; // 18432
pub const SWIGLU_BLOCK: usize = 16;
pub const VOCAB: usize = 248_320;
pub const EPS: f32 = 1e-6;
pub const TEMPERATURE: f64 = 1.99241824;
pub const MAX_TOKENS: usize = 8192;
pub const CHUNK: usize = 512;
pub const MAX_CANDIDATES: usize = 62;

pub fn is_full_attention(layer: usize) -> bool {
    (layer + 1) % 4 == 0
}

/// RoPE inverse frequencies, computed in f32 as HF does.
pub fn inv_freq() -> [f32; ROT_DIM / 2] {
    std::array::from_fn(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / ROT_DIM as f32))
}

pub fn check_config(config: &serde_json::Value) -> anyhow::Result<()> {
    let t = &config["text_config"];
    let expect: [(&str, f64); 14] = [
        ("hidden_size", HIDDEN as f64), ("num_hidden_layers", N_LAYERS as f64),
        ("linear_num_key_heads", NK as f64), ("linear_key_head_dim", DK as f64),
        ("linear_num_value_heads", NV as f64), ("linear_value_head_dim", DV as f64),
        ("linear_conv_kernel_dim", CONV_K as f64), ("num_attention_heads", NQ as f64),
        ("num_key_value_heads", NKV as f64), ("head_dim", HEAD_DIM as f64),
        ("intermediate_size", INTER as f64), ("vocab_size", VOCAB as f64),
        ("full_attention_interval", 4.0), ("rms_norm_eps", 1e-6),
    ];
    for (key, want) in expect {
        let got = t[key].as_f64().ok_or_else(|| anyhow::anyhow!("text_config.{key} missing"))?;
        anyhow::ensure!(got == want, "text_config.{key} = {got}; decide is built for {want}");
    }
    let rope = &t["rope_parameters"];
    anyhow::ensure!(rope["partial_rotary_factor"].as_f64() == Some(0.25), "partial_rotary_factor");
    anyhow::ensure!(rope["rope_theta"].as_f64() == Some(1e7), "rope_theta");
    anyhow::ensure!(rope["rope_type"].as_str() == Some("default"), "rope_type {}", rope["rope_type"]);
    anyhow::ensure!(t["attn_output_gate"].as_bool() == Some(true), "attn_output_gate");
    anyhow::ensure!(t["hidden_act"].as_str() == Some("silu"), "hidden_act {}", t["hidden_act"]);
    anyhow::ensure!(config["tie_word_embeddings"].as_bool() == Some(true), "tie_word_embeddings");
    // Optional lists; when present they must agree with `is_full_attention`
    // and with every layer having an MLP.
    if !t["layer_types"].is_null() {
        let kinds = t["layer_types"].as_array().ok_or_else(|| anyhow::anyhow!("layer_types is not a list"))?;
        anyhow::ensure!(kinds.len() == N_LAYERS, "layer_types has {} entries, not {N_LAYERS}", kinds.len());
        for (layer, kind) in kinds.iter().enumerate() {
            let want = if is_full_attention(layer) { "full_attention" } else { "linear_attention" };
            anyhow::ensure!(kind.as_str() == Some(want), "layer_types[{layer}] = {kind}; decide is built for {want}");
        }
    }
    if !t["mlp_only_layers"].is_null() {
        let mlp_only = t["mlp_only_layers"].as_array();
        anyhow::ensure!(mlp_only.is_some_and(|l| l.is_empty()), "mlp_only_layers {}", t["mlp_only_layers"]);
    }
    Ok(())
}
