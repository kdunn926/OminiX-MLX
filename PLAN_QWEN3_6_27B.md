# Plan: Extend Qwen3.6-MLX to Support Qwen3.6-27B Dense Model

## Overview
The Qwen3.6-35B-A3B-mlx crate currently supports only the MoE (Mixture of Experts) variant. The Qwen3.6-27B model is a dense version of the same architecture, sharing the same:
- Hidden size: 5120
- Intermediate size: 17408
- Head dimension: 256
- Number of attention heads: 24
- Number of key-value heads: 4
- Number of layers: 64
- Layer type pattern: [linear_attention, linear_attention, linear_attention, full_attention] × 16
- RoPE configuration: theta=10M, partial_rotary_factor=0.25, mrope_section=[11,11,10]
- Vocab size: 248320
- Quantization: 4-bit affine, group_size=64

The key difference is that the 27B model replaces the MoE layers with standard MLP layers (gate_proj, up_proj, down_proj) without expert routing.

## Changes Required

### 1. Model Architecture Detection
Modify `src/model.rs` to detect whether the model is MoE or dense based on config.json:
- Check for `model_type` field: `"qwen3_5_moe"` vs `"qwen3_5"`
- Alternatively, check for presence of MoE-specific fields like `num_experts`

### 2. Conditional Layer Construction
In `src/model.rs::load_model()`:
- When loading each layer, check if model is MoE or dense
- For MoE: use existing MoeBlock construction
- For Dense: construct standard MLP with:
  - gate_proj: QuantizedLinear
  - up_proj: QuantizedLinear  
  - down_proj: QuantizedLinear
  - No routing gate, no shared expert, no expert switching logic

### 3. Attention Logic Unchanged
The attention mechanism (GatedAttention for full_attention, GatedDeltaNet for linear_attention) remains identical between MoE and dense variants.

### 4. Weight Loading Adaptation
Update weight loading functions to handle both layouts:
- MoE expects keys like: `.mlp.gate`, `.mlp.switch_mlp.*`, `.mlp.shared_expert.*`, `.mlp.shared_expert_gate`
- Dense expects keys like: `.mlp.gate_proj`, `.mlp.up_proj`, `.mlp.down_proj`

### 5. Configuration Updates
Update `src/config.rs` TextConfig to optionally include MoE fields with sensible defaults:
- Make `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` optional
- Default to 0/None for dense models

### 6. Testing Strategy
1. Verify existing 35B MoE model still works
2. Test loading 27B dense model from `/home/copilot/repos/mac/OminiX-MLX/models/Qwen3.6-27B-4bit/`
3. Run inference tests with both models to ensure parity in architecture-expected behaviors
4. Validate token generation quality with sample prompts

## Implementation Steps

### Step 1: Update Config Detection
```rust
// In src/config.rs
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    // ... existing fields ...
    
    // MoE config (optional for dense models)
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<i32>,
    
    // Helper method
    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 0 
            && self.num_experts_per_tok.unwrap_or(0) > 0
    }
}
```

### Step 2: Refactor MLP Loading
```rust
// In src/model.rs, replace MoeBlock construction with:
let mlp = if tc.is_moe() {
    // Existing MoeBlock construction
    MoeBlock { ... }
} else {
    // Dense MLP: gate_proj + up_proj + down_proj
    let gate_proj = MaybeQuantized::Quantized(make_quantized_linear(
        &weights,
        &format!("{}.mlp.gate_proj", layer_prefix),
        group_size,
        bits,
    )?);
    
    let up_proj = MaybeQuantized::Quantized(make_quantized_linear(
        &weights,
        &format!("{}.mlp.up_proj", layer_prefix),
        group_size,
        bits,
    )?);
    
    let down_proj = MaybeQuantized::Quantized(make_quantized_linear(
        &weights,
        &format!("{}.mlp.down_proj", layer_prefix),
        group_size,
        bits,
    )?;
    
    MlpBlock { gate_proj, up_proj, down_proj }
};

// Update TransformerBlock to use mlp.forward() instead of moe.forward()
```

### Step 3: Define Dense MLP Structure
```rust
// In src/model.rs
pub struct MlpBlock {
    pub gate_proj: MaybeQuantized<nn::Linear>,
    pub up_proj: MaybeQuantized<nn::Linear>,
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl MlpBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Standard SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_silu = gate.silu()?;
        let intermediate = gate_silu.multiply(&up)?;
        self.down_proj.forward(&intermediate)
    }
}
```

### Step 4: Update Weight Mapping Logic
Add detection for dense vs MoE weight keys in loading functions.

## Files to Modify
- `src/config.rs` - Add optional MoE fields and detection
- `src/model.rs` - Main logic for conditional layer construction
- Possibly `src/lib.rs` - Update feature flags/docs if needed

## Verification
After implementation:
1. `cargo build --release` should succeed for both models
2. Running the example with 27B model should generate coherent text
3. Compare perplexity scores between HF transformers and MLX-RS implementations
4. Validate that layer counts and types match config.json expectations

## Estimated Effort
- 2-3 hours for implementation
- 1 hour for testing
- 1 hour for documentation updates

This approach maintains backward compatibility while extending support to the dense variant through conditional compilation based on model configuration.