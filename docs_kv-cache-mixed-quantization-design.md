# KV Cache Mixed Quantization Design (K/V Split, Q4/Q8/TurboQuant)

## Summary

This document proposes a repository-grounded design for adding **KV-cache quantization** to OminiX-MLX, with support for:

- `none`
- `q8`
- `q4`
- `turbo`

and importantly, **independent quantization policies for keys and values**:

- `K != V` is allowed
- recommended default optimized profile: **K=Q8, V=Q4**

The design is grounded primarily in the existing GPT-SoVITS autoregressive code paths and the MLX Rust quantization utilities already present in the repository.

## Repository grounding

### Current cache/update flow

The clearest existing KV cache path is in GPT-SoVITS Python:

- `gpt-sovits-mlx/python/models/cache.py`
  - Defines `KVCache` and `ConcatKVCache`
  - `KVCache.update(keys, values)` appends into step-allocated buffers
- `gpt-sovits-mlx/python/models/attention.py`
  - Computes `q`, `k`, `v`
  - Applies RoPE to `q` and `k`
  - Calls `cache.update(k, v)`
  - Runs attention over full cached `k`, `v`
- `gpt-sovits-mlx/python/models/gpt.py`
  - Builds one cache per layer with `create_caches(...)`

Relevant Rust-side generation plumbing also exists:

- `gpt-sovits-mlx/src/inference.rs`
  - Uses generic `KeyValueCache + Default`
  - Creates one cache per layer for autoregressive generation
- `gpt-sovits-mlx/src/cache.rs`
  - Re-exports cache types from `mlx-rs-core`

### Existing quantization utilities in repo

The repository already contains quantization primitives and patterns:

- `mlx-rs/src/ops/quantization.rs`
  - exposes `quantize`, `dequantize`, `quantized_matmul`
- `mlx-rs/src/nn/quantized.rs`
  - builder/config pattern for quantized modules
- `moxin-vlm-mlx/src/lib.rs`
- `moxin-vlm-mlx/src/projector.rs`
- `moxin-vlm-mlx/src/vision.rs`

These show:
- group-size-driven quantization config already exists in the codebase
- the repo is already comfortable with optional quantized/non-quantized execution paths

## Why mixed K/V quantization

Keys and values do not have equal sensitivity.

### Keys (K)
Keys directly influence:
- attention score computation
- softmax ranking
- next-token selection sensitivity

Errors in `K` tend to perturb token choice more aggressively.

### Values (V)
Values primarily affect:
- weighted aggregation after softmax
- hidden-state composition

`V` is often more tolerant of aggressive quantization than `K`.

### Practical implication

A mixed profile is desirable:

- conservative:
  - `K=none`, `V=q8`
- balanced:
  - `K=q8`, `V=q4`
- aggressive:
  - `K=q4`, `V=q4`
- adaptive:
  - `K=turbo`, `V=turbo` with different internal policies

**Recommended default performance profile:**
- `K=q8`
- `V=q4`

## Design goals

1. Preserve current behavior when quantization is disabled
2. Allow **independent K and V policies**
3. Make minimal changes to attention call sites
4. Support a simple implementation first:
   - quantize on write
   - dequantize on read
5. Leave room for later fused decode/attention kernels
6. Keep config surfaces consistent between Python and Rust paths where possible

## Proposed public API

## 1. Quantization mode enum

Use a common conceptual enum:

```rust
enum KvQuantMode {
    None,
    Q8,
    Q4,
    Turbo,
}
```

Python equivalent:

```python
from enum import Enum

class KvQuantMode(str, Enum):
    NONE = "none"
    Q8 = "q8"
    Q4 = "q4"
    TURBO = "turbo"
```

## 2. K/V split config

### Rust shape

```rust
#[derive(Debug, Clone)]
pub struct KvCacheQuantConfig {
    pub k_mode: KvQuantMode,
    pub v_mode: KvQuantMode,
    pub group_size: i32,
    pub recent_window_k: usize,
    pub recent_window_v: usize,
    pub residual_length_k: usize,
    pub residual_length_v: usize,
}
```

### Python shape

```python
from dataclasses import dataclass

@dataclass
class KvCacheQuantConfig:
    k_mode: str = "none"
    v_mode: str = "none"
    group_size: int = 64
    recent_window_k: int = 256
    recent_window_v: int = 128
    residual_length_k: int = 0
    residual_length_v: int = 0
```

## 3. User-facing flags

Suggested CLI/API flags:

- `--kv-cache-quantization none|q8|q4|turbo`
  - shorthand applying to both K and V
- `--kv-cache-k-quantization none|q8|q4|turbo`
- `--kv-cache-v-quantization none|q8|q4|turbo`
- `--kv-cache-group-size 64`
- `--kv-cache-recent-window-k 256`
- `--kv-cache-recent-window-v 128`
- `--kv-cache-residual-length-k 0`
- `--kv-cache-residual-length-v 0`

Override precedence:
1. explicit `k/v` options
2. shared `kv-cache-quantization`
3. default `none`

## Proposed internal architecture

## 1. Split the cache into `key_cache` and `value_cache`

Instead of assuming one symmetric storage format, represent each layer’s cache as:

```rust
pub struct LayerKvCache {
    pub key_cache: CacheTensorStore,
    pub value_cache: CacheTensorStore,
}
```

Python equivalent:

```python
class LayerKVCache:
    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache
```

This is the most important structural change for mixed quantization.

## 2. Introduce a storage abstraction

### Common interface

```rust
pub trait CacheTensorCodec {
    fn append(&mut self, x: &Array) -> Result<()>;
    fn materialize(&self) -> Result<Array>;
    fn seq_len(&self) -> usize;
    fn reset(&mut self);
}
```

Python equivalent:

```python
class CacheTensorCodec:
    def append(self, x): ...
    def materialize(self): ...
    def seq_len(self): ...
    def reset(self): ...
```

Implementations:

- `FpCacheTensorCodec`
- `Q8CacheTensorCodec`
- `Q4CacheTensorCodec`
- `TurboCacheTensorCodec`

## 3. Quantize on append, decode on read

The current attention flow is effectively:

1. compute `q`, `k`, `v`
2. apply RoPE to `q` and `k`
3. append `k`, `v` to cache
4. read full cache for attention

The least disruptive implementation is:

- quantize `k` and `v` **when appended**
- dequantize them **when attention needs them**

This keeps the existing attention kernel intact.

### Important detail
**RoPE should be applied before K is quantized.**

That matches current logic in:
- `gpt-sovits-mlx/python/models/attention.py`

So the order remains:

1. project Q/K/V
2. reshape
3. apply RoPE to Q and K
4. append K/V using cache codec
5. materialize K/V for attention

## Proposed rough implementation

## 1. Python-first design sketch

### Current code shape

Today in `gpt-sovits-mlx/python/models/attention.py` the logic is roughly:

```python
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

q = ...
k = ...
v = ...

q = self.rope(q, offset=offset)
k = self.rope(k, offset=offset)

if cache is not None:
    k, v = cache.update(k, v)
```

### Proposed replacement

```python
if cache is not None:
    cache.append(k, v)
    k = cache.materialize_keys()
    v = cache.materialize_values()
```

That avoids requiring `update()` to keep returning raw tensors from a single homogeneous store.

### Proposed Python cache classes

```python
class QuantizedTensorStore:
    def append(self, x: mx.array) -> None:
        ...
    def materialize(self) -> mx.array:
        ...
    def reset(self) -> None:
        ...
    @property
    def seq_len(self) -> int:
        ...
```

```python
class LayerKVCache:
    def __init__(self, key_store, value_store):
        self.key_store = key_store
        self.value_store = value_store

    @property
    def seq_len(self) -> int:
        return self.key_store.seq_len

    def append(self, k, v):
        self.key_store.append(k)
        self.value_store.append(v)

    def materialize_keys(self):
        return self.key_store.materialize()

    def materialize_values(self):
        return self.value_store.materialize()

    def reset(self):
        self.key_store.reset()
        self.value_store.reset()
```

## 2. Rust-side design sketch

Rust can mirror the same pattern at the `KeyValueCache` level.

### Current usage

In `gpt-sovits-mlx/src/inference.rs`, generation uses:

```rust
let mut caches: Vec<Option<C>> = (0..num_layers).map(|_| None).collect();
```

with `C: KeyValueCache + Default`.

### Proposed direction

Either:

#### Option A: extend the current trait
Add split append/materialize methods.

```rust
pub trait KeyValueCache {
    fn append(&mut self, k: &Array, v: &Array) -> Result<()>;
    fn keys(&self) -> Result<Array>;
    fn values(&self) -> Result<Array>;
    fn seq_len(&self) -> usize;
    fn reset(&mut self);
}
```

#### Option B: add a new trait for quantized-capable caches
This is safer if existing users rely on the current trait contract.

```rust
pub trait QuantizableKeyValueCache {
    fn append(&mut self, k: &Array, v: &Array) -> Result<()>;
    fn materialize_keys(&self) -> Result<Array>;
    fn materialize_values(&self) -> Result<Array>;
    fn seq_len(&self) -> usize;
    fn reset(&mut self);
}
```

Recommendation:
- use **Option B** first to avoid breaking unrelated crates

## Q8 design

## Storage
- store quantized tensor payload for each appended segment
- store per-group scales
- optionally store biases/zero-points
- group along the head-dimension (`head_dim`) or flattened last dimension

## Suggested default
- group size: `64`

## Rationale
This aligns well with the repo’s existing quantization config conventions:
- `qwen3.5-35B-mlx/src/config.rs`
- `mlx-rs/src/nn/quantized.rs`
- `mlx-rs/src/ops/quantization.rs`

## Rough codec structure

```rust
pub struct Q8CacheTensorCodec {
    packed: Vec<Array>,
    scales: Vec<Array>,
    biases: Vec<Array>,
    seq_len: usize,
    group_size: i32,
}
```

Python analogue:

```python
class Q8CacheTensorStore:
    def __init__(self, group_size=64):
        self.packed = []
        self.scales = []
        self.biases = []
        self._seq_len = 0
```

## Q4 design

Q4 is similar to Q8 but more aggressive.

## Storage
- packed 4-bit values
- per-group scales
- optional biases/zero-points
- either:
  - manual nibble packing, or
  - rely on MLX quantization packing format where possible

## Important note
If MLX quantization ops already expose packing/dequantization in a reusable way, prefer reusing them rather than implementing custom bit-packing first.

Repository evidence:
- `mlx-rs/src/ops/quantization.rs` already exposes:
  - `quantize`
  - `dequantize`
  - `quantized_matmul`

That suggests a Rust implementation can likely leverage MLX-native quantization rather than inventing a parallel format.

## TurboQuant design

TurboQuant should not just mean “another bits mode”.
It should mean a **policy** optimized for long-context decoding.

## Recommended TurboQuant behavior

### For K
- keep most recent `recent_window_k` tokens in higher precision
  - `none` or `q8`
- compress older tokens more aggressively
  - often `q4`

### For V
- keep a smaller recent window
- quantize older values more aggressively than keys

### Suggested default policy
- `K`: recent window in `q8`, older in `q4`
- `V`: recent window in `q4`, older in `q4`

or, if quality needs more protection:

- `K`: recent window in `none`, older in `q8`
- `V`: recent window in `q8`, older in `q4`

## TurboQuant codec idea

```rust
pub struct TurboCacheTensorCodec {
    recent_fp: Option<Array>,
    recent_q8: Option<QuantizedBlockStore>,
    older_q4: Option<QuantizedBlockStore>,
    seq_len: usize,
    recent_window: usize,
}
```

The implementation can start simpler:

1. append into an uncompressed recent buffer
2. once recent buffer exceeds threshold
3. quantize the oldest chunk into long-term storage
4. materialize by concatenating:
   - older dequantized blocks
   - recent raw/less-quantized blocks

## Mixed K/V policy matrix

Recommended supported combinations:

| K mode | V mode | Notes |
|---|---|---|
| none | none | baseline |
| q8 | q8 | safest quantized mode |
| q8 | q4 | best default optimized profile |
| none | q4 | useful for quality-sensitive scoring |
| q4 | q4 | max compression |
| turbo | turbo | adaptive mode |
| turbo | q4 | acceptable advanced mode |
| q8 | turbo | acceptable advanced mode |

## Attention integration points

## Python

### File
- `gpt-sovits-mlx/python/models/attention.py`

### Current integration point
After RoPE and before attention execution.

### Proposed patch shape
Replace:

```python
if cache is not None:
    k, v = cache.update(k, v)
```

with:

```python
if cache is not None:
    cache.append(k, v)
    k = cache.materialize_keys()
    v = cache.materialize_values()
```

### Why this is the right seam
This is the smallest change that:
- preserves current attention call structure
- allows K and V to use different backing stores
- supports incremental migration from raw to quantized cache

## GPT model cache creation point

### File
- `gpt-sovits-mlx/python/models/gpt.py`

### Current function
- `create_caches(dtype: mx.Dtype = mx.float32) -> List[KVCache]`

### Proposed expansion
Allow a quantization config parameter:

```python
def create_caches(
    self,
    dtype: mx.Dtype = mx.float32,
    quant_config: Optional[KvCacheQuantConfig] = None,
) -> List[LayerKVCache]:
    ...
```

## Rust generation integration point

### Files
- `gpt-sovits-mlx/src/inference.rs`
- possibly `mlx-rs-core::cache` trait definitions

### Current behavior
The autoregressive loop creates one cache per layer and reuses it each step.

### Proposed change
Allow cache construction with config:

```rust
let cache_cfg = KvCacheQuantConfig {
    k_mode: KvQuantMode::Q8,
    v_mode: KvQuantMode::Q4,
    group_size: 64,
    recent_window_k: 256,
    recent_window_v: 128,
    residual_length_k: 0,
    residual_length_v: 0,
};
```

Then instantiate a quantized-capable cache type per layer.

## Rough implementation stages

## Stage 1: refactor for split K/V storage
- Introduce `LayerKVCache`
- Replace single homogeneous cache assumptions
- Keep both stores unquantized initially

## Stage 2: add Q8 support
- Implement `Q8CacheTensorStore`
- Quantize on append
- Dequantize on materialize
- Validate parity vs baseline

## Stage 3: add Q4 support
- Implement `Q4CacheTensorStore`
- Add error tolerance tests
- Benchmark memory and speed

## Stage 4: mixed K/V configuration
- Add config surface
- Support `K=q8, V=q4`
- Add combination tests

## Stage 5: TurboQuant
- Add recent-window policy
- Requantize older blocks
- Benchmark long-context decode

## Rough pseudocode

## Python append/materialize

```python
class Q8CacheTensorStore:
    def __init__(self, group_size=64):
        self.q = []
        self.scales = []
        self.biases = []
        self._seq_len = 0
        self.group_size = group_size

    def append(self, x):
        q, scales, biases = quantize_tensor(x, bits=8, group_size=self.group_size)
        self.q.append(q)
        self.scales.append(scales)
        self.biases.append(biases)
        self._seq_len += x.shape[2]

    def materialize(self):
        parts = [
            dequantize_tensor(q, s, b, bits=8, group_size=self.group_size)
            for q, s, b in zip(self.q, self.scales, self.biases)
        ]
        return mx.concatenate(parts, axis=2) if parts else None
```

## Layer cache wrapper

```python
class LayerKVCache:
    def __init__(self, key_store, value_store):
        self.key_store = key_store
        self.value_store = value_store

    def append(self, k, v):
        self.key_store.append(k)
        self.value_store.append(v)

    def materialize_keys(self):
        return self.key_store.materialize()

    def materialize_values(self):
        return self.value_store.materialize()

    @property
    def seq_len(self):
        return self.key_store.seq_len
```

## Rust sketch

```rust
pub struct LayerKvCache<KC, VC> {
    pub key_cache: KC,
    pub value_cache: VC,
}

impl<KC, VC> LayerKvCache<KC, VC>
where
    KC: CacheTensorCodec,
    VC: CacheTensorCodec,
{
    pub fn append(&mut self, k: &Array, v: &Array) -> Result<()> {
        self.key_cache.append(k)?;
        self.value_cache.append(v)?;
        Ok(())
    }

    pub fn materialize_keys(&self) -> Result<Array> {
        self.key_cache.materialize()
    }

    pub fn materialize_values(&self) -> Result<Array> {
        self.value_cache.materialize()
    }
}
```

## Validation strategy

## Unit tests

### Quantization round-trip tests
- Q8 round-trip max error under threshold
- Q4 round-trip max error under threshold
- mixed shapes:
  - `[1, 8, 1, 64]`
  - `[1, 8, 32, 64]`
  - `[1, 16, 128, 128]`

### Packing/storage tests
- append multiple segments
- verify sequence length
- verify materialized concat order
- verify reset clears state

### Mixed K/V tests
- `K=q8, V=q4`
- `K=none, V=q4`
- `K=q4, V=q4`
- `K=turbo, V=q4`

## Integration tests

### Attention equivalence
For `gpt-sovits-mlx/python/models/attention.py`:
- compare baseline logits/output with:
  - Q8/Q8
  - Q8/Q4
  - Q4/Q4

### Generation regression
For autoregressive generation:
- fixed prompt
- deterministic sampling / greedy decode
- compare:
  - produced token sequence
  - divergence point
  - logits drift threshold

### Cache behavior
- prefill then single-token decode
- repeated decode updates
- long context append beyond one allocation step/window

## Benchmark plan

The repo already has useful benchmark precedent in:
- `gpt-sovits-mlx/scripts/benchmark.py`
- `gpt-sovits-mlx/docs/PERFORMANCE_ANALYSIS.md`

Extend benchmarks to measure:

### Memory
- peak cache memory
- bytes/token for K and V

### Latency
- prefill latency
- decode latency per token
- append-only time
- materialize/dequantize time

### Quality
- logits error
- decode divergence rate
- if available, task-specific generation quality impact

### Benchmark matrix
- baseline `none/none`
- `q8/q8`
- `q8/q4`
- `q4/q4`
- `turbo/turbo`

## Risks and tradeoffs

## 1. Dequantize-on-read may erase some speed gains
The first version should optimize for correctness and maintainability.

Mitigation:
- stage later fused attention/dequantization work after correctness is proven

## 2. Fragmented block storage may add concat overhead
If each append stores a tiny quantized segment, materialization may become costly.

Mitigation:
- store in chunked blocks
- periodically coalesce blocks
- use larger append windows

## 3. TurboQuant policy complexity
TurboQuant can become overdesigned quickly.

Mitigation:
- define TurboQuant initially as:
  - recent window + older compressed region
- defer outlier-aware and residual refinements

## 4. Trait breakage in Rust cache abstractions
Changing existing cache traits may ripple through crates.

Mitigation:
- add a new trait or new cache type instead of mutating the old one first

## Recommended defaults

### Initial safe defaults
- `k_mode = q8`
- `v_mode = q8`
- `group_size = 64`

### Recommended optimized defaults
- `k_mode = q8`
- `v_mode = q4`
- `group_size = 64`

### Turbo defaults
- `k_mode = turbo`
- `v_mode = turbo`
- `recent_window_k = 256`
- `recent_window_v = 128`
- older region compressed to Q4

## File-by-file implementation checklist

## Python path

### `gpt-sovits-mlx/python/models/cache.py`
- [ ] Add split `LayerKVCache`
- [ ] Add `FpCacheTensorStore`
- [ ] Add `Q8CacheTensorStore`
- [ ] Add `Q4CacheTensorStore`
- [ ] Add `TurboCacheTensorStore`
- [ ] Add K/V-specific config handling
- [ ] Preserve existing `KVCache` during migration if needed

### `gpt-sovits-mlx/python/models/attention.py`
- [ ] Replace `cache.update(k, v)` assumption with append/materialize split
- [ ] Preserve current no-cache path
- [ ] Preserve RoPE-before-cache behavior

### `gpt-sovits-mlx/python/models/gpt.py`
- [ ] Extend `create_caches(...)` to accept quant config
- [ ] Create per-layer split caches
- [ ] Default to unquantized behavior

### tests / benchmarks
- [ ] Add round-trip quantization tests
- [ ] Add cache append/materialize tests
- [ ] Add generation regression tests
- [ ] Add mixed `K/V` benchmark cases

## Rust path

### `mlx-rs-core` / cache abstraction
- [ ] Inspect current `KeyValueCache` trait and concrete types
- [ ] Decide between extending trait vs adding parallel quantized trait
- [ ] Add split K/V cache representation

### `gpt-sovits-mlx/src/cache.rs`
- [ ] Re-export new quantized cache types if implemented in core

### `gpt-sovits-mlx/src/inference.rs`
- [ ] Thread quantization config into cache construction
- [ ] Keep current default path unchanged

### `mlx-rs/src/ops/quantization.rs`
- [ ] Reuse existing quantize/dequantize ops where practical
- [ ] Avoid custom pack/unpack unless MLX-native path is insufficient

## Suggested first milestone

Implement **Python-side mixed K/V cache quantization for GPT-SoVITS** with:

- `none`
- `q8`
- `q4`

and config:

- `K=q8`
- `V=q4`

This gives:
- a concrete proof of design
- the smallest path to user-visible benefit
- a direct benchmark target using existing `gpt-sovits-mlx/scripts/benchmark.py`

## Acceptance criteria

A first acceptable implementation should satisfy:

- no behavior change when quantization is disabled
- successful autoregressive generation with cache enabled
- support for `K=q8, V=q4`
- measurable memory reduction vs baseline
- no crashes on long decode
- bounded logits drift and acceptable output quality degradation

## Bottom-line recommendation

Implement mixed quantization as a **split cache architecture**, not as a single shared cache format.

Start with:
1. split K/V stores
2. quantize-on-append
3. dequantize-on-materialize
4. support `K=q8, V=q4`
5. add TurboQuant later as a policy layer over the same abstraction

This matches the current repository structure, reuses existing quantization patterns already present in `mlx-rs`, and minimizes disruption to the existing attention and generation code paths.