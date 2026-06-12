//! Production session for Gemma4 target + assistant pair speculative
//! decoding (the MTPLX-Optimized-Speed checkpoints).
//!
//! Encapsulates the configuration validated in
//! `examples/draft_verify_spike.rs` (see `gemma4-pair-adapter-wip.md`,
//! 2026-06-11):
//!
//! * **Folded cycle** — ONE target pass per cycle over
//!   `[pending, drafts..]`: it verifies the drafts, installs the pending
//!   token's K/V, and yields the hidden state seeding the next drafter
//!   round. No dedicated bonus forward.
//! * **GPU-resident greedy drafting** — argmax + next-step embedding stay
//!   on device; one eval per draft block.
//! * **Shrink-only adaptive block** — shrink by 1 (floor 2) after a
//!   zero-accept cycle, recover by 1 after a full-accept cycle, never
//!   above the configured block. Default block 4.
//! * Greedy at `temp == 0`; stochastic drafting + Leviathan-Chen
//!   acceptance (with optional top-k / top-p truncation) at `temp > 0`.
//!
//! Measured on Gemma4-27B-MTPLX (AR baseline 11.7 tok/s, 96 tok, chat
//! template, temp 0): sky-blue 16.7 tok/s, fibonacci 18.5, french-rev
//! ~12.3 — 1.05-1.6x over AR.

use std::path::Path;

use mlx_rs::{
    argmax_axis,
    error::Exception,
    module::Module,
    ops::{indexing::{IndexOp, NewAxis}, softmax_axis},
    transforms::eval,
    Array, Dtype,
};
use mlx_rs_core::error::Error;

use crate::assistant::{
    build_inputs_embeds, load_assistant_model, AssistantModel, SharedKvStates,
};
use crate::init_cache;
use crate::model::ModelInput;
use crate::mtplx_target::load_mtplx_target;
use crate::{KVCache, Model};

/// Gemma4 end-of-sequence ids (`<eos>`, `<turn|>`, channel marker).
pub const PAIR_EOS_TOKEN_IDS: &[i32] = &[1, 106, 50];

#[derive(Debug, Clone)]
pub struct PairGenerateOptions {
    /// Draft block length (and adaptive ceiling). Chains rarely survive
    /// past ~4 and drafter steps dominate cycle cost — keep this small.
    pub block: usize,
    /// Shrink-only adaptive block policy (floor 2, ceiling `block`).
    pub adaptive: bool,
    pub max_tokens: usize,
    /// 0.0 = greedy draft + longest-prefix-match acceptance. > 0.0 =
    /// stochastic draft + Leviathan-Chen acceptance (distribution-
    /// preserving at temperature).
    pub temp: f32,
    /// Top-k truncation for stochastic sampling (0 = off).
    pub top_k: usize,
    /// Top-p (nucleus) truncation for stochastic sampling (0.0 = off).
    pub top_p: f32,
    /// RNG seed for stochastic runs. `None` seeds from the wall clock.
    pub seed: Option<u64>,
    /// When set, reuse a saved KV prefix for the longest matching prompt
    /// prefix (multi-turn chat: each turn re-prefills only the new
    /// suffix) and save the extended prefix after prefill.
    pub prompt_cache_dir: Option<std::path::PathBuf>,
}

impl Default for PairGenerateOptions {
    fn default() -> Self {
        Self {
            block: 4,
            adaptive: true,
            max_tokens: 512,
            temp: 0.0,
            top_k: 0,
            top_p: 0.0,
            seed: None,
            prompt_cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PairMetrics {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub cycles: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
}

impl PairMetrics {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted > 0 {
            self.accepted as f64 / self.drafted as f64
        } else {
            0.0
        }
    }

    pub fn decode_tok_per_s(&self) -> f64 {
        if self.decode_s > 0.0 {
            self.emitted as f64 / self.decode_s
        } else {
            0.0
        }
    }
}

/// Splitmix64 RNG in [0,1).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5851f42d4c957f2d)
        });
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = x;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Target + assistant pair with the KV tap layers resolved.
pub struct Gemma4PairSession {
    pub target: Model,
    pub assistant: AssistantModel,
    last_sliding: usize,
    last_full: usize,
}

impl Gemma4PairSession {
    /// Load from an MTPLX pair root containing `target/` and `assistant/`
    /// subdirectories (e.g. `models/Gemma4-27B-MTPLX-Optimized-Speed`).
    pub fn load(pair_root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = pair_root.as_ref();
        let target = load_mtplx_target(root.join("target"))?;
        let assistant = load_assistant_model(root.join("assistant"))?;
        Self::new(target, assistant)
    }

    /// Build from already-loaded models. Tap layers (the target layers
    /// whose K/V the assistant borrows) are the LAST sliding and LAST
    /// full-attention layers, per the HF reference.
    pub fn new(target: Model, assistant: AssistantModel) -> Result<Self, Error> {
        let layer_types = &target.args.layer_types;
        let last_sliding = layer_types
            .iter()
            .rposition(|t| t == "sliding_attention")
            .ok_or_else(|| Error::Model("pair target has no sliding_attention layer".into()))?;
        let last_full = layer_types
            .iter()
            .rposition(|t| t == "full_attention")
            .ok_or_else(|| Error::Model("pair target has no full_attention layer".into()))?;
        Ok(Self {
            target,
            assistant,
            last_sliding,
            last_full,
        })
    }

    /// Generate up to `opts.max_tokens` tokens. `on_token` is called once
    /// per committed token, in order. Returns the committed tokens and
    /// run metrics.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        opts: &PairGenerateOptions,
        eos: &[i32],
        mut on_token: impl FnMut(i32),
    ) -> Result<(Vec<i32>, PairMetrics), Exception> {
        let mut metrics = PairMetrics::default();
        let n_layers = self.target.args.num_hidden_layers as usize;
        let hidden_h = self.target.args.hidden_size;
        let mut cache: Vec<KVCache> = init_cache::<KVCache>(n_layers);
        let mut rng = Rng::new(opts.seed);

        // ── Prefill. The cache holds only validated tokens; `pending`
        // (the model's next-token prediction) joins it with the first
        // folded verify pass. With a prompt-cache dir, reuse the longest
        // matching saved prefix and prefill only the suffix (the loader
        // always leaves >= 1 suffix token, so the seed hidden below is
        // well-defined).
        let prefill_start = std::time::Instant::now();
        let mut cached_prefix = 0usize;
        if let Some(dir) = opts.prompt_cache_dir.as_ref() {
            if let Some((loaded, n)) = KVCache::try_load_kv_caches(prompt_ids, dir)? {
                if loaded.len() == cache.len() {
                    cache = loaded;
                    cached_prefix = n;
                }
            }
        }
        let prompt_arr = Array::from(&prompt_ids[cached_prefix..]).index(NewAxis);
        let prefill_hidden = self.target.model.forward(ModelInput {
            inputs: &prompt_arr,
            mask: None,
            cache: &mut cache,
        })?;
        let mut seed_hidden = prefill_hidden
            .index((.., -1, ..))
            .reshape(&[1, 1, hidden_h])?;
        let first_logits = self.target.forward_via_hidden(&seed_hidden)?;
        eval([&first_logits])?;
        let mut pending: i32 = if opts.temp > 0.0 {
            let probs = softmax_probs(
                &first_logits.reshape(&[-1])?,
                opts.temp,
                opts.top_k,
                opts.top_p,
            )?;
            sample_categorical(&probs, &mut rng)
        } else {
            argmax_host(&first_logits.reshape(&[-1])?)?
        };
        // GPU-resident twin of `pending`, used to build the verify chain
        // without a host round-trip.
        let mut pending_arr = Array::from(&[pending][..]);
        // Persist the (possibly extended) prompt prefix for the next turn.
        if let Some(dir) = opts.prompt_cache_dir.as_ref() {
            if prompt_ids.len() > cached_prefix {
                KVCache::save_kv_caches(&cache, prompt_ids, dir)?;
            }
        }
        metrics.prefill_s = prefill_start.elapsed().as_secs_f64();

        // ── Folded speculative loop.
        let decode_start = std::time::Instant::now();
        let mut emitted: Vec<i32> = Vec::new();
        let mut cur_block = opts.block.max(1);

        'outer: while emitted.len() < opts.max_tokens {
            emitted.push(pending);
            on_token(pending);
            if eos.contains(&pending) || emitted.len() >= opts.max_tokens {
                break;
            }

            let kv = self.build_shared_kv(&cache)?;
            // `pending` sits at position cache_len (frozen for the whole
            // draft block, per the HF candidate generator).
            let draft_pos = kv.sliding_k.shape()[2];

            // Draft + verify. Greedy fuses the entire cycle — draft chain,
            // verify forward, and a batched argmax over all verify rows —
            // into ONE lazy graph with a single eval and a single
            // device→host copy. Stochastic mode inherently syncs per draft
            // step (host-side sampling) and keeps the unfused shape.
            let (drafted, accepted, next_pending, hidden);
            if opts.temp > 0.0 {
                let (d, q_rows) = self.draft_stochastic(
                    &kv,
                    draft_pos,
                    pending,
                    seed_hidden.clone(),
                    cur_block,
                    opts,
                    &mut rng,
                )?;
                let mut chain: Vec<i32> = Vec::with_capacity(d.len() + 1);
                chain.push(pending);
                chain.extend_from_slice(&d);
                let in_arr = Array::from(chain.as_slice()).index(NewAxis);
                let h = self.target.model.forward(ModelInput {
                    inputs: &in_arr,
                    mask: None,
                    cache: &mut cache,
                })?;
                let logits = self.target.forward_via_hidden(&h)?;
                eval([&logits])?;
                let q_rows = q_rows.expect("stochastic drafting returns q_rows");
                let (acc, next) = lc_accept(&logits, &d, &q_rows, opts, &mut rng)?;
                drafted = d;
                accepted = acc;
                next_pending = next;
                hidden = h;
            } else {
                let draft_toks = self.draft_greedy(
                    &kv,
                    draft_pos,
                    &pending_arr,
                    seed_hidden.clone(),
                    cur_block,
                )?;
                // Verify input [1, N+1] assembled on device from the lazy
                // draft tokens.
                let mut parts: Vec<&Array> = Vec::with_capacity(draft_toks.len() + 1);
                parts.push(&pending_arr);
                parts.extend(draft_toks.iter());
                let chain_flat = mlx_rs::ops::concatenate(&parts)?;
                let in_arr = chain_flat.reshape(&[1, -1])?;
                let h = self.target.model.forward(ModelInput {
                    inputs: &in_arr,
                    mask: None,
                    cache: &mut cache,
                })?;
                let logits = self.target.forward_via_hidden(&h)?;
                // Row i of `preds` is the target's argmax for drafted[i]'s
                // position; row N is the bonus row.
                let preds = argmax_axis!(&logits, -1)?.as_dtype(Dtype::Int32)?;
                let drafted_flat = if draft_toks.len() > 1 {
                    let refs: Vec<&Array> = draft_toks.iter().collect();
                    mlx_rs::ops::concatenate(&refs)?
                } else {
                    draft_toks[0].clone()
                };
                // THE one sync of the cycle.
                eval([&preds, &drafted_flat])?;
                let preds_row = preds.index((0, ..)).contiguous()?;
                eval([&preds_row])?;
                let preds_host = preds_row.as_slice::<i32>().to_vec();
                let d = drafted_flat.as_slice::<i32>().to_vec();

                let mut acc = 0usize;
                for (i, &dt) in d.iter().enumerate() {
                    if dt == preds_host[i] {
                        acc += 1;
                    } else {
                        break;
                    }
                }
                drafted = d;
                accepted = acc;
                next_pending = preds_host[accepted];
                hidden = h;
            }

            // Keep [pending + accepted drafts]; trim the rejected tail.
            let to_drop = (drafted.len() - accepted) as i32;
            if to_drop > 0 {
                for c in cache.iter_mut() {
                    c.trim(to_drop);
                }
            }
            seed_hidden = hidden
                .index((.., accepted as i32, ..))
                .reshape(&[1, 1, hidden_h])?;

            metrics.cycles += 1;
            metrics.drafted += drafted.len();
            metrics.accepted += accepted;
            for i in 0..accepted {
                emitted.push(drafted[i]);
                on_token(drafted[i]);
                if eos.contains(&drafted[i]) || emitted.len() >= opts.max_tokens {
                    break 'outer;
                }
            }

            if opts.adaptive {
                if accepted == cur_block {
                    cur_block = (cur_block + 1).min(opts.block);
                } else if accepted == 0 {
                    cur_block = cur_block.saturating_sub(1).max(2);
                }
            }
            pending = next_pending;
            pending_arr = Array::from(&[pending][..]);
        }

        metrics.decode_s = decode_start.elapsed().as_secs_f64();
        metrics.emitted = emitted.len();
        Ok((emitted, metrics))
    }

    fn build_shared_kv(&self, cache: &[KVCache]) -> Result<SharedKvStates, Exception> {
        let (sk, sv) = cache[self.last_sliding]
            .current_kv()
            .ok_or_else(|| Exception::custom("pair session: sliding cache empty"))?;
        let (fk, fv) = cache[self.last_full]
            .current_kv()
            .ok_or_else(|| Exception::custom("pair session: full cache empty"))?;
        Ok(SharedKvStates {
            sliding_k: sk,
            sliding_v: sv,
            full_k: fk,
            full_v: fv,
        })
    }

    fn scaled_embed(&mut self, ids: &Array) -> Result<Array, Exception> {
        let embed = self.target.model.embed_tokens.forward(ids)?;
        let scale = self.target.model.embed_scale.as_dtype(embed.dtype())?;
        embed.multiply(&scale)?.as_dtype(Dtype::Bfloat16)
    }

    /// GPU-resident greedy drafter. Returns the per-step token arrays
    /// (each `[1]`, i32) WITHOUT evaluating — the caller fuses them into
    /// the verify graph so the whole cycle materializes with one eval.
    fn draft_greedy(
        &mut self,
        kv: &SharedKvStates,
        draft_pos: i32,
        seed_token: &Array,
        seed_hidden: Array,
        n: usize,
    ) -> Result<Vec<Array>, Exception> {
        let seed_2d = seed_token.reshape(&[1, 1])?;
        let mut prev_embed = self.scaled_embed(&seed_2d)?;
        let mut recurrent = seed_hidden;
        let mut toks: Vec<Array> = Vec::with_capacity(n);
        for _ in 0..n {
            let inputs = build_inputs_embeds(&prev_embed, &recurrent)?;
            let aout = self.assistant.forward(&inputs, draft_pos, kv)?;
            let tok = argmax_axis!(&aout.logits.index((.., -1, ..)), -1)?
                .as_dtype(Dtype::Int32)?;
            let tok_2d = tok.reshape(&[1, 1])?;
            prev_embed = self.scaled_embed(&tok_2d)?;
            recurrent = aout.last_hidden;
            toks.push(tok);
        }
        Ok(toks)
    }

    /// Stochastic drafter: samples each step from the truncated softmax;
    /// returns the drafted tokens and the q-rows for LC acceptance.
    #[allow(clippy::too_many_arguments)]
    fn draft_stochastic(
        &mut self,
        kv: &SharedKvStates,
        draft_pos: i32,
        seed_token: i32,
        seed_hidden: Array,
        n: usize,
        opts: &PairGenerateOptions,
        rng: &mut Rng,
    ) -> Result<(Vec<i32>, Option<Vec<Vec<f32>>>), Exception> {
        let seed_arr = Array::from(&[seed_token][..]).index(NewAxis);
        let mut prev_embed = self.scaled_embed(&seed_arr)?;
        let mut recurrent = seed_hidden;
        let mut tokens = Vec::with_capacity(n);
        let mut q_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let inputs = build_inputs_embeds(&prev_embed, &recurrent)?;
            let aout = self.assistant.forward(&inputs, draft_pos, kv)?;
            let row = aout.logits.index((.., -1, ..)).reshape(&[-1])?;
            let probs = softmax_probs(&row, opts.temp, opts.top_k, opts.top_p)?;
            let tok = sample_categorical(&probs, rng);
            tokens.push(tok);
            q_rows.push(probs);
            let tok_arr = Array::from(&[tok][..]).index(NewAxis);
            prev_embed = self.scaled_embed(&tok_arr)?;
            recurrent = aout.last_hidden;
        }
        Ok((tokens, Some(q_rows)))
    }
}

fn argmax_host(a: &Array) -> Result<i32, Exception> {
    Ok(argmax_axis!(a, -1)?.as_dtype(Dtype::Int32)?.item::<i32>())
}

/// fp32 softmax of a `[vocab]` logits row with temperature + optional
/// top-k / top-p truncation (renormalized).
fn softmax_probs(
    row_logits: &Array,
    temp: f32,
    top_k: usize,
    top_p: f32,
) -> Result<Vec<f32>, Exception> {
    let f32_row = row_logits.as_dtype(Dtype::Float32)?;
    let scaled = if (temp - 1.0).abs() < f32::EPSILON {
        f32_row
    } else {
        let inv = Array::from_slice(&[1.0_f32 / temp], &[1]);
        mlx_rs::ops::multiply(&f32_row, &inv)?
    };
    let sm = softmax_axis(&scaled, -1, Some(true))?;
    eval([&sm])?;
    let mut row = sm.as_slice::<f32>().to_vec();
    if top_k > 0 {
        truncate_top_k(&mut row, top_k);
    }
    if top_p > 0.0 && top_p < 1.0 {
        truncate_top_p(&mut row, top_p);
    }
    Ok(row)
}

fn truncate_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let mut sorted: Vec<f32> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    let mut kept = 0usize;
    let mut sum = 0.0_f32;
    for p in probs.iter_mut() {
        if *p >= threshold && kept < k {
            kept += 1;
            sum += *p;
        } else {
            *p = 0.0;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

fn truncate_top_p(probs: &mut [f32], p: f32) {
    let mut nz: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| *v > 0.0)
        .collect();
    if nz.is_empty() {
        return;
    }
    nz.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = nz.iter().map(|(_, v)| *v).sum();
    let mut cum = 0.0_f32;
    let mut keep = std::collections::HashSet::new();
    for (i, v) in &nz {
        keep.insert(*i);
        cum += *v;
        if cum >= p * total {
            break;
        }
    }
    let mut sum = 0.0_f32;
    for (i, v) in probs.iter_mut().enumerate() {
        if keep.contains(&i) {
            sum += *v;
        } else {
            *v = 0.0;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

fn sample_categorical(probs: &[f32], rng: &mut Rng) -> i32 {
    let u = rng.next_f32();
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as i32;
        }
    }
    (probs.len() - 1) as i32
}

/// Leviathan-Chen acceptance over folded verify logits `[1, N+1, V]`
/// (row i = distribution for drafted[i]'s position; row N = bonus row).
/// Returns `(n_accepted, next_pending)`.
fn lc_accept(
    logits: &Array,
    drafted: &[i32],
    q_rows: &[Vec<f32>],
    opts: &PairGenerateOptions,
    rng: &mut Rng,
) -> Result<(usize, i32), Exception> {
    let n = drafted.len();
    let mut accepted = 0usize;
    for i in 0..n {
        let p = softmax_probs(
            &logits.index((.., i as i32, ..)).reshape(&[-1])?,
            opts.temp,
            opts.top_k,
            opts.top_p,
        )?;
        let q = &q_rows[i];
        let t = drafted[i] as usize;
        if t >= p.len() || t >= q.len() {
            return Err(Exception::custom(format!(
                "draft token {t} out of vocab"
            )));
        }
        let ratio = if q[t] > 0.0 { p[t] / q[t] } else { f32::INFINITY };
        if rng.next_f32() < ratio.min(1.0) {
            accepted += 1;
            continue;
        }
        // Reject: sample the correction from (p - q)+ normalized.
        let mut s = 0.0_f32;
        let mut residual = vec![0.0_f32; p.len()];
        for j in 0..p.len() {
            let d = p[j] - q[j];
            if d > 0.0 {
                residual[j] = d;
                s += d;
            }
        }
        let correction = if s > 0.0 {
            let u = rng.next_f32() * s;
            let mut acc = 0.0_f32;
            let mut idx = residual.len() - 1;
            for (j, &r) in residual.iter().enumerate() {
                acc += r;
                if r > 0.0 && u < acc {
                    idx = j;
                    break;
                }
            }
            idx as i32
        } else {
            // Degenerate overlap: fall back to sampling from p.
            sample_categorical(&p, rng)
        };
        return Ok((accepted, correction));
    }
    // Full acceptance: sample the bonus from the last row.
    let p_bonus = softmax_probs(
        &logits.index((.., n as i32, ..)).reshape(&[-1])?,
        opts.temp,
        opts.top_k,
        opts.top_p,
    )?;
    Ok((accepted, sample_categorical(&p_bonus, rng)))
}
