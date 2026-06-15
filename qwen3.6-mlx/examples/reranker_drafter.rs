//! Reranker "drafter" for kvflash host paging.
//!
//! The kvflash paged cache keeps every 64-token chunk resident in unified
//! memory but bounds the attention *working set*. Which chunks stay resident is
//! the recall question. The source PR (lucebox-hub#373) answers it with a small
//! drafter that scores chunk↔query relevance; this is that role, played by
//! **Qwen3-Reranker-0.6B** — a Qwen3 dense causal LM used as a cross-encoder.
//!
//! A reranker scores *text* `(query, document)` pairs, so it sidesteps the
//! target/drafter tokenizer mismatch entirely: we chunk by the *target's* token
//! stream (the same 64-token blocks the cache freezes), decode each chunk to
//! text, and let the reranker tokenize it however it likes. The output is one
//! relevance score per chunk — ranked best-first and installed as the cache's
//! residency policy via `kvflash::set_drafter_pins`.
//!
//! Relevance score (per the model card): `softmax([logit("no"), logit("yes")])`
//! at the last position. For *ranking* we only need the order, so the cheaper
//! monotone `logit("yes") − logit("no")` suffices (no softmax needed).

use anyhow::{anyhow, Result};
use mlx_rs::{ops::indexing::IndexOp, transforms::eval, Array, Dtype};
use mlx_rs_core::Tokenizer;
use qwen3_mlx::{load_model, load_tokenizer, AttentionMask, KVCache, Model, ModelInput};

const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const INSTRUCT: &str = "Given a web search query, retrieve relevant passages that answer the query";

pub struct RerankerDrafter {
    model: Model,
    tok: Tokenizer,
    pre: Vec<i32>,
    suf: Vec<i32>,
    yes_id: i32,
    no_id: i32,
}

impl RerankerDrafter {
    pub fn load(dir: &str) -> Result<Self> {
        let model = load_model(dir).map_err(|e| anyhow!("load reranker: {e}"))?;
        let tok = load_tokenizer(dir).map_err(|e| anyhow!("load reranker tokenizer: {e}"))?;
        let enc = |s: &str| -> Result<Vec<i32>> {
            Ok(tok
                .encode(s, false)
                .map_err(|e| anyhow!("{e}"))?
                .get_ids()
                .iter()
                .map(|&x| x as i32)
                .collect())
        };
        let pre = enc(PREFIX)?;
        let suf = enc(SUFFIX)?;
        let yes_id = tok.token_to_id("yes").ok_or_else(|| anyhow!("no 'yes' token"))? as i32;
        let no_id = tok.token_to_id("no").ok_or_else(|| anyhow!("no 'no' token"))? as i32;
        Ok(Self { model, tok, pre, suf, yes_id, no_id })
    }

    /// Token ids for one `(query, doc)` reranker prompt.
    fn encode_pair(&self, query: &str, doc: &str) -> Result<Vec<i32>> {
        let content = format!("<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc}");
        let cids = self.tok.encode(content, false).map_err(|e| anyhow!("{e}"))?;
        let mut ids = Vec::with_capacity(self.pre.len() + cids.len() + self.suf.len());
        ids.extend_from_slice(&self.pre);
        ids.extend(cids.get_ids().iter().map(|&x| x as i32));
        ids.extend_from_slice(&self.suf);
        Ok(ids)
    }

    /// Score one **batch** of pre-tokenized prompts in a single forward.
    ///
    /// Sequences are **left-padded** to the batch's max length so every row's
    /// last real token (the suffix end, where the yes/no logit lives) sits at
    /// index -1 — letting us reuse `forward_last_logits`. A boolean attention
    /// mask `(j ≤ i) AND (j ≥ pad_count[b])` keeps it causal *and* blocks the
    /// leading pad keys; partial RoPE is relative so the per-row position shift
    /// from padding doesn't change a sequence's internal attention.
    fn score_batch(&mut self, seqs: &[Vec<i32>]) -> Result<Vec<f32>> {
        let b = seqs.len() as i32;
        let max_l = seqs.iter().map(|s| s.len()).max().unwrap_or(1) as i32;
        let pad_id = 0i32;
        let mut flat = vec![pad_id; (b * max_l) as usize];
        let mut pad_counts = vec![0i32; b as usize];
        for (row, s) in seqs.iter().enumerate() {
            let pad = max_l as usize - s.len();
            pad_counts[row] = pad as i32;
            let base = row * max_l as usize + pad;
            flat[base..base + s.len()].copy_from_slice(s);
        }
        let inputs = Array::from_slice(&flat, &[b, max_l]);

        // Bool mask [B,1,L,L]: causal (i≥j) AND key not in this row's pad region.
        let idx: Vec<i32> = (0..max_l).collect();
        let cols = Array::from_slice(&idx, &[1, 1, 1, max_l]); // j
        let rows = Array::from_slice(&idx, &[1, 1, max_l, 1]); // i
        let causal = rows.ge(&cols).map_err(|e| anyhow!("{e}"))?; // [1,1,L,L] i≥j
        let pc = Array::from_slice(&pad_counts, &[b, 1, 1, 1]);
        let not_pad = cols.ge(&pc).map_err(|e| anyhow!("{e}"))?; // [B,1,1,L] j≥pad
        let mask = causal.logical_and(&not_pad).map_err(|e| anyhow!("{e}"))?;
        let am = AttentionMask::Array(mask);

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let logits = self
            .model
            .forward_last_logits(ModelInput { inputs: &inputs, mask: Some(&am), cache: &mut cache })
            .map_err(|e| anyhow!("reranker forward: {e}"))?; // [B, vocab]
        let yes = logits.index((.., self.yes_id));
        let no = logits.index((.., self.no_id));
        let diff = yes
            .subtract(&no)
            .and_then(|d| d.as_dtype(Dtype::Float32))
            .map_err(|e| anyhow!("reranker score: {e}"))?; // [B]
        eval([&diff]).map_err(|e| anyhow!("eval: {e}"))?;
        Ok(diff.as_slice::<f32>().to_vec())
    }

    /// Rank `chunks` by relevance to `query`; returns chunk indices best-first.
    ///
    /// Chunks are sorted by token length and scored in batches (one forward per
    /// batch instead of one per chunk) — sorting groups similar lengths so the
    /// left-padding overhead stays small. `DFLASH_RERANK_BATCH` sets the batch
    /// size (default 32).
    pub fn rank(&mut self, query: &str, chunks: &[String]) -> Result<Vec<usize>> {
        let batch: usize = std::env::var("DFLASH_RERANK_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(32);
        // Tokenize all, then sort indices by length to minimize padding waste.
        let mut tokenized: Vec<(usize, Vec<i32>)> = Vec::with_capacity(chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            tokenized.push((i, self.encode_pair(query, c)?));
        }
        tokenized.sort_by_key(|(_, s)| s.len());

        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(chunks.len());
        for group in tokenized.chunks(batch) {
            let seqs: Vec<Vec<i32>> = group.iter().map(|(_, s)| s.clone()).collect();
            let scores = self.score_batch(&seqs)?;
            for ((orig_i, _), score) in group.iter().zip(scores) {
                scored.push((*orig_i, score));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(i, _)| i).collect())
    }
}
