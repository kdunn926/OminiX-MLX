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
use qwen3_mlx::{load_model, load_tokenizer, KVCache, Model, ModelInput};

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

    /// Relevance of `doc` to `query`: `logit(yes) − logit(no)` at the last
    /// position (monotone with the model card's softmax score; we only rank).
    fn score_one(&mut self, query: &str, doc: &str) -> Result<f32> {
        let content = format!("<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc}");
        let cids: Vec<i32> = self
            .tok
            .encode(content, false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect();
        let mut ids = Vec::with_capacity(self.pre.len() + cids.len() + self.suf.len());
        ids.extend_from_slice(&self.pre);
        ids.extend_from_slice(&cids);
        ids.extend_from_slice(&self.suf);
        let arr = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let logits = self
            .model
            .forward_last_logits(ModelInput { inputs: &arr, mask: None, cache: &mut cache })
            .map_err(|e| anyhow!("reranker forward: {e}"))?;
        let yes = logits.index((0, self.yes_id));
        let no = logits.index((0, self.no_id));
        let diff = yes
            .subtract(&no)
            .and_then(|d| d.as_dtype(Dtype::Float32))
            .and_then(|d| d.reshape(&[1]))
            .map_err(|e| anyhow!("reranker score: {e}"))?;
        eval([&diff]).map_err(|e| anyhow!("eval: {e}"))?;
        Ok(diff.as_slice::<f32>()[0])
    }

    /// Rank `chunks` by relevance to `query`; returns chunk indices best-first.
    pub fn rank(&mut self, query: &str, chunks: &[String]) -> Result<Vec<usize>> {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            scored.push((i, self.score_one(query, c)?));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(i, _)| i).collect())
    }
}
