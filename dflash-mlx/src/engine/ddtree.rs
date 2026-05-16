//! DDTree spike — Diffusion Draft Tree (Ringel & Romano, 2026).
//!
//! Port of the algorithm in <https://github.com/liranringel/ddtree> on top of
//! the existing DFlash drafter + target adapter. Specifically reuses:
//!   - `DFlashDraftModel` / `DFlashDraftAdapter` for the block-diffusion
//!     drafter (its `draft_block` already returns per-position logits over
//!     the whole block).
//!   - `Gemma4TargetAdapter` / `Qwen36TargetAdapter` via `TargetModel` for
//!     verify + KV rollback.
//!
//! What DDTree adds over plain DFlash:
//!   - Instead of taking argmax-per-position from `draft_logits` → linear
//!     chain of `block_size - 1` candidates, it builds a budget-N tree of
//!     joint-top-k continuations (heap-prioritised by joint log-prob).
//!   - Verifies the tree and accepts the longest path the target argmax
//!     walks.
//!
//! Spike simplifications vs the Python reference:
//!   1. No fused tree-mask target forward. Each branch is verified
//!      sequentially with KV snapshot/restore between branches. This costs
//!      `n_branches × verify` per cycle instead of `1 × verify` in the
//!      paper. Useful as a directional measurement; the production port
//!      needs `forward_with_position_ids + tree_mask` (RoPE rework in
//!      `gemma4-mlx::Attention`).
//!   2. No in-place interior-KV compaction. Spike relies on the existing
//!      `TargetModel::rollback_kv` (trim-from-tail) between branches.
//!
//! These simplifications mean the spike's wall-clock cost grows linearly
//! with branch count. Speedup over plain DFlash will be muted relative to
//! the fully-fused implementation; an above-baseline result is a lower
//! bound on the algorithm's real-world benefit.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use mlx_rs::{
    argmax_axis, error::Exception, ops::indexing::IndexOp, transforms::eval, Array, Dtype,
};

use crate::engine::spec_epoch::TargetModel;

/// One node in the draft tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub token_id: u32,
    /// Index in the linear `nodes` vector of this node's parent. `usize::MAX`
    /// for the root token (which has no parent in the tree itself; the root
    /// is the staged "seed" token whose K/V already lives in the target's
    /// cache).
    pub parent: usize,
    /// 1-based depth in the tree. depth=1 ⇒ direct child of root.
    pub depth: usize,
    /// Joint log-probability from root to this node (sum of per-step log-
    /// probs).
    pub joint_logp: f32,
}

/// Heap entry for tree expansion. Negative log-prob so `BinaryHeap` (max-
/// heap) gives the smallest `-logp` first ⇒ highest probability first.
#[derive(Debug, Clone, PartialEq)]
struct HeapEntry {
    /// -joint_logp so the max-heap pops largest joint_logp first.
    neg_logp: f32,
    /// Index in the `nodes` Vec of this node's parent (or `usize::MAX` for
    /// children of the implicit root). The parent's depth determines this
    /// candidate's depth (depth = parent.depth + 1, or 1 for root children).
    parent_idx: usize,
    /// Depth this candidate would have if accepted (= parent_depth + 1).
    depth: usize,
    /// Rank into the depth-th row's top-k token list (0 = top-1).
    rank: usize,
}

impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Smaller neg_logp = larger logp = higher priority. BinaryHeap is a
        // max-heap, so we want the natural order; but we want the SMALLEST
        // neg_logp to pop first → reverse.
        other
            .neg_logp
            .partial_cmp(&self.neg_logp)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a DDTree from per-position top-k token IDs and log-probs.
///
/// `top_token_ids[d][r]` = the r-th most likely token at depth d (0-indexed
/// from depth 1). `top_log_probs[d][r]` = its log-prob (already
/// log-softmax'd; values <= 0).
///
/// `budget` is the maximum number of tree nodes to materialise (excluding
/// the implicit root). The algorithm expands the heap budget times,
/// choosing the next-highest-joint-logp candidate each step.
pub fn build_tree(
    top_token_ids: &[Vec<u32>],
    top_log_probs: &[Vec<f32>],
    budget: usize,
) -> Vec<TreeNode> {
    let depth_limit = top_token_ids.len();
    if budget == 0 || depth_limit == 0 {
        return Vec::new();
    }
    let topk = top_token_ids[0].len();
    if topk == 0 {
        return Vec::new();
    }

    let mut nodes: Vec<TreeNode> = Vec::with_capacity(budget);
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    // Seed: root-children candidates at depth 1, rank 0.
    heap.push(HeapEntry {
        neg_logp: -top_log_probs[0][0],
        parent_idx: usize::MAX,
        depth: 1,
        rank: 0,
    });

    while let Some(entry) = heap.pop() {
        if nodes.len() >= budget {
            break;
        }
        let depth_idx = entry.depth - 1;
        let token_id = top_token_ids[depth_idx][entry.rank];
        let logp_at_step = top_log_probs[depth_idx][entry.rank];
        let parent_joint = if entry.parent_idx == usize::MAX {
            0.0
        } else {
            nodes[entry.parent_idx].joint_logp
        };
        let joint_logp = parent_joint + logp_at_step;
        let my_idx = nodes.len();
        nodes.push(TreeNode {
            token_id,
            parent: entry.parent_idx,
            depth: entry.depth,
            joint_logp,
        });

        // Push sibling at same parent + same depth + next rank.
        if entry.rank + 1 < topk {
            let next_step_logp = top_log_probs[depth_idx][entry.rank + 1];
            let sibling_joint = parent_joint + next_step_logp;
            heap.push(HeapEntry {
                neg_logp: -sibling_joint,
                parent_idx: entry.parent_idx,
                depth: entry.depth,
                rank: entry.rank + 1,
            });
        }
        // Push first child at next depth.
        if entry.depth < depth_limit {
            let child_step_logp = top_log_probs[entry.depth][0];
            let child_joint = joint_logp + child_step_logp;
            heap.push(HeapEntry {
                neg_logp: -child_joint,
                parent_idx: my_idx,
                depth: entry.depth + 1,
                rank: 0,
            });
        }
    }

    nodes
}

/// Walk down the tree from the root using the target's per-path posteriors
/// to pick the longest accepted path.
///
/// `nodes`: the tree built by `build_tree`.
/// `predicted_after_node`: for each node, the target argmax token id it
///     would emit AFTER seeing the path root→…→node. This is what we
///     compare against the next-depth candidates to extend.
/// `root_predicted_token`: target's argmax for the seed (= the prediction
///     that would extend the empty path → depth 1).
///
/// Returns indices into `nodes` of the accepted path, plus the bonus token
/// (target's prediction after the last accepted node).
pub fn accept_path(
    nodes: &[TreeNode],
    predicted_after_node: &[u32],
    root_predicted_token: u32,
) -> (Vec<usize>, u32) {
    let mut accepted: Vec<usize> = Vec::new();
    let mut current_parent: usize = usize::MAX;
    let mut current_predicted = root_predicted_token;
    let mut current_depth: usize = 0;

    loop {
        // Find a child of current_parent whose token_id == current_predicted.
        let next_child = nodes.iter().enumerate().find(|(_, n)| {
            n.parent == current_parent
                && n.depth == current_depth + 1
                && n.token_id == current_predicted
        });
        match next_child {
            None => return (accepted, current_predicted),
            Some((idx, _)) => {
                accepted.push(idx);
                current_predicted = predicted_after_node[idx];
                current_parent = idx;
                current_depth += 1;
            }
        }
    }
}

/// Pull top-k token IDs and log-probs out of a `[block, vocab]` logits
/// tensor for each draft position. Returns (top_token_ids, top_log_probs)
/// as `block`-length vectors of length-`k` rows.
pub fn topk_per_position(
    logits: &Array,
    k: usize,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<f32>>), Exception> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(Exception::custom(format!(
            "topk_per_position expects 2D logits, got shape {:?}",
            shape
        )));
    }
    let block = shape[0] as usize;
    let vocab = shape[1] as usize;
    // Compute log-softmax along last axis.
    let f32_logits = logits.as_dtype(Dtype::Float32)?;
    eval([&f32_logits])?;
    let max_vals = mlx_rs::ops::max_axis(&f32_logits, -1, true)?;
    let shifted = f32_logits.subtract(&max_vals)?;
    let exp = mlx_rs::ops::exp(&shifted)?;
    let sum = mlx_rs::ops::sum_axis(&exp, -1, true)?;
    let logsumexp = mlx_rs::ops::log(&sum)?.add(&max_vals)?;
    let log_probs = f32_logits.subtract(&logsumexp)?;
    eval([&log_probs])?;
    let log_probs_slice = log_probs.as_slice::<f32>();

    let mut ids_out = Vec::with_capacity(block);
    let mut lp_out = Vec::with_capacity(block);
    for d in 0..block {
        let row_start = d * vocab;
        let row = &log_probs_slice[row_start..row_start + vocab];
        let mut idx: Vec<usize> = (0..vocab).collect();
        // Partial sort: select top-k by descending log-prob.
        idx.select_nth_unstable_by(k.min(vocab - 1), |&a, &b| {
            row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top: Vec<(usize, f32)> = (0..k.min(vocab))
            .map(|i| (idx[i], row[idx[i]]))
            .collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ids_out.push(top.iter().map(|(i, _)| *i as u32).collect());
        lp_out.push(top.iter().map(|(_, lp)| *lp).collect());
    }
    Ok((ids_out, lp_out))
}

/// Tree-aware verify driver. Verifies each branch of the tree sequentially
/// on the target with KV snapshot/restore between branches, then walks the
/// tree to accept the longest path.
///
/// Returns (accepted_node_indices, bonus_token_id, accepted_tokens).
///
/// Side effect: leaves the target's KV cache committed for the accepted
/// path + bonus (via the final verify on `[accepted_tokens..., bonus]`).
pub fn verify_tree_naive<T: TargetModel>(
    target: &mut T,
    seed_token: u32,
    nodes: &[TreeNode],
) -> Result<(Vec<usize>, u32, Vec<u32>), Exception> {
    if nodes.is_empty() {
        // Trivial: just verify the seed.
        let seed_arr = Array::from(&[seed_token][..]).reshape(&[1, 1])?;
        let logits = target.verify(&seed_arr)?;
        eval([&logits])?;
        let bonus_arr = argmax_axis!(logits.index((.., -1, ..)).reshape(&[-1])?, -1)?
            .as_dtype(Dtype::Uint32)?;
        let bonus = bonus_arr.item::<u32>();
        return Ok((Vec::new(), bonus, Vec::new()));
    }

    // Enumerate all root-to-leaf paths in the tree. Each branch we verify is
    // [seed, path tokens...] starting from the target's current KV state.
    let leaves: Vec<usize> = (0..nodes.len())
        .filter(|&i| !nodes.iter().any(|n| n.parent == i))
        .collect();
    let mut paths: Vec<Vec<usize>> = leaves
        .into_iter()
        .map(|leaf| {
            let mut path = Vec::new();
            let mut cur = leaf;
            loop {
                path.push(cur);
                if nodes[cur].parent == usize::MAX {
                    break;
                }
                cur = nodes[cur].parent;
            }
            path.reverse();
            path
        })
        .collect();
    // Stable order: by descending joint_logp of the leaf (highest-prob path
    // first). This way the most likely branch's KV is in cache last, which
    // is usually what we want to keep committed.
    paths.sort_by(|a, b| {
        nodes[*b.last().unwrap()]
            .joint_logp
            .partial_cmp(&nodes[*a.last().unwrap()].joint_logp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // We store, for each NODE INDEX in the tree, the target's prediction
    // (argmax) AFTER visiting root + path-to-that-node. This is filled in
    // as we walk the branches. Also store the root's prediction (seed
    // alone).
    let n = nodes.len();
    let mut predicted_after_node: Vec<u32> = vec![0u32; n];
    let mut root_predicted: Option<u32> = None;

    // Verify each path. Between branches, rollback KV to just the seed.
    // We always feed `[seed, branch_tokens...]` so the seed's K/V gets
    // re-installed each branch — the bonus correction makes this clean.
    let mut last_verify_kept = 0usize;
    for path in &paths {
        // Roll back any prior verify state (drop the prior branch fully).
        if last_verify_kept > 0 {
            // The prior verify left the target advanced by `last_verify_kept`
            // positions. Roll back all of them so we restart from the same
            // post-prefill state for this branch.
            target.rollback_kv(0)?;
        }
        // Build input: [seed, branch[0].token, branch[1].token, ...]
        let mut ids: Vec<u32> = Vec::with_capacity(1 + path.len());
        ids.push(seed_token);
        for &idx in path {
            ids.push(nodes[idx].token_id);
        }
        let in_arr = Array::from(&ids[..]).reshape(&[1, ids.len() as i32])?;
        let logits = target.verify(&in_arr)?;
        eval([&logits])?;

        // logits[i] predicts the token AFTER input[i]. input[0] = seed →
        // logits[0] = prediction for "next token after seed". input[1+k]
        // = branch[k].token → logits[1+k] = prediction for "next token
        // after root + branch[0..=k]".
        let preds = argmax_axis!(logits, -1)?
            .as_dtype(Dtype::Uint32)?
            .reshape(&[-1])?;
        eval([&preds])?;
        let preds_slice = preds.as_slice::<u32>();
        // preds_slice has length = ids.len()
        if root_predicted.is_none() {
            root_predicted = Some(preds_slice[0]);
        }
        for (k, &node_idx) in path.iter().enumerate() {
            // preds_slice[1 + k] is the prediction after visiting node_idx.
            predicted_after_node[node_idx] = preds_slice[1 + k];
        }
        last_verify_kept = ids.len();
    }

    // Walk to find longest accepted path.
    let root_pred = root_predicted.unwrap_or(0);
    let (accepted, bonus) = accept_path(nodes, &predicted_after_node, root_pred);

    // After the loop, the target's KV reflects the LAST verified branch.
    // Roll back to the accepted prefix + bonus.
    // First fully roll back the last branch.
    target.rollback_kv(0)?;
    // Then re-feed [seed, accepted_tokens..., bonus] in one verify to
    // install the correct KV path. This also re-extracts the accepted
    // hidden, important for downstream draft cycles that use
    // `last_target_hidden`.
    let mut commit_ids: Vec<u32> = Vec::with_capacity(2 + accepted.len());
    commit_ids.push(seed_token);
    for &idx in &accepted {
        commit_ids.push(nodes[idx].token_id);
    }
    commit_ids.push(bonus);
    let commit_arr = Array::from(&commit_ids[..]).reshape(&[1, commit_ids.len() as i32])?;
    let _ = target.verify(&commit_arr)?;

    let accepted_tokens: Vec<u32> = accepted.iter().map(|&i| nodes[i].token_id).collect();
    Ok((accepted, bonus, accepted_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_tree_top1_chain() {
        // 3 depths, top-1 at each gives a clear chain of 3 nodes.
        let ids = vec![vec![10u32, 99], vec![20u32, 99], vec![30u32, 99]];
        let lps = vec![vec![-0.1f32, -10.0], vec![-0.1, -10.0], vec![-0.1, -10.0]];
        let tree = build_tree(&ids, &lps, 4);
        // Tree should be: [10, 20, 30, 99(depth1, sibling of 10)] in priority order.
        assert!(tree.iter().any(|n| n.token_id == 10 && n.depth == 1));
        assert!(tree.iter().any(|n| n.token_id == 20 && n.depth == 2));
        assert!(tree.iter().any(|n| n.token_id == 30 && n.depth == 3));
    }
}
