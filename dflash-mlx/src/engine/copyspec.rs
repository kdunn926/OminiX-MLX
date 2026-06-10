//! Prompt-tail N-gram lookup for free drafts.
//!
//! Port of `dflash_mlx.engine.copyspec.CopySpecIndex`. When generation
//! re-emits a 6-token window that already appeared in the prompt + history,
//! we propose the next `block_len - 1` prompt tokens as the draft block,
//! skipping the draft model forward entirely. The target still verifies
//! every proposed token before commit, so this is lossless.
//!
//! Matches Python's API and FNV-1a hashing semantics.

use std::collections::{hash_map::Entry, HashMap};

pub const COPYSPEC_WINDOW_SIZE: usize = 6;

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

#[derive(Debug, Clone)]
enum Positions {
    One(usize),
    Many(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct CopySpecIndex {
    window_size: usize,
    tokens: Vec<u32>,
    /// hash(tokens[start..start+window_size]) -> position(s) of the END
    /// (= start + window_size) so `tokens[pos..]` is the continuation
    /// available after a match.
    index: HashMap<u64, Positions>,
    enabled: bool,
}

impl CopySpecIndex {
    pub fn new(prompt_tokens: &[u32]) -> Self {
        Self::with_window(prompt_tokens, COPYSPEC_WINDOW_SIZE)
    }

    pub fn with_window(prompt_tokens: &[u32], window_size: usize) -> Self {
        assert!(window_size > 0, "copyspec window_size must be positive");
        let tokens = prompt_tokens.to_vec();
        let enabled = tokens.len() > window_size;
        let mut idx = Self {
            window_size,
            tokens,
            index: HashMap::new(),
            enabled,
        };
        idx.build();
        idx
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Try to draft the next `max_tokens` tokens from the prompt tail.
    /// Returns `Some(tokens)` only when the trailing window (last
    /// `window_size - 1` committed tokens + `staged_first`) appears in the
    /// indexed prompt AND there are at least `max_tokens` tokens available
    /// after that match. Otherwise `None` (fall back to the draft model).
    pub fn draft_after(
        &self,
        staged_first: u32,
        max_tokens: usize,
        forbidden: Option<&[u32]>,
    ) -> Option<Vec<u32>> {
        if !self.enabled || max_tokens == 0 {
            return None;
        }
        let window = self.tail_window(staged_first)?;
        let h = hash_tokens(&window);
        let mut best_pos: Option<usize> = None;
        let mut best_available = 0usize;
        if let Some(positions) = self.index.get(&h) {
            let iter: Box<dyn Iterator<Item = usize>> = match positions {
                Positions::One(v) => Box::new(std::iter::once(*v)),
                Positions::Many(list) => Box::new(list.iter().copied()),
            };
            for source_pos in iter {
                if !self.matches_window(source_pos, &window) {
                    continue;
                }
                let available = self.tokens.len().saturating_sub(source_pos);
                if available > best_available {
                    best_pos = Some(source_pos);
                    best_available = available;
                }
            }
        }
        let pos = best_pos?;
        if best_available < max_tokens {
            return None;
        }
        let copied: Vec<u32> = self.tokens[pos..pos + max_tokens].to_vec();
        if let Some(forbidden_list) = forbidden {
            if copied.iter().any(|t| forbidden_list.contains(t)) {
                return None;
            }
        }
        Some(copied)
    }

    pub fn append_committed(&mut self, token_ids: &[u32]) {
        for &t in token_ids {
            self.tokens.push(t);
            if self.enabled {
                self.index_latest_window();
            } else if self.tokens.len() > self.window_size {
                // Crossed the threshold — enable and index everything.
                self.enabled = true;
                self.build();
            }
        }
    }

    fn build(&mut self) {
        self.index.clear();
        if !self.enabled || self.tokens.len() < self.window_size {
            return;
        }
        let n = self.tokens.len() - self.window_size + 1;
        for start in 0..n {
            self.index_window(start);
        }
    }

    fn index_latest_window(&mut self) {
        if self.tokens.len() < self.window_size {
            return;
        }
        let start = self.tokens.len() - self.window_size;
        self.index_window(start);
    }

    fn index_window(&mut self, start: usize) {
        let end = start + self.window_size;
        let h = hash_tokens(&self.tokens[start..end]);
        match self.index.entry(h) {
            Entry::Vacant(v) => {
                v.insert(Positions::One(end));
            }
            Entry::Occupied(mut o) => {
                let new_val = match o.get_mut() {
                    Positions::One(prev) => Some(Positions::Many(vec![*prev, end])),
                    Positions::Many(list) => {
                        list.push(end);
                        None
                    }
                };
                if let Some(v) = new_val {
                    o.insert(v);
                }
            }
        }
    }

    fn tail_window(&self, staged_first: u32) -> Option<Vec<u32>> {
        if self.tokens.len() + 1 < self.window_size {
            return None;
        }
        if self.window_size == 1 {
            return Some(vec![staged_first]);
        }
        let tail_len = self.window_size - 1;
        let tail_start = self.tokens.len() - tail_len;
        let mut w = self.tokens[tail_start..].to_vec();
        w.push(staged_first);
        Some(w)
    }

    fn matches_window(&self, source_pos: usize, window: &[u32]) -> bool {
        if source_pos < self.window_size {
            return false;
        }
        if source_pos > self.tokens.len() {
            return false;
        }
        let start = source_pos - self.window_size;
        &self.tokens[start..source_pos] == window
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut value: u64 = FNV_OFFSET_BASIS;
    for &t in tokens {
        value ^= t as u64;
        value = value.wrapping_mul(FNV_PRIME);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_on_short_prompt() {
        let idx = CopySpecIndex::new(&[1, 2, 3]);
        assert!(!idx.is_enabled());
        assert_eq!(idx.draft_after(4, 3, None), None);
    }

    #[test]
    fn draft_after_match() {
        // Prompt: 1 2 3 4 5 6 | 7 8 9 10 11 12 — second half follows the
        // window [1..7). Committed history then re-emits [1..6); asking
        // for staged=6 should propose [7,8,9,...].
        let mut idx = CopySpecIndex::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        // Re-emit the prefix to align the tail.
        idx.append_committed(&[1, 2, 3, 4, 5]);
        // Tail window before staged: last (window-1=5) committed = [1,2,3,4,5].
        // staged=6 → window = [1,2,3,4,5,6], which matches prompt[0..6]. End=6.
        // Continuation tokens 7..len = [7,8,9,10,11,12], so 6 available.
        let drafted = idx.draft_after(6, 4, None).expect("expected match");
        assert_eq!(drafted, vec![7, 8, 9, 10]);
    }

    #[test]
    fn no_match_when_window_unseen() {
        let mut idx = CopySpecIndex::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        idx.append_committed(&[100, 101, 102, 103, 104]);
        assert_eq!(idx.draft_after(105, 3, None), None);
    }

    #[test]
    fn fnv_hash_matches_reference() {
        // FNV-1a over [1,2,3] in u64 with offset basis 0xcbf29ce484222325,
        // prime 0x100000001b3. Pre-computed reference:
        //   v = 0xcbf29ce484222325
        //   v ^= 1; v *= 0x100000001b3
        //   v ^= 2; v *= 0x100000001b3
        //   v ^= 3; v *= 0x100000001b3
        let h = hash_tokens(&[1, 2, 3]);
        // Hardcoded literal (verified against an independent FNV-1a
        // implementation) — recomputing it with the same loop here made
        // the test tautological.
        assert_eq!(h, 0xd0aa6218672cf5ab);
    }
}
