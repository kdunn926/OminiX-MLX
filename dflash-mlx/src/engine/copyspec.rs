use fnv::FnvHasher;
use std::collections::HashMap;
use std::hash::Hasher;

const WINDOW: usize = 6;

/// FNV-based 6-token sliding window index over the prompt.
/// Returns the starting offset in the prompt if staged_first begins a contiguous
/// match, or None.
#[derive(Debug, Clone)]
pub struct CopySpecIndex {
    index: HashMap<u64, usize>,
    tokens: Vec<u32>,
}

impl CopySpecIndex {
    pub fn build(tokens: &[u32]) -> Self {
        let mut index = HashMap::new();
        if tokens.len() >= WINDOW + 1 {
            for start in 0..=(tokens.len() - WINDOW - 1) {
                index
                    .entry(hash_window(&tokens[start..start + WINDOW]))
                    .or_insert(start);
            }
        }
        Self {
            index,
            tokens: tokens.to_vec(),
        }
    }

    pub fn find_continuation(&self, staged_first: u32) -> Option<usize> {
        if self.tokens.len() < WINDOW {
            return None;
        }
        let suffix = &self.tokens[self.tokens.len() - WINDOW..];
        let start = *self.index.get(&hash_window(suffix))?;
        let continuation = start + WINDOW;
        self.tokens
            .get(continuation)
            .copied()
            .filter(|candidate| *candidate == staged_first)
            .map(|_| continuation)
    }
}

fn hash_window(tokens: &[u32]) -> u64 {
    let mut hasher = FnvHasher::default();
    for token in tokens {
        hasher.write_u32(*token);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_continuation_from_suffix_window() {
        let index = CopySpecIndex::build(&[1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8]);
        assert_eq!(index.find_continuation(3), Some(8));
        assert_eq!(index.find_continuation(9), None);
    }
}
