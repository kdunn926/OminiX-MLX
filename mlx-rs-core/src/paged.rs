//! Paged KV cache — Phase 2 scaffolding for block-based, refcounted,
//! shareable KV storage. See `OminiX-API/docs/PAGED_RADIX_ATTENTION_PLAN.md`.
//!
//! # Why
//!
//! The default [`crate::cache::KVCache`] stores K/V as one contiguous
//! `[B, H, capacity, D]` buffer per layer and regrows it by concatenation.
//! That fragments the Metal allocator at long context and forces a full
//! per-request snapshot when sharing a prefix across requests. Paging splits
//! KV into fixed-size **blocks** drawn from a shared [`PagedKvPool`]; each
//! sequence holds a [`BlockTable`] mapping its logical positions to physical
//! blocks. Shared-prefix blocks are **refcounted and read-only**, so two
//! sequences that branch off a common prefix share physical KV with no copy —
//! a diverging write copies only the first partial block (copy-on-write).
//!
//! # Status (scaffolding)
//!
//! - [`PagedKvPool`] allocator + refcounting and [`BlockTable`] logic are
//!   complete and unit-tested (pure index/refcount math, no MLX needed).
//! - [`PagedKvCache`] implements [`KeyValueCache`] with a **concat-based
//!   gather fallback** (`gather_blocks`) standing in for the future
//!   `gather_blocks` Metal kernel (mirror `metal_kernels::kv_compact`).
//! - NOT yet wired into any model's attention forward pass, and the
//!   cross-sequence shared pool / radix-tree integration on the API side is
//!   future work. Sliding-window layers must keep using the contiguous
//!   `KVCache` (see [`PagedKvCache::max_size`]); page only full-attention
//!   layers.

use std::cell::RefCell;
use std::rc::Rc;

use mlx_rs::ops::indexing::{Ellipsis, IndexMutOp, IndexOp};
use mlx_rs::ops::{concatenate_axis, zeros_dtype};
use mlx_rs::{error::Exception, Array, Dtype};

use crate::cache::KeyValueCache;

/// Default tokens per block. Matches `KVCache`'s 256-token allocation step so
/// block granularity lines up with the existing prefill chunking.
pub const DEFAULT_BLOCK_SIZE: i32 = 256;

/// Index of a physical block within a [`PagedKvPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

// ============================================================================
// PagedKvPool — physical block arena with refcounting
// ============================================================================

thread_local! {
    /// Admission budget shared across all per-layer pools on this thread: the
    /// max number of *live* blocks summed over every pool. With one private
    /// pool per layer (see [`PagedKvCache::default`]) there is no single pool to
    /// cap, so the budget is global. `None` ⇒ uncapped. The API derives it from
    /// the KV budget; exhaustion makes `alloc` return `None` → 503.
    static GLOBAL_BLOCK_CAP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Live blocks currently allocated across all pools on this thread.
    static GLOBAL_BLOCKS_USED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// An arena of fixed-size KV blocks with a free list and per-block reference
/// counts. Block storage is a single contiguous tensor per K/V — shape
/// `[num_blocks, B, H, block_size, D]` — so allocation never reserves one
/// giant per-sequence buffer and long contexts don't fragment the allocator.
/// Shared blocks (`refcount > 1`) are read-only; a write copies them first
/// (copy-on-write). One pool backs all layers of a model and is shared across
/// sequences for prefix reuse.
///
/// The arena dims/dtype are fixed on the first write; before that the pool is
/// shape-agnostic. K and V are stored separately (their head_dim may differ).
pub struct PagedKvPool {
    block_size: i32,
    /// Per-block reference count; `0` ⇒ on the free list.
    refcounts: Vec<u32>,
    /// Recycled block ids available for allocation.
    free_list: Vec<usize>,
    /// Optional hard cap on resident blocks (`None` ⇒ grow on demand).
    capacity_blocks: Option<usize>,
    /// Blocks promised to in-flight requests but not yet allocated. Counted
    /// against the cap so admission control can reserve up front (the API's
    /// KV-budget 503 gate keys off this instead of a byte estimate).
    reserved_blocks: usize,
    /// Contiguous key arena `[cap_blocks, B, H, block_size, k_d]`.
    k_arena: Option<Array>,
    /// Contiguous value arena `[cap_blocks, B, H, block_size, v_d]`.
    v_arena: Option<Array>,
    /// `(B, H, k_d, v_d)` once the arena exists.
    dims: Option<(i32, i32, i32, i32)>,
    /// `(k_dtype, v_dtype)` once the arena exists.
    dtypes: Option<(Dtype, Dtype)>,
}

impl PagedKvPool {
    /// Create an empty pool with the given block size and optional cap.
    pub fn new(block_size: i32, capacity_blocks: Option<usize>) -> Self {
        Self {
            block_size: block_size.max(1),
            refcounts: Vec::new(),
            free_list: Vec::new(),
            capacity_blocks,
            reserved_blocks: 0,
            k_arena: None,
            v_arena: None,
            dims: None,
            dtypes: None,
        }
    }

    /// Shareable handle, so one pool can back many layers/sequences.
    pub fn shared(block_size: i32, capacity_blocks: Option<usize>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::new(block_size, capacity_blocks)))
    }

    pub fn block_size(&self) -> i32 {
        self.block_size
    }

    /// `(B, H, k_head_dim, v_head_dim)` once the arena exists.
    pub fn dims(&self) -> Option<(i32, i32, i32, i32)> {
        self.dims
    }

    /// Number of blocks currently allocated (`refcount > 0`).
    pub fn used_blocks(&self) -> usize {
        self.refcounts.iter().filter(|&&r| r > 0).count()
    }

    /// Number of blocks the pool may still hand out before hitting its cap,
    /// accounting for outstanding reservations. `None` when uncapped.
    pub fn free_blocks(&self) -> Option<usize> {
        self.capacity_blocks
            .map(|cap| cap.saturating_sub(self.used_blocks() + self.reserved_blocks))
    }

    /// Blocks needed to hold `n_tokens` at this pool's block size.
    pub fn blocks_for_tokens(&self, n_tokens: i32) -> usize {
        ((n_tokens.max(0) + self.block_size - 1) / self.block_size) as usize
    }

    /// Currently outstanding reserved blocks.
    pub fn reserved_blocks(&self) -> usize {
        self.reserved_blocks
    }

    /// Admission control: reserve `n` blocks up front, returning `false`
    /// (reserving nothing) if that would exceed the cap. Always succeeds when
    /// the pool is uncapped. Pair every successful reserve with
    /// [`Self::release_reservation`] when the request finishes.
    pub fn try_reserve(&mut self, n: usize) -> bool {
        if let Some(free) = self.free_blocks() {
            if n > free {
                return false;
            }
        }
        self.reserved_blocks += n;
        true
    }

    /// Release a prior reservation (caller-tracked count).
    pub fn release_reservation(&mut self, n: usize) {
        self.reserved_blocks = self.reserved_blocks.saturating_sub(n);
    }

    /// Allocate a fresh block id (refcount 1). Storage is materialized lazily
    /// by [`Self::ensure_arena`] on first write. Returns `None` when either the
    /// per-pool cap or the thread-global block budget is exhausted.
    pub fn alloc(&mut self) -> Option<BlockId> {
        // Thread-global admission: a block going live (here or via free-list
        // reuse) must fit the global budget summed across all per-layer pools.
        let used = GLOBAL_BLOCKS_USED.with(|u| u.get());
        if let Some(cap) = GLOBAL_BLOCK_CAP.with(|c| c.get()) {
            if used >= cap {
                return None;
            }
        }
        if let Some(idx) = self.free_list.pop() {
            self.refcounts[idx] = 1;
            GLOBAL_BLOCKS_USED.with(|u| u.set(used + 1));
            return Some(BlockId(idx));
        }
        if let Some(cap) = self.capacity_blocks {
            if self.used_blocks() >= cap {
                return None;
            }
        }
        let idx = self.refcounts.len();
        self.refcounts.push(1);
        GLOBAL_BLOCKS_USED.with(|u| u.set(used + 1));
        Some(BlockId(idx))
    }

    /// Set the thread-global block-admission budget (live blocks across all
    /// per-layer pools). `None` ⇒ uncapped. Resets the live counter.
    pub fn set_global_block_cap(cap: Option<usize>) {
        GLOBAL_BLOCK_CAP.with(|c| c.set(cap));
        GLOBAL_BLOCKS_USED.with(|u| u.set(0));
    }

    /// Live blocks across all per-layer pools on this thread.
    pub fn global_blocks_used() -> usize {
        GLOBAL_BLOCKS_USED.with(|u| u.get())
    }

    /// Increment a block's refcount (a new sharer references it).
    pub fn incref(&mut self, id: BlockId) {
        self.refcounts[id.0] = self.refcounts[id.0].saturating_add(1);
    }

    /// Decrement a block's refcount; returns it to the free list at zero.
    /// The arena slot keeps its stale bytes until a later write overwrites it.
    pub fn decref(&mut self, id: BlockId) {
        let rc = &mut self.refcounts[id.0];
        if *rc == 0 {
            return;
        }
        *rc -= 1;
        if *rc == 0 {
            self.free_list.push(id.0);
            GLOBAL_BLOCKS_USED.with(|u| u.set(u.get().saturating_sub(1)));
        }
    }

    pub fn refcount(&self, id: BlockId) -> u32 {
        self.refcounts[id.0]
    }

    fn is_shared(&self, id: BlockId) -> bool {
        self.refcounts[id.0] > 1
    }

    fn k_arena_ref(&self) -> Option<&Array> {
        self.k_arena.as_ref()
    }

    fn v_arena_ref(&self) -> Option<&Array> {
        self.v_arena.as_ref()
    }

    /// Physical token capacity of the arena (`cap_blocks * block_size`), or 0
    /// before the first write.
    fn arena_tokens(&self) -> i32 {
        self.k_arena.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }

    /// Ensure the arenas exist and span at least `min_blocks` blocks.
    ///
    /// Layout is one contiguous `[B, H, cap_blocks * block_size, D]` buffer
    /// (the vLLM physical-KV layout): block `pb` owns the token range
    /// `[pb*block_size, (pb+1)*block_size)`. A sequence whose physical blocks
    /// are contiguous is then a free slice (no gather) — see
    /// [`PagedKvCache::gather`]. Grows by concatenation on the token axis.
    #[allow(clippy::too_many_arguments)]
    fn ensure_arena(
        &mut self,
        min_blocks: usize,
        b: i32,
        h: i32,
        k_d: i32,
        v_d: i32,
        k_dt: Dtype,
        v_dt: Dtype,
    ) -> Result<(), Exception> {
        let bs = self.block_size;
        let want_tokens = (min_blocks.max(1)) as i32 * bs;
        if self.k_arena.is_none() {
            self.k_arena = Some(zeros_dtype(&[b, h, want_tokens, k_d], k_dt)?);
            self.v_arena = Some(zeros_dtype(&[b, h, want_tokens, v_d], v_dt)?);
            self.dims = Some((b, h, k_d, v_d));
            self.dtypes = Some((k_dt, v_dt));
            return Ok(());
        }
        let cap = self.arena_tokens();
        if want_tokens > cap {
            let grow = want_tokens - cap;
            let kz = zeros_dtype(&[b, h, grow, k_d], k_dt)?;
            let vz = zeros_dtype(&[b, h, grow, v_d], v_dt)?;
            let k = self.k_arena.take().unwrap();
            let v = self.v_arena.take().unwrap();
            self.k_arena = Some(concatenate_axis(&[k, kz], 2)?);
            self.v_arena = Some(concatenate_axis(&[v, vz], 2)?);
        }
        Ok(())
    }

    /// Write a block-aligned segment into physical block `pb` at in-block
    /// offset `in_off` (length `seg`). The arena must already span `pb`.
    fn write_segment(&mut self, pb: usize, in_off: i32, seg: i32, k_src: &Array, v_src: &Array) {
        let lo = pb as i32 * self.block_size + in_off;
        let hi = lo + seg;
        if let Some(k) = self.k_arena.as_mut() {
            k.index_mut((Ellipsis, lo..hi, ..), k_src);
        }
        if let Some(v) = self.v_arena.as_mut() {
            v.index_mut((Ellipsis, lo..hi, ..), v_src);
        }
    }

    /// Copy physical block `src`'s tokens into block `dst` (in-arena GPU copy).
    fn copy_block(&mut self, src: usize, dst: usize) {
        let bs = self.block_size;
        let (slo, shi) = (src as i32 * bs, src as i32 * bs + bs);
        let (dlo, dhi) = (dst as i32 * bs, dst as i32 * bs + bs);
        if let Some(k) = self.k_arena.as_mut() {
            let s = k.index((Ellipsis, slo..shi, ..));
            k.index_mut((Ellipsis, dlo..dhi, ..), &s);
        }
        if let Some(v) = self.v_arena.as_mut() {
            let s = v.index((Ellipsis, slo..shi, ..));
            v.index_mut((Ellipsis, dlo..dhi, ..), &s);
        }
    }

    /// Copy-on-write: if `id` is shared, allocate a private copy of its
    /// contents, drop one reference to the original, and return the new id.
    /// If `id` is already unique, returns it unchanged. `None` on exhaustion.
    fn make_unique(&mut self, id: BlockId) -> Option<BlockId> {
        if !self.is_shared(id) {
            return Some(id);
        }
        let new_id = self.alloc()?;
        // Copy contents only if the source block was ever written (arena exists).
        if self.k_arena.is_some() {
            let (b, h, k_d, v_d) = self.dims?;
            let (k_dt, v_dt) = self.dtypes?;
            self.ensure_arena(new_id.0 + 1, b, h, k_d, v_d, k_dt, v_dt).ok()?;
            self.copy_block(id.0, new_id.0);
        }
        self.decref(id);
        Some(new_id)
    }
}

// ============================================================================
// BlockTable — per-sequence logical -> physical mapping
// ============================================================================

/// Maps a single sequence's logical token positions to physical blocks in a
/// [`PagedKvPool`]. Logical position `p` lives in `blocks[p / block_size]` at
/// in-block offset `p % block_size`.
#[derive(Clone, Default)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    block_size: i32,
    n_tokens: i32,
}

impl BlockTable {
    pub fn new(block_size: i32) -> Self {
        Self {
            blocks: Vec::new(),
            block_size: block_size.max(1),
            n_tokens: 0,
        }
    }

    pub fn n_tokens(&self) -> i32 {
        self.n_tokens
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Number of logical blocks needed to hold `tokens`.
    fn blocks_needed(&self, tokens: i32) -> usize {
        ((tokens + self.block_size - 1) / self.block_size).max(0) as usize
    }

    /// Ensure enough blocks are allocated to hold `n_tokens + extra` tokens,
    /// allocating from `pool` as needed. Returns `Err` if the pool is
    /// exhausted (caller should surface backpressure / 503).
    pub fn ensure_capacity(
        &mut self,
        pool: &mut PagedKvPool,
        extra: i32,
    ) -> Result<(), Exception> {
        let need = self.blocks_needed(self.n_tokens + extra);
        while self.blocks.len() < need {
            let id = pool.alloc().ok_or_else(|| {
                Exception::custom("PagedKvPool exhausted: no free blocks")
            })?;
            self.blocks.push(id);
        }
        Ok(())
    }

    /// Fork this table for a new sequence sharing the same prefix: clones the
    /// logical mapping and bumps every block's refcount. The fork and the
    /// original then diverge via copy-on-write on first write to a shared
    /// block. This is the prefix-sharing primitive the radix tree drives.
    pub fn fork(&self, pool: &mut PagedKvPool) -> BlockTable {
        for &id in &self.blocks {
            pool.incref(id);
        }
        self.clone()
    }

    /// Release every block back to the pool (decref). Call on drop / reset.
    pub fn release(&mut self, pool: &mut PagedKvPool) {
        for &id in &self.blocks {
            pool.decref(id);
        }
        self.blocks.clear();
        self.n_tokens = 0;
    }

    /// Logical position `p` -> (block index within `self.blocks`, in-block offset).
    fn locate(&self, p: i32) -> (usize, i32) {
        ((p / self.block_size) as usize, p % self.block_size)
    }
}

// ============================================================================
// PagedKvCache — KeyValueCache backed by a pool + block table
// ============================================================================

/// A [`KeyValueCache`] whose storage is paged through a shared [`PagedKvPool`].
///
/// `update_and_fetch`/`current_kv` gather the sequence's blocks into a
/// contiguous tensor via the `paged_gather_blocks` Metal kernel
/// ([`crate::metal_kernels::gather_blocks`]) — one dispatch per K/V.
///
/// Full-attention layers only; sliding-window layers keep the contiguous
/// `KVCache` (`max_size` returns `None` here).
pub struct PagedKvCache {
    pool: Rc<RefCell<PagedKvPool>>,
    table: BlockTable,
    block_size: i32,
    /// Cached int32 block-table device array for the kernels, rebuilt only when
    /// the block set changes (a new block is appended or copy-on-written).
    /// Avoids a host→device `Array::from_slice` on every decode step × layer.
    cached_block_table: std::cell::RefCell<Option<Array>>,
}

impl Default for PagedKvCache {
    /// Each layer gets its **own** private pool/arena. Crucially this keeps a
    /// single layer's blocks contiguous in its arena (a shared cross-layer pool
    /// interleaves layers' blocks, defeating the contiguous-slice fast path in
    /// [`PagedKvCache::gather`] and forcing an O(N²) gather during prefill).
    /// Cross-request prefix sharing still works: a stored layer cache is forked
    /// (`Clone`) within its own pool.
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }
}

impl PagedKvCache {
    /// Create a cache over its own private pool (single-sequence use).
    pub fn new(block_size: i32) -> Self {
        let pool = PagedKvPool::shared(block_size, None);
        Self::with_pool(pool, block_size)
    }

    /// Create a cache over a shared pool (cross-sequence prefix sharing).
    pub fn with_pool(pool: Rc<RefCell<PagedKvPool>>, block_size: i32) -> Self {
        Self {
            pool,
            table: BlockTable::new(block_size),
            block_size: block_size.max(1),
            cached_block_table: std::cell::RefCell::new(None),
        }
    }

    /// The sequence's int32 block-table device array, built lazily and cached
    /// until the block set changes (see invalidation in `append`).
    fn block_table_array(&self) -> Array {
        let mut cache = self.cached_block_table.borrow_mut();
        if cache.is_none() {
            let ids: Vec<i32> = self.table.blocks.iter().map(|id| id.0 as i32).collect();
            *cache = Some(Array::from_slice(&ids, &[ids.len() as i32]));
        }
        cache.as_ref().unwrap().clone()
    }

    pub fn pool(&self) -> &Rc<RefCell<PagedKvPool>> {
        &self.pool
    }

    /// Append `[B, H, n, D]` K/V (e.g. a disk-loaded prefix block) into the
    /// cache, growing it — used to preload a sequence from the disk tier.
    /// Public wrapper over the internal append.
    pub fn append_tokens(&mut self, keys: &Array, values: &Array) -> Result<(), Exception> {
        self.append(keys, values)
    }

    /// Extract each *full* block's K/V (`[B, H, block_size, D]`) in logical
    /// order, for persisting to the disk tier. Drops a trailing partial block.
    pub fn full_block_kvs(&self) -> Result<Vec<(Array, Array)>, Exception> {
        let bs = self.block_size;
        let n_full = (self.table.n_tokens / bs) as usize;
        if n_full == 0 {
            return Ok(Vec::new());
        }
        let Some((k, v)) = self.gather()? else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(n_full);
        for b in 0..n_full as i32 {
            let (lo, hi) = (b * bs, b * bs + bs);
            out.push((k.index((Ellipsis, lo..hi, ..)), v.index((Ellipsis, lo..hi, ..))));
        }
        Ok(out)
    }

    pub fn table(&self) -> &BlockTable {
        &self.table
    }

    /// Write `num_new` tokens of `keys`/`values` (`[B, H, num_new, D]`) into the
    /// sequence's tail blocks starting at the current offset, copying-on-write
    /// any shared block before mutating it. Allocates blocks as needed.
    fn append(&mut self, keys: &Array, values: &Array) -> Result<(), Exception> {
        let num_new = keys.shape()[2];
        if num_new == 0 {
            return Ok(());
        }
        let b = keys.shape()[0];
        let h = keys.shape()[1];
        let k_d = keys.shape()[3];
        let v_d = values.shape()[3];
        let k_dtype = keys.dtype();
        let v_dtype = values.dtype();

        let start = self.table.n_tokens;
        let blocks_before = self.table.blocks.len();
        let mut blocks_changed = false;
        let mut pool = self.pool.borrow_mut();
        self.table.ensure_capacity(&mut pool, num_new)?;

        // Walk the [start, start+num_new) span block-by-block, writing each
        // block-aligned segment in one index_mut.
        let mut written = 0i32;
        while written < num_new {
            let logical = start + written;
            let (blk_idx, in_off) = self.table.locate(logical);
            let room = self.block_size - in_off;
            let seg = room.min(num_new - written);

            let prev_id = self.table.blocks[blk_idx];
            // Copy-on-write before mutating a shared block.
            let block_id = pool
                .make_unique(prev_id)
                .ok_or_else(|| Exception::custom("PagedKvPool exhausted during CoW"))?;
            if block_id != prev_id {
                blocks_changed = true; // CoW remapped this block
            }
            self.table.blocks[blk_idx] = block_id;

            // Ensure the arena spans this physical block (fixes dims on first write).
            pool.ensure_arena(block_id.0 + 1, b, h, k_d, v_d, k_dtype, v_dtype)?;

            let k_src = keys.index((Ellipsis, written..written + seg, ..));
            let v_src = values.index((Ellipsis, written..written + seg, ..));
            pool.write_segment(block_id.0, in_off, seg, &k_src, &v_src);
            written += seg;
        }

        self.table.n_tokens += num_new;
        // Invalidate the cached block-table device array iff the block set
        // changed (new blocks appended or a block was copy-on-written).
        if blocks_changed || self.table.blocks.len() != blocks_before {
            *self.cached_block_table.borrow_mut() = None;
        }
        Ok(())
    }

    /// Return the sequence's logical KV as `[B, H, n_tokens, D]`.
    ///
    /// Fast path: when the sequence's physical blocks are contiguous in the
    /// arena (the common single-sequence case), this is a **free slice** of the
    /// contiguous arena — no copy. This is what keeps prefill O(N) instead of
    /// O(N²) (the per-chunk gather is gone). Non-contiguous block tables
    /// (forked / free-list-reused sequences) fall back to the
    /// `paged_gather_blocks` Metal kernel.
    fn gather(&self) -> Result<Option<(Array, Array)>, Exception> {
        if self.table.n_tokens == 0 {
            return Ok(None);
        }
        // Contiguous single-sequence case → free arena slice, no copy.
        if let Some(kv) = self.contiguous_slice()? {
            return Ok(Some(kv));
        }
        // Non-contiguous (forked / free-list-reused) → gather kernel.
        let n = self.table.n_tokens;
        let pool = self.pool.borrow();
        let (b, h, k_d, v_d) = pool
            .dims()
            .ok_or_else(|| Exception::custom("PagedKvCache::gather before any write"))?;
        let bs = pool.block_size();
        let k_arena = pool
            .k_arena_ref()
            .ok_or_else(|| Exception::custom("PagedKvCache::gather: no key arena"))?;
        let v_arena = pool
            .v_arena_ref()
            .ok_or_else(|| Exception::custom("PagedKvCache::gather: no value arena"))?;
        let seq = pool.arena_tokens();
        let table = self.block_table_array();
        let k = crate::metal_kernels::gather_blocks(k_arena, &table, n, bs, seq, b, h, k_d)?;
        let v = crate::metal_kernels::gather_blocks(v_arena, &table, n, bs, seq, b, h, v_d)?;
        Ok(Some((k, v)))
    }

    /// If the sequence's physical blocks are contiguous in the arena (the
    /// common single-sequence case), return a **free slice** `[B, Hkv, n, D]`
    /// of K and V — no copy, no gather kernel. Returns `None` when the block
    /// table is non-contiguous (forked / free-list-reused) so the caller can
    /// gather or fall back to the paged kernel. Shared by `gather` and the
    /// `try_fused_attention` contiguous decode fast-path.
    fn contiguous_slice(&self) -> Result<Option<(Array, Array)>, Exception> {
        let n = self.table.n_tokens;
        if n == 0 {
            return Ok(None);
        }
        let blocks = &self.table.blocks;
        let contiguous = blocks
            .iter()
            .enumerate()
            .all(|(i, id)| id.0 == blocks[0].0 + i);
        if !contiguous {
            return Ok(None);
        }
        let pool = self.pool.borrow();
        let bs = pool.block_size();
        let k_arena = pool
            .k_arena_ref()
            .ok_or_else(|| Exception::custom("PagedKvCache::contiguous_slice: no key arena"))?;
        let v_arena = pool
            .v_arena_ref()
            .ok_or_else(|| Exception::custom("PagedKvCache::contiguous_slice: no value arena"))?;
        let start = blocks[0].0 as i32 * bs;
        let k = k_arena.index((Ellipsis, start..start + n, ..));
        let v = v_arena.index((Ellipsis, start..start + n, ..));
        Ok(Some((k, v)))
    }
}

impl std::fmt::Debug for PagedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedKvCache")
            .field("offset", &self.table.n_tokens)
            .field("blocks", &self.table.blocks.len())
            .field("block_size", &self.block_size)
            .finish()
    }
}

impl Clone for PagedKvCache {
    /// Cloning **forks**: the new cache shares the same pool and prefix blocks,
    /// bumping their refcounts (copy-on-write on later divergent writes). This
    /// is what lets the prefix cache store a snapshot that shares blocks with
    /// the live sequence instead of deep-copying KV.
    fn clone(&self) -> Self {
        let table = {
            let mut pool = self.pool.borrow_mut();
            self.table.fork(&mut pool)
        };
        Self {
            pool: self.pool.clone(),
            table,
            block_size: self.block_size,
            cached_block_table: std::cell::RefCell::new(None),
        }
    }
}

impl Drop for PagedKvCache {
    /// Release this cache's blocks back to the pool (decref). `try_borrow_mut`
    /// guards the (single-threaded, non-reentrant) case where the pool is
    /// already borrowed; blocks would leak rather than panic.
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pool.try_borrow_mut() {
            self.table.release(&mut pool);
        }
    }
}

impl KeyValueCache for PagedKvCache {
    fn offset(&self) -> i32 {
        self.table.n_tokens
    }

    /// Full-attention only; windowed layers must use the contiguous `KVCache`.
    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        self.append(&keys, &values)?;
        self.gather()?
            .ok_or_else(|| Exception::custom("PagedKvCache: empty after append"))
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        self.gather().ok().flatten()
    }

    /// Fused decode-path attention: append the new K/V and compute the
    /// single-query output directly from the paged arena via the
    /// `paged_attention_decode` kernel — no gather. Handles only `q_len == 1`
    /// with no explicit mask (pure decode attends to all cached positions);
    /// otherwise returns `Ok(None)` so the caller gathers + runs SDPA. When it
    /// returns `Some`, it has already appended the K/V (the caller must not
    /// also call `update_and_fetch`).
    fn try_fused_attention(
        &mut self,
        q: &Array,
        k_new: Array,
        v_new: Array,
        scale: f32,
        mask: Option<&Array>,
        kv_repeat: i32,
    ) -> Result<Option<Array>, Exception> {
        if q.shape()[2] != 1 || mask.is_some() {
            return Ok(None);
        }
        // The fused decode kernel caps cached length at PAGED_DECODE_MAX_KV (its
        // s_p threadgroup buffer). Beyond that, fall back to gather+SDPA — do it
        // BEFORE appending so the caller's update_and_fetch handles the append.
        let projected = self.table.n_tokens + k_new.shape()[2];
        if projected > crate::metal_kernels::PAGED_DECODE_MAX_KV {
            return Ok(None);
        }
        self.append(&k_new, &v_new)?;

        // Contiguous-decode fast path. When the sequence's physical blocks are
        // contiguous in the arena (the common single-sequence case), a free
        // arena slice + standard SDPA is much faster than the
        // `paged_attention_decode` kernel, which is register-bound at D=256 and
        // ~never beats contiguous SDPA below ~2300 tokens (its block-table
        // indirection is pure overhead when blocks are already contiguous).
        // This recovers the regression vs the default contiguous `KVCache` for
        // single-sequence decode while keeping the paged kernel for the
        // genuinely scattered (forked / free-list-reused) case below.
        if let Some((k, v)) = self.contiguous_slice()? {
            let out = crate::utils::scaled_dot_product_attention::<crate::cache::KVCache>(
                q.clone(),
                k,
                v,
                None,
                scale,
                None,
            )?;
            return Ok(Some(out));
        }

        let b = q.shape()[0];
        let hq = q.shape()[1];
        let d = q.shape()[3];
        let hkv = if kv_repeat > 0 { hq / kv_repeat } else { k_new.shape()[1] };
        let n = self.table.n_tokens;

        let table = self.block_table_array();
        let pool = self.pool.borrow();
        let bs = pool.block_size();
        let seq = pool.arena_tokens();
        let out = crate::metal_kernels::paged_attention_decode(
            q,
            pool.k_arena_ref().ok_or_else(|| Exception::custom("no key arena"))?,
            pool.v_arena_ref().ok_or_else(|| Exception::custom("no value arena"))?,
            &table,
            scale,
            n,
            bs,
            seq,
            b,
            hq,
            hkv,
            d,
        )?;
        Ok(Some(out))
    }

    fn trim_kv(&mut self, n_drop: i32) -> Result<(), Exception> {
        if n_drop <= 0 {
            return Ok(());
        }
        let drop = n_drop.min(self.table.n_tokens);
        self.table.n_tokens -= drop;
        // Free now-unused trailing blocks.
        let need = self.table.blocks_needed(self.table.n_tokens);
        let mut pool = self.pool.borrow_mut();
        while self.table.blocks.len() > need {
            if let Some(id) = self.table.blocks.pop() {
                pool.decref(id);
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        let mut pool = self.pool.borrow_mut();
        self.table.release(&mut pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_alloc_recycles_freed_blocks() {
        let mut pool = PagedKvPool::new(256, None);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_eq!(pool.used_blocks(), 2);
        pool.decref(a);
        assert_eq!(pool.used_blocks(), 1);
        // Freed id is recycled before growing the arena.
        let c = pool.alloc().unwrap();
        assert_eq!(c, a);
        assert_eq!(pool.used_blocks(), 2);
        let _ = b;
    }

    #[test]
    fn pool_honors_capacity_cap() {
        let mut pool = PagedKvPool::new(256, Some(2));
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_some());
        assert!(pool.alloc().is_none(), "third alloc must fail at cap=2");
        assert_eq!(pool.free_blocks(), Some(0));
    }

    #[test]
    fn block_reservation_gates_against_capacity() {
        let mut pool = PagedKvPool::new(4, Some(3));
        assert_eq!(pool.blocks_for_tokens(9), 3); // ceil(9/4)
        assert!(pool.try_reserve(2), "2 of 3 blocks reservable");
        assert_eq!(pool.free_blocks(), Some(1));
        assert!(!pool.try_reserve(2), "can't reserve beyond cap");
        assert_eq!(pool.reserved_blocks(), 2, "failed reserve adds nothing");

        // A real allocation also counts against the reservation headroom.
        let _a = pool.alloc().unwrap();
        assert_eq!(pool.free_blocks(), Some(0)); // 1 used + 2 reserved = cap 3
        pool.release_reservation(2);
        assert_eq!(pool.free_blocks(), Some(2)); // 1 used, 0 reserved
    }

    #[test]
    fn global_block_cap_gates_across_pools() {
        // The budget is shared across per-layer pools (set resets the counter).
        PagedKvPool::set_global_block_cap(Some(2));
        let mut p1 = PagedKvPool::new(4, None);
        let mut p2 = PagedKvPool::new(4, None);
        let a = p1.alloc().expect("1st block");
        let _b = p2.alloc().expect("2nd block, different pool");
        assert!(p1.alloc().is_none(), "global budget exhausted across pools");
        assert_eq!(PagedKvPool::global_blocks_used(), 2);
        p1.decref(a); // frees one globally
        assert_eq!(PagedKvPool::global_blocks_used(), 1);
        assert!(p2.alloc().is_some(), "freed budget is reusable");
        PagedKvPool::set_global_block_cap(None); // reset for other tests
    }

    #[test]
    fn refcount_frees_only_at_zero() {
        let mut pool = PagedKvPool::new(256, None);
        let a = pool.alloc().unwrap();
        pool.incref(a); // refcount 2
        pool.decref(a); // -> 1, still alive
        assert_eq!(pool.used_blocks(), 1);
        assert_eq!(pool.refcount(a), 1);
        pool.decref(a); // -> 0, freed
        assert_eq!(pool.used_blocks(), 0);
    }

    #[test]
    fn block_table_capacity_and_layout() {
        let mut pool = PagedKvPool::new(4, None); // tiny blocks for the test
        let mut table = BlockTable::new(4);
        table.n_tokens = 0;
        table.ensure_capacity(&mut pool, 10).unwrap();
        assert_eq!(table.blocks().len(), 3, "10 tokens / 4 per block -> 3 blocks");
        assert_eq!(table.locate(0), (0, 0));
        assert_eq!(table.locate(5), (1, 1));
        assert_eq!(table.locate(9), (2, 1));
    }

    // ── GPU parity tests (require Metal; build deterministic K/V tensors) ──

    use crate::cache::KVCache;

    /// Deterministic `[1, h, len, d]` tensor with values offset by `base`.
    fn kv(h: i32, len: i32, d: i32, base: f32) -> Array {
        let n = (h * len * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| base + i as f32 * 0.01).collect();
        Array::from_slice(&data, &[1, h, len, d])
    }

    fn approx_eq(a: &Array, b: &Array) -> bool {
        assert_eq!(a.shape(), b.shape(), "shape mismatch");
        let close = a.all_close(b, Some(1e-5), Some(1e-6), None).unwrap();
        close.as_slice::<bool>()[0]
    }

    #[test]
    fn paged_matches_contiguous_kvcache_across_block_boundaries() {
        // Small blocks so the appends span multiple blocks.
        let block = 4;
        let (h, d) = (2, 8);
        let mut reference = KVCache::with_step(block);
        let mut paged = PagedKvCache::new(block);

        // Append several chunks of varying length; compare full K/V each step.
        let mut pos = 0.0f32;
        for &len in &[3, 1, 4, 5, 2] {
            let k = kv(h, len, d, pos);
            let v = kv(h, len, d, pos + 100.0);
            pos += 1000.0;

            let (rk, rv) = reference.update_and_fetch(k.clone(), v.clone()).unwrap();
            let (pk, pv) = paged.update_and_fetch(k, v).unwrap();

            assert_eq!(paged.offset(), reference.offset());
            assert!(approx_eq(&rk, &pk), "keys diverged at offset {}", paged.offset());
            assert!(approx_eq(&rv, &pv), "values diverged at offset {}", paged.offset());
        }
        assert_eq!(paged.offset(), 15);
    }

    #[test]
    fn paged_trim_matches_contiguous() {
        let block = 4;
        let (h, d) = (1, 4);
        let mut reference = KVCache::with_step(block);
        let mut paged = PagedKvCache::new(block);

        reference.update_and_fetch(kv(h, 10, d, 0.0), kv(h, 10, d, 50.0)).unwrap();
        paged.update_and_fetch(kv(h, 10, d, 0.0), kv(h, 10, d, 50.0)).unwrap();

        reference.trim_kv(4).unwrap();
        paged.trim_kv(4).unwrap();
        assert_eq!(paged.offset(), 6);
        assert_eq!(reference.offset(), 6);

        let (rk, _) = reference.current_kv().unwrap();
        let (pk, _) = paged.current_kv().unwrap();
        assert!(approx_eq(&rk, &pk), "keys diverged after trim");
    }

    #[test]
    fn fork_cow_isolates_divergent_writes() {
        // A forked sequence sharing a prefix block must not corrupt the parent
        // when it writes divergent tokens (copy-on-write).
        let block = 8;
        let (h, d) = (1, 4);
        let pool = PagedKvPool::shared(block, None);
        let mut parent = PagedKvCache::with_pool(pool.clone(), block);

        // Prefill a shared prefix of 5 tokens.
        let pk = kv(h, 5, d, 1.0);
        let pv = kv(h, 5, d, 9.0);
        let (parent_k0, _) = parent.update_and_fetch(pk, pv).unwrap();
        // Materialize the parent's pre-divergence snapshot so a later CoW bug
        // (child writing into the shared block) would be detectable.
        mlx_rs::transforms::eval([&parent_k0]).unwrap();

        // Fork a child sharing the parent's block table.
        let mut child = PagedKvCache::with_pool(pool.clone(), block);
        child.table = {
            let mut p = pool.borrow_mut();
            parent.table.fork(&mut p)
        };
        assert_eq!(child.offset(), 5);

        // Child writes 3 divergent tokens into the (shared) first block → CoW.
        child
            .update_and_fetch(kv(h, 3, d, 777.0), kv(h, 3, d, 888.0))
            .unwrap();

        // Parent's first-5-token K must be unchanged by the child's write.
        let (parent_k1, _) = parent.current_kv().unwrap();
        assert!(
            approx_eq(&parent_k0, &parent_k1),
            "parent prefix corrupted by child CoW write"
        );
        assert_eq!(parent.offset(), 5);
        assert_eq!(child.offset(), 8);
    }

    #[test]
    fn paged_attention_decode_matches_reference_sdpa() {
        // Cover BOTH dispatch paths: small n → single-pass (V1), large n →
        // register-light 2-pass (V2), gated at PAGED_DECODE_TWO_PASS_MIN.
        check_paged_decode(6, 4); // spans 2 blocks → V1
        check_paged_decode(1100, 256); // ≥ two-pass threshold → V2
    }

    fn check_paged_decode(n: i32, block: i32) {
        // kv_repeat = 1 (hq == hkv) so the reference can matmul head-to-head.
        let (b, hq, hkv, d) = (1, 2, 2, 4);

        let mut cache = PagedKvCache::new(block);
        let k_full = kv(hkv, n, d, 1.0);
        let v_full = kv(hkv, n, d, 50.0);
        let (kg, vg) = cache.update_and_fetch(k_full, v_full).unwrap();

        let q = kv(hq, 1, d, 7.0); // [1, hq, 1, d]
        let scale = 1.0 / (d as f32).sqrt();
        let scale_arr = Array::from_slice(&[scale], &[1]);

        // Native paged kernel (no gather).
        let out = {
            let pool = cache.pool().borrow();
            let ids: Vec<i32> = cache.table().blocks().iter().map(|x| x.0 as i32).collect();
            let table = Array::from_slice(&ids, &[ids.len() as i32]);
            let k_arena = pool.k_arena_ref().unwrap();
            let seq = k_arena.shape()[2];
            crate::metal_kernels::paged_attention_decode(
                &q,
                k_arena,
                pool.v_arena_ref().unwrap(),
                &table,
                scale,
                n,
                block,
                seq,
                b,
                hq,
                hkv,
                d,
            )
            .unwrap()
        };

        // Reference: softmax(scale * q · kᵀ) · v over the gathered cache.
        let kt = kg.transpose_axes(&[0, 1, 3, 2]).unwrap(); // [1,h,d,n]
        let scores = q.matmul(&kt).unwrap(); // [1,h,1,n]
        let scores = mlx_rs::ops::multiply(&scores, &scale_arr).unwrap();
        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None).unwrap();
        let ref_out = weights.matmul(&vg).unwrap(); // [1,h,1,d]

        assert!(
            approx_eq(&ref_out, &out),
            "paged decode kernel != reference SDPA (n={n}, block={block})"
        );
    }

    #[test]
    fn try_fused_attention_matches_gather_then_sdpa() {
        let (_b, hq, hkv, d) = (1, 2, 2, 4);
        let block = 4;
        let mut cache = PagedKvCache::new(block);
        cache
            .update_and_fetch(kv(hkv, 5, d, 1.0), kv(hkv, 5, d, 30.0))
            .unwrap(); // prefill 5

        let q = kv(hq, 1, d, 7.0);
        let scale = 1.0 / (d as f32).sqrt();
        let scale_arr = Array::from_slice(&[scale], &[1]);

        // Fused decode step appends the new K/V and returns attention directly.
        let fused = cache
            .try_fused_attention(&q, kv(hkv, 1, d, 99.0), kv(hkv, 1, d, 130.0), scale, None, 1)
            .unwrap()
            .expect("decode path should be handled");

        // Reference over the now-6-token gathered cache.
        let (kg, vg) = cache.current_kv().unwrap();
        let kt = kg.transpose_axes(&[0, 1, 3, 2]).unwrap();
        let scores = mlx_rs::ops::multiply(q.matmul(&kt).unwrap(), &scale_arr).unwrap();
        let w = mlx_rs::ops::softmax_axis(&scores, -1, None).unwrap();
        let ref_out = w.matmul(&vg).unwrap();

        assert_eq!(cache.offset(), 6, "fused path must have appended the new token");
        assert!(approx_eq(&ref_out, &fused), "fused decode != gather+SDPA");
    }

    #[test]
    fn clone_forks_blocks_and_drop_releases() {
        let mut a = PagedKvCache::new(8);
        a.update_and_fetch(kv(1, 5, 4, 1.0), kv(1, 5, 4, 2.0)).unwrap(); // 1 block
        let pool = a.pool().clone();
        let blk = a.table().blocks()[0];
        assert_eq!(pool.borrow().refcount(blk), 1);
        {
            let b = a.clone(); // fork → incref
            assert_eq!(pool.borrow().refcount(blk), 2);
            assert_eq!(b.offset(), 5, "fork shares the prefix length");
        } // b dropped → decref
        assert_eq!(pool.borrow().refcount(blk), 1, "drop must release blocks");
    }

    #[test]
    fn default_gives_each_layer_its_own_pool() {
        // Per-layer isolation keeps each layer's blocks contiguous (the shared
        // cross-layer pool interleaved them and defeated the gather fast path).
        let a = PagedKvCache::default();
        let b = PagedKvCache::default();
        assert!(!Rc::ptr_eq(a.pool(), b.pool()), "each layer gets its own pool");
    }

    #[test]
    fn gather_blocks_kernel_follows_block_table() {
        // Contiguous arena [B=1, H=1, SEQ=8, D=2]; block pb owns tokens
        // [pb*BLOCK, (pb+1)*BLOCK). Value at physical token pt encodes
        // (pb=pt/BLOCK, off=pt%BLOCK, d) so we can verify the indirection.
        let (b, h, blk, d) = (1, 1, 4, 2);
        let nb = 2;
        let seq = nb * blk; // 8
        let mut arena = vec![0f32; (b * h * seq * d) as usize];
        for pt in 0..seq {
            let (pb, off) = (pt / blk, pt % blk);
            for dd in 0..d {
                arena[(pt * d + dd) as usize] = (pb * 100 + off * 10 + dd) as f32;
            }
        }
        let pool = Array::from_slice(&arena, &[b, h, seq, d]);
        // logical block 0 -> physical 1, logical block 1 -> physical 0.
        let table = Array::from_slice(&[1i32, 0], &[2]);
        let n_tokens = 6;

        let out = crate::metal_kernels::gather_blocks(&pool, &table, n_tokens, blk, seq, b, h, d)
            .unwrap();

        // Expected: token t reads arena[table[t/blk], 0,0, t%blk, d].
        let mut expected = vec![0f32; (b * h * n_tokens * d) as usize];
        let tbl = [1i32, 0];
        for t in 0..n_tokens {
            let pb = tbl[(t / blk) as usize];
            let off = t % blk;
            for dd in 0..d {
                expected[(t * d + dd) as usize] = (pb * 100 + off * 10 + dd) as f32;
            }
        }
        let expected = Array::from_slice(&expected, &[b, h, n_tokens, d]);
        assert!(approx_eq(&expected, &out), "kernel gather diverged from block table");
    }

    #[test]
    fn fork_shares_blocks_until_cow() {
        let mut pool = PagedKvPool::new(4, None);
        let mut parent = BlockTable::new(4);
        parent.n_tokens = 4;
        parent.ensure_capacity(&mut pool, 0).unwrap();
        assert_eq!(parent.blocks().len(), 1);
        let shared = parent.blocks()[0];

        let _child = parent.fork(&mut pool);
        assert_eq!(pool.refcount(shared), 2, "fork increfs shared prefix blocks");
        assert!(pool.is_shared(shared));

        // make_unique on a shared block yields a fresh id and drops one ref.
        let unique = pool.make_unique(shared).unwrap();
        assert_ne!(unique, shared);
        assert_eq!(pool.refcount(shared), 1);
        assert_eq!(pool.refcount(unique), 1);
    }
}
