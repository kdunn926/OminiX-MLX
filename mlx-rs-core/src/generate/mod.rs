use std::marker::PhantomData;

use mlx_rs::{error::Exception, module::Module, Array};
use tokenizers::Tokenizer;

use crate::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    generate::generate_token::{GenerateToken, Stage},
    sampler::{DefaultSampler, Sampler},
    try_unwrap,
    ModelInput, ModelOutput,
};

mod generate_token;

// The default KV cache for generation. `KVCache` pre-allocates in fixed steps
// and writes new tokens in place; the previous default, `ConcatKeyValueCache`,
// re-`concatenate`s the whole history every decode step (O(n²) memory traffic
// over a generation), so it must not be the default for the hot path.
pub struct Generate<M, I, S = DefaultSampler, C = KVCache, T = ()> {
    tokenizer: Tokenizer,
    token_generator: GenerateToken<M, I, S, C, T>,
    max_tokens: usize,
    eos_token_ids: Vec<u32>,
    ids: Vec<u32>,
}

impl Generate<(), ()> {
    pub fn builder() -> Builder<(), (), (), ()> {
        Builder {
            tokenizer: (),
            model: (),
            model_input_marker: PhantomData,
            prompt: (),
            temp: 0.0,
            max_tokens: 256,
            eos_token_ids: Vec::new(),
            sampler: DefaultSampler,
            cache_marker: PhantomData,
            state: (),
        }
    }
}

pub struct Builder<Tok, M, I, P, S = DefaultSampler, C = KVCache, T = ()> {
    pub tokenizer: Tok,
    pub model: M,
    pub model_input_marker: PhantomData<I>,
    pub prompt: P,
    pub temp: f32,
    pub max_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub sampler: S,
    pub cache_marker: PhantomData<C>,
    pub state: T,
}

impl<Tok, M, I, P, S, C, T> Builder<Tok, M, I, P, S, C, T> {
    pub fn tokenizer(
        self,
        tokenizer: Tokenizer,
    ) -> Builder<Tokenizer, M, I, P, S, C, T> {
        Builder {
            tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn model<M2, I2>(self, model: M2) -> Builder<Tok, M2, I2, P, S, C, T>
    where
        M2: Module<I2>,
    {
        Builder {
            tokenizer: self.tokenizer,
            model,
            model_input_marker: PhantomData,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn prompt(self, prompt: Array) -> Builder<Tok, M, I, Array, S, C, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn temp(mut self, temp: f32) -> Self {
        self.temp = temp;
        self
    }

    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Token ids that terminate generation. When the sampler emits one of these
    /// it is included in the output and generation stops. Defaults to empty
    /// (generation runs to `max_tokens`).
    pub fn eos_token_ids(mut self, eos_token_ids: Vec<u32>) -> Self {
        self.eos_token_ids = eos_token_ids;
        self
    }

    pub fn sampler<S2>(self, sampler: S2) -> Builder<Tok, M, I, P, S2, C, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler,
            cache_marker: self.cache_marker,
            state: self.state,
        }
    }

    pub fn cache_marker<C2>(self) -> Builder<Tok, M, I, P, S, C2, T> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler: self.sampler,
            cache_marker: PhantomData,
            state: self.state,
        }
    }

    pub fn state<T2>(self, state: T2) -> Builder<Tok, M, I, P, S, C, T2> {
        Builder {
            tokenizer: self.tokenizer,
            model: self.model,
            model_input_marker: self.model_input_marker,
            prompt: self.prompt,
            temp: self.temp,
            max_tokens: self.max_tokens,
            eos_token_ids: self.eos_token_ids,
            sampler: self.sampler,
            cache_marker: self.cache_marker,
            state,
        }
    }
}

impl<M, I, S, C, T> Builder<Tokenizer, M, I, Array, S, C, T>
where
    M: Module<I>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    pub fn build(self) -> Generate<M, I, S, C, T> {
        let Self {
            tokenizer,
            model,
            model_input_marker: _,
            prompt,
            temp,
            sampler,
            cache_marker: _,
            state,
            max_tokens,
            eos_token_ids,
        } = self;

        let stage = Stage::Prefill { prompt, state };

        let token_generator = GenerateToken {
            model,
            model_input_marker: PhantomData,
            sampler,
            temp,
            stage,
        };

        let ids = Vec::with_capacity(max_tokens);
        Generate {
            tokenizer,
            token_generator,
            max_tokens,
            eos_token_ids,
            ids,
        }
    }
}

impl<M, I, S, C, T> Generate<M, I, S, C, T> {
    /// Name of the KV cache type this generator is parameterized with. Used by
    /// a regression test to assert the default is the pre-allocating cache.
    #[cfg(test)]
    fn cache_type_name() -> &'static str {
        std::any::type_name::<C>()
    }
}

pub struct Response {
    pub text: String,
    pub ids: Vec<u32>,
}

impl<M, I, S, C, T> Iterator for Generate<M, I, S, C, T>
where
    M: Module<I>,
    M::Error: Into<Exception>,
    M::Output: ModelOutput,
    for<'input> I: ModelInput<'input, C, T>,
    S: Sampler,
    C: KeyValueCache + Default,
{
    type Item = Result<Response, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = try_unwrap!(self.token_generator.next()?);
            let id = try_unwrap!(token.try_item());
            self.ids.push(id);

            let hit_eos = self.eos_token_ids.contains(&id);
            if hit_eos || self.ids.len() >= self.max_tokens {
                let text = try_unwrap!(self.tokenizer.decode(&self.ids, true));
                let mut ids = Vec::with_capacity(self.max_tokens);
                std::mem::swap(&mut self.ids, &mut ids);
                return Some(Ok(Response { text, ids }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KVCache;
    use crate::generate::generate_token::{GenerateToken, Stage};
    use crate::{ModelInput, ModelInputBuilder};
    use mlx_rs::macros::ModuleParameters;
    use mlx_rs::module::Module;
    use mlx_rs::{Array, Dtype};
    use tokenizers::models::wordlevel::WordLevel;

    const VOCAB: i32 = 8;

    // Owned input: clones the fed token(s) and ignores the cache, so it
    // satisfies `for<'input> ModelInput<'input, C, T>` (a borrowing input
    // cannot). Sufficient to exercise the loop's control flow.
    struct MockInput {
        y: Array,
    }

    impl<'a, C, T> ModelInput<'a, C, T> for MockInput {
        fn from_model_input_builder(b: ModelInputBuilder<'a, C, T>) -> Self {
            MockInput { y: b.y.clone() }
        }
    }

    #[derive(Debug, ModuleParameters)]
    struct MockModel {}

    // Emits one-hot logits [B, T, VOCAB] peaked at (tok + 1) % VOCAB, so the
    // greedy next token after any position `p` is deterministically known.
    impl Module<MockInput> for MockModel {
        type Output = Array;
        type Error = mlx_rs::error::Exception;

        fn forward(&mut self, input: MockInput) -> Result<Array, Self::Error> {
            let shape = input.y.shape().to_vec();
            let (b, t) = (shape[0], shape[1]);
            let ids = input.y.as_dtype(Dtype::Int32)?;
            let ids = ids.as_slice::<i32>();
            let v = VOCAB as usize;
            let mut data = vec![0f32; ids.len() * v];
            for (i, &tok) in ids.iter().enumerate() {
                let target = (tok + 1).rem_euclid(VOCAB) as usize;
                data[i * v + target] = 10.0;
            }
            Ok(Array::from_slice(&data, &[b, t, VOCAB]))
        }

        fn training_mode(&mut self, _mode: bool) {}
    }

    type Gt = GenerateToken<MockModel, MockInput, DefaultSampler, KVCache, ()>;

    fn first_token(gt: &mut Gt) -> i32 {
        let arr = gt.next().unwrap().unwrap();
        arr.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>()[0]
    }

    // F2: prefill must sample from the LAST prompt position, not the first.
    #[test]
    fn prefill_samples_last_position() {
        let prompt = Array::from_slice(&[3i32, 5i32], &[1, 2]); // [B=1, T=2]
        let mut gt: Gt = GenerateToken {
            model: MockModel {},
            model_input_marker: PhantomData,
            sampler: DefaultSampler,
            temp: 0.0,
            stage: Stage::Prefill { prompt, state: () },
        };
        // (5 + 1) % 8 == 6 (last position), NOT (3 + 1) == 4 (first position).
        assert_eq!(first_token(&mut gt), 6);
    }

    // F8: the generic generator must default to the pre-allocating KVCache, not
    // the O(n²) ConcatKeyValueCache. Guards against a silent regression.
    #[test]
    fn default_cache_is_preallocating_kvcache() {
        // Relies on the default C type param of Generate<M, I>.
        let name = Generate::<MockModel, MockInput>::cache_type_name();
        assert!(
            name.contains("KVCache") && !name.contains("Concat"),
            "default Generate cache type is {name}",
        );
    }

    fn toy_tokenizer() -> Tokenizer {
        // Minimal word-level vocab "0".."7" so Response decoding doesn't panic.
        // `.collect()` infers the builder's AHashMap type from the method arg.
        let wl = WordLevel::builder()
            .vocab((0..VOCAB).map(|i| (i.to_string(), i as u32)).collect())
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        Tokenizer::new(wl)
    }

    // F5: generation stops when an EOS id is emitted (and includes it), rather
    // than always running to max_tokens.
    #[test]
    fn stops_on_eos() {
        // Greedy chain from prompt [.., 5]: 6 -> 7 -> 0 -> 1 ...
        let prompt = Array::from_slice(&[5i32], &[1, 1]);
        let mut gen = Generate::builder()
            .tokenizer(toy_tokenizer())
            .model(MockModel {})
            .prompt(prompt)
            .max_tokens(50)
            .eos_token_ids(vec![0])
            .build();
        // The generator is unbounded (resumes after each Response), so take the
        // first terminal Response only.
        let resp = gen.next().expect("a response").unwrap();
        assert_eq!(resp.ids, vec![6, 7, 0]); // stopped at EOS=0, not 50 tokens
    }
}
