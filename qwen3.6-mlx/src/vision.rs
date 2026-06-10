use std::collections::HashMap;

use image::DynamicImage;
use mlx_rs::{
    array,
    module::{Module, Param},
    nn, Array,
};
use mlx_rs_core::error::Error;

use crate::config::VisionConfig;

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn apply_gelu(x: Array) -> Result<Array, Error> {
    Ok(nn::gelu_approximate(x)?)
}

/// PatchEmbed: Conv3d to extract image patches.
/// Weight: [out_channels, temporal, H_patch, W_patch, in_channels]
pub struct PatchEmbed {
    pub proj: nn::Conv3d,
}

impl PatchEmbed {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let out = self.proj.forward(x)?;
        let shape = out.shape();
        let n = shape[2] * shape[3];
        Ok(out.reshape(&[n, shape[4]])?)
    }
}

pub struct VisionLayerNorm {
    pub weight: Array,
    pub bias: Array,
    pub eps: f32,
}

impl VisionLayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array, Error> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )?)
    }
}

pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub qkv: nn::Linear,
    pub proj: nn::Linear,
}

impl VisionAttention {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let shape = x.shape();
        let n = shape[0];
        let hidden = shape[1];

        let qkv = self.qkv.forward(x)?;
        let parts = qkv.split(3, -1)?;
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        let q = q
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(array!(self.scale))?;
        let attn = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let out = attn
            .matmul(&v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[n, hidden])?;

        Ok(self.proj.forward(&out)?)
    }
}

pub struct VisionMlp {
    pub linear_fc1: nn::Linear,
    pub linear_fc2: nn::Linear,
}

impl VisionMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let h = apply_gelu(self.linear_fc1.forward(x)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

pub struct VisionBlock {
    pub norm1: VisionLayerNorm,
    pub norm2: VisionLayerNorm,
    pub attn: VisionAttention,
    pub mlp: VisionMlp,
}

impl VisionBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let h = x.add(&attn_out)?;
        let normed2 = self.norm2.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(h.add(&mlp_out)?)
    }
}

pub struct DeepStackMerger {
    pub norm: VisionLayerNorm,
    pub linear_fc1: nn::Linear,
    pub linear_fc2: nn::Linear,
}

impl DeepStackMerger {
    pub fn forward(&mut self, x: &Array, spatial_merge_size: i32) -> Result<Array, Error> {
        let shape = x.shape();
        let n = shape[0];
        let d = shape[1];
        let m = spatial_merge_size * spatial_merge_size;
        let merged = x.reshape(&[n / m, m * d])?;
        let normed = self.norm.forward(&merged)?;
        let h = apply_gelu(self.linear_fc1.forward(&normed)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

pub struct PatchMerger {
    pub norm: VisionLayerNorm,
    pub linear_fc1: nn::Linear,
    pub linear_fc2: nn::Linear,
}

impl PatchMerger {
    pub fn forward(&mut self, x: &Array, spatial_merge_size: i32) -> Result<Array, Error> {
        let normed = self.norm.forward(x)?;
        let shape = normed.shape();
        let n = shape[0];
        let d = shape[1];
        let m = spatial_merge_size * spatial_merge_size;
        let merged = normed.reshape(&[n / m, m * d])?;
        let h = apply_gelu(self.linear_fc1.forward(&merged)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

pub struct VisionTower {
    pub patch_embed: PatchEmbed,
    pub pos_embed: nn::Embedding,
    pub blocks: Vec<VisionBlock>,
    pub deepstack_mergers: Vec<DeepStackMerger>,
    pub merger: PatchMerger,
    pub config: VisionConfig,
}

impl VisionTower {
    pub fn forward(
        &mut self,
        pixels: &Array,
        h_patches: i32,
        w_patches: i32,
    ) -> Result<Array, Error> {
        let mut hidden = self.patch_embed.forward(pixels)?;

        let n = h_patches * w_patches;
        let pos_ids: Vec<i32> = (0..n).collect();
        let pos_ids_arr = Array::from_slice(&pos_ids, &[n]);
        let pos_emb = self.pos_embed.forward(&pos_ids_arr)?;
        hidden = hidden.add(&pos_emb)?;

        let spatial_merge = self.config.spatial_merge_size;
        let mut deepstack_idx = 0usize;
        let mut deepstack_outputs: Vec<Array> = Vec::new();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            hidden = block.forward(&hidden)?;
            if deepstack_idx < self.config.deepstack_visual_indexes.len()
                && i == self.config.deepstack_visual_indexes[deepstack_idx]
            {
                let ds_out =
                    self.deepstack_mergers[deepstack_idx].forward(&hidden, spatial_merge)?;
                deepstack_outputs.push(ds_out);
                deepstack_idx += 1;
            }
        }

        let final_out = self.merger.forward(&hidden, spatial_merge)?;
        let mut result = final_out;
        for ds in deepstack_outputs {
            result = result.add(&ds)?;
        }
        Ok(result)
    }
}

fn load_vision_layer_norm(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<VisionLayerNorm, Error> {
    Ok(VisionLayerNorm {
        weight: get_weight(weights, &format!("{}.weight", prefix))?,
        bias: get_weight(weights, &format!("{}.bias", prefix))?,
        eps: 1e-6,
    })
}

fn load_vision_linear(weights: &HashMap<String, Array>, prefix: &str) -> Result<nn::Linear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let bias = get_weight(weights, &format!("{}.bias", prefix))?;
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(Some(bias)),
    })
}

pub fn load_vision_tower(
    weights: &HashMap<String, Array>,
    config: &VisionConfig,
) -> Result<VisionTower, Error> {
    let vp = "vision_tower";

    let conv_weight = get_weight(weights, &format!("{}.patch_embed.proj.weight", vp))?;
    let conv_bias = get_weight(weights, &format!("{}.patch_embed.proj.bias", vp))?;
    let patch_embed = PatchEmbed {
        proj: nn::Conv3d {
            weight: Param::new(conv_weight),
            bias: Param::new(Some(conv_bias)),
            stride: (
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
        },
    };

    let pos_embed = nn::Embedding {
        weight: Param::new(get_weight(weights, &format!("{}.pos_embed.weight", vp))?),
    };

    let head_dim = config.hidden_size / config.num_heads;
    let mut blocks = Vec::with_capacity(config.depth);
    for i in 0..config.depth {
        let bp = format!("{}.blocks.{}", vp, i);

        let qkv_weight = get_weight(weights, &format!("{}.attn.qkv.weight", bp))?;
        let qkv_bias = get_weight(weights, &format!("{}.attn.qkv.bias", bp))?;
        let proj_weight = get_weight(weights, &format!("{}.attn.proj.weight", bp))?;
        let proj_bias = get_weight(weights, &format!("{}.attn.proj.bias", bp))?;

        let attn = VisionAttention {
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            qkv: nn::Linear {
                weight: Param::new(qkv_weight),
                bias: Param::new(Some(qkv_bias)),
            },
            proj: nn::Linear {
                weight: Param::new(proj_weight),
                bias: Param::new(Some(proj_bias)),
            },
        };

        let mlp = VisionMlp {
            linear_fc1: load_vision_linear(weights, &format!("{}.mlp.linear_fc1", bp))?,
            linear_fc2: load_vision_linear(weights, &format!("{}.mlp.linear_fc2", bp))?,
        };

        blocks.push(VisionBlock {
            norm1: load_vision_layer_norm(weights, &format!("{}.norm1", bp))?,
            norm2: load_vision_layer_norm(weights, &format!("{}.norm2", bp))?,
            attn,
            mlp,
        });
    }

    let mut deepstack_mergers = Vec::with_capacity(config.deepstack_visual_indexes.len());
    for i in 0..config.deepstack_visual_indexes.len() {
        let dp = format!("{}.deepstack_merger_list.{}", vp, i);
        deepstack_mergers.push(DeepStackMerger {
            norm: load_vision_layer_norm(weights, &format!("{}.norm", dp))?,
            linear_fc1: load_vision_linear(weights, &format!("{}.linear_fc1", dp))?,
            linear_fc2: load_vision_linear(weights, &format!("{}.linear_fc2", dp))?,
        });
    }

    let mp = format!("{}.merger", vp);
    let merger = PatchMerger {
        norm: load_vision_layer_norm(weights, &format!("{}.norm", mp))?,
        linear_fc1: load_vision_linear(weights, &format!("{}.linear_fc1", mp))?,
        linear_fc2: load_vision_linear(weights, &format!("{}.linear_fc2", mp))?,
    };

    Ok(VisionTower {
        patch_embed,
        pos_embed,
        blocks,
        deepstack_mergers,
        merger,
        config: config.clone(),
    })
}

pub fn preprocess_image(
    img: &DynamicImage,
    patch_size: i32,
    temporal_patch_size: i32,
) -> Result<(Array, i32, i32), Error> {
    let target = 448i32;
    let img = img.resize_exact(
        target as u32,
        target as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let rgb = img.to_rgb8();

    let h = target;
    let w = target;
    let h_patches = h / patch_size;
    let w_patches = w / patch_size;

    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
        .collect();

    let frame_pixels = pixels.len();
    let mut temporal_pixels = Vec::with_capacity(temporal_patch_size as usize * frame_pixels);
    for _ in 0..temporal_patch_size {
        temporal_pixels.extend_from_slice(&pixels);
    }

    let arr = Array::from_slice(&temporal_pixels, &[1, temporal_patch_size, h, w, 3]);
    Ok((arr, h_patches, w_patches))
}
