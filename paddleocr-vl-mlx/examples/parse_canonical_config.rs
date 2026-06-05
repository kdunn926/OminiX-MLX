use paddleocr_vl_mlx::PaddleOcrVlConfig;

fn main() {
    let json = std::fs::read_to_string("/tmp/test_paddleocr_config.json").unwrap();
    let cfg: PaddleOcrVlConfig = serde_json::from_str(&json).unwrap();
    cfg.validate().unwrap();
    println!("OK — model_type={} hidden={} layers={} kv_heads={} head_dim={}",
        cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers,
        cfg.num_key_value_heads, cfg.head_dim);
    println!("MROPE: section={:?} theta={}",
        cfg.rope_scaling.mrope_section, cfg.rope_theta);
    println!("vision: hidden={} layers={} patch={} merge={}",
        cfg.vision_config.hidden_size, cfg.vision_config.num_hidden_layers,
        cfg.vision_config.patch_size, cfg.vision_config.spatial_merge_size);
}
