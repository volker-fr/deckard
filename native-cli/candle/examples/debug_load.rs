fn main() {
    let config = std::env::args().nth(1).expect("config");
    let weights = std::env::args().nth(2).expect("weights");
    let handle = aih_candle_wire::load_for_debug(config, weights);
    println!("{handle:?}");
}

mod aih_candle_wire {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_transformers::models::debertav2::{Config, DebertaV2SeqClassificationModel};

    pub fn load_for_debug(config_path: String, weights_path: String) -> Option<String> {
        let config_text = std::fs::read(config_path).ok()?;
        let config: Config = serde_json::from_slice(&config_text).ok()?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[std::path::Path::new(&weights_path)], DType::F32, &device)
        }
        .expect("mmap");
        match DebertaV2SeqClassificationModel::load(vb.pp("deberta"), &config, None) {
            Ok(model) => {
                use candle_core::Tensor;
                let ids = Vec::from([1u32, 1, 2]);
                let mask = Vec::from([1.0f32; 3]);
                let seq = ids.len();
                let input_ids = Tensor::from_vec(ids, (1, seq), &device).unwrap();
                let attention_mask =
                    Tensor::from_vec(mask, (1, seq), &device).unwrap();
                match model.forward(&input_ids, None, Some(attention_mask)) {
                    Ok(t) => Some(format!(
                        "forward ok {:?}",
                        t.flatten_all()
                            .and_then(|t| t.squeeze(0))
                            .and_then(|t| t.to_scalar::<f32>())
                    )),
                    Err(e) => Some(format!("forward err: {e:?}")),
                }
            }
            Err(e) => Some(format!("{e:?}")),
        }
    }
}