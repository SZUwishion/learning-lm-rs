use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use half::{bf16, f16};

pub struct LLamaParams<'a, T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<'a, T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<'a, T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<'a, T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<'a, T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<'a, T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<'a, T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<'a, T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<'a, T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<'a, T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<'a, T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<'a, T>, // (hidden_size, )
    pub lm_head: Tensor<'a, T>,   // (vocab_size, dim)
}

impl<'a, T> LLamaParams<'a, T> 
{
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson, device: &'a infinicore::Device) -> Self 
    where 
        T: Copy + Clone + Default + From<f32>
    {
        let get_tensor = |name: &str| -> Tensor<'a, T> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data = tensor.data();
            let data: Vec<T> = match tensor.dtype() {
                safetensors::Dtype::F32 => {
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len() / 4).to_vec() }
                },
                safetensors::Dtype::BF16 => {
                    let data: Vec<u16> = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2).to_vec() };
                    data.iter().map(|&x| T::from(f32::from(bf16::from_bits(x)))).collect()
                },
                _ => panic!("不支持的数据类型: {:?}", tensor.dtype()),
            };
            Tensor::new(data, &shape, device)
        };

        let n_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }

        let embedding_table = if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
