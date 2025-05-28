use once_cell::sync::Lazy;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use half::f16;
use safetensors::SafeTensors;
use std::path::Path;

use infinicore::DeviceType::CPU as CPU;
use infinicore::DeviceType::CUDA as CUDA;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl<
        T: Clone
            + Copy
            + Default
            + Into<f32>
            + From<f32>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + std::cmp::PartialOrd
            + std::fmt::Debug,
    > Llama<T>
{
    pub fn from_safetensors(model_dir: impl AsRef<Path>, device: &infinicore::Device) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config, device);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self, device: &infinicore::Device) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0, device)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>, device: &infinicore::Device, stream: &infinicore::Stream) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d], device);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d], device);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv], device);
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len], device);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di], device);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di], device);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table, device, stream);

        residual.sync_data(CPU, device.get(), device, stream);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
                device,
                stream
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len, device); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len, device); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0, device, stream);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0, device, stream);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0, device, stream);

            q.sync_data(device.get(), CPU, device, stream);
            k.sync_data(device.get(), CPU, device, stream);
            v.sync_data(device.get(), CPU, device, stream);

            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
                device,
                stream
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
                device,
                stream
            );

            q.sync_data(device.get(), CPU, device, stream);
            k.sync_data(device.get(), CPU, device, stream);

            let full_k = &mut cache.k_cache(layer, 0, device); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0, device); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
                device,
                stream
            );

            hidden_states.sync_data(CPU, device.get(), device, stream);

            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
                device,
                stream
            );

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
                device,
                stream
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab], device);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d], device);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1, self.d], device);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
            device,
            stream
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0, device, stream);
        
        logits.sync_data(device.get(), CPU, device, stream);

        // 创建一个新的Tensor，形状为[self.vocab]
        Tensor::<T>::new(logits.data().to_vec(), &vec![self.vocab], device)
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        device: &infinicore::Device,
        stream: &infinicore::Stream,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();

        // 复制输入token
        result.extend_from_slice(token_ids);

        // 初始化KV缓存
        let mut cache = self.new_cache(device);

        // 处理输入序列
        let input_tensor = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()], device);

        // 获取输入序列的logits
        let mut logits = self.forward(&input_tensor, &mut cache, device, &stream);

        for _ in 0..max_len {
            // 采样下一个token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature, device, &stream);
            result.push(next_token);

            // 检查是否生成了结束符
            if next_token == self.eos_token_id {
                break;
            }

            // 创建单个token的输入张量
            let next_input = Tensor::new(vec![next_token], &vec![1], device);

            // 获取下一个logits
            logits = self.forward(&next_input, &mut cache, device, &stream);
        }

        result
    }

    pub fn chat(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        device: &infinicore::Device,
        stream: &infinicore::Stream,
    ) {
        println!("Enter your message(type 'exit' to quit):");
        let mut input = String::new();
        loop {
            std::io::stdin().read_line(&mut input).unwrap();
            if input.trim() == "exit" {
                break;
            }
            let binding = tokenizer.encode(input.clone(), true).unwrap();
            let input_ids = binding.get_ids();
            let mut result = self.generate(&input_ids, max_len, top_p, top_k, temperature, device, &stream);
            stream.synchronize();
            println!("{}", tokenizer.decode(&result, true).unwrap());
        }
    }
}

pub fn self_attention<T: Copy + Clone + Default + Into<f32> + std::fmt::Debug>(
    hidden_states: &mut Tensor<T>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<T>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
    device: &infinicore::Device,
    stream: &infinicore::Stream,
) {
    let mut att_output = Tensor::<T>::default(&vec![seq_len, n_kv_h, n_groups, dqkv], device);

    // 对每个KV头和组计算注意力并应用
    for h in 0..n_kv_h {
        for g in 0..n_groups {
            // 1. 计算QK注意力分数
            // 修改q的slice方式：为每个位置收集属于当前head和group的向量
            let mut q_data = Vec::with_capacity(seq_len * dqkv);
            for s in 0..seq_len {
                let start = s * (n_kv_h * n_groups * dqkv) + (h * n_groups + g) * dqkv;
                q_data.extend_from_slice(&q.data()[start..start + dqkv]);
            }
            let q_slice = Tensor::new(q_data, &vec![seq_len, dqkv], device);

            // 修改k的slice方式：为每个位置收集属于当前head的向量
            let mut k_data = Vec::with_capacity(total_seq_len * dqkv);
            for s in 0..total_seq_len {
                let start = s * (n_kv_h * dqkv) + h * dqkv;
                k_data.extend_from_slice(&k.data()[start..start + dqkv]);
            }
            let k_slice = Tensor::new(k_data, &vec![total_seq_len, dqkv], device);

            let mut score_slice = att_scores.slice(
                (h * n_groups + g) * seq_len * total_seq_len,
                &vec![seq_len, total_seq_len],
                device
            );
            OP::matmul_transb(
                &mut score_slice,
                0.0,
                &q_slice,
                &k_slice,
                1.0 / (dqkv as f32).sqrt(),
                device,
                stream
            );

            // 2. 应用因果掩码和softmax
            OP::causal_softmax(&mut score_slice, device, stream);

            // 3. 计算注意力输出
            // 修改v的slice方式：与k相同
            let mut v_data = Vec::with_capacity(total_seq_len * dqkv);
            for s in 0..total_seq_len {
                let start = s * (n_kv_h * dqkv) + h * dqkv;
                v_data.extend_from_slice(&v.data()[start..start + dqkv]);
            }
            let v_slice = Tensor::new(v_data, &vec![total_seq_len, dqkv], device);

            let mut output_data = Tensor::<T>::default(&vec![seq_len, dqkv], device);
            OP::matmul(&mut output_data, 0.0, &score_slice, &v_slice, 1.0, device, stream);
            output_data.sync_data(device.get(), CPU, device, stream);
            for s in 0..seq_len {
                let mut output_slice = att_output.slice(
                    s * n_kv_h * n_groups * dqkv + (h * n_groups + g) * dqkv,
                    &vec![dqkv],
                    device
                );
                unsafe {
                    output_slice
                        .data_mut()
                        .copy_from_slice(output_data.slice(s * dqkv, &vec![dqkv], device).data());
                }
            }
        }
    }
    let reshaped = att_output.reshape(&vec![seq_len, n_kv_h * n_groups * dqkv]);
    unsafe {
        hidden_states.data_mut().copy_from_slice(reshaped.data());
    }
}

fn mlp<T: Copy + Clone + Default + Into<f32> + std::fmt::Debug>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: f32,
    device: &infinicore::Device,
    stream: &infinicore::Stream,
) {
    OP::rms_norm(hidden_states, residual, rms_w, eps, device, stream);

    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0, device, stream);
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0, device, stream);

    OP::swiglu(up, gate, device, stream);
    // device.synchronize();

    OP::matmul_transb(residual, 1.0, up, w_down, 1.0, device, stream);
}

// #[test]
// pub fn test_mlp() {
//     let seq_len = 4;
//     let d = 2;
//     let di = 3;
//     let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
//     let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
//     let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
//     let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
//     let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
//     let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
//     let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
//     let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
//     let eps = 1e-6;
//     mlp(
//         &mut residual,
//         &mut hidden_states,
//         &mut gate_buf,
//         &mut up_buf,
//         &w_up,
//         &w_down,
//         &w_gate,
//         &rms_w,
//         eps,
//     );

//     assert!(residual.close_to(
//         &Tensor::<f32>::new(
//             vec![
//                 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
//                 1.7290739
//             ],
//             &vec![seq_len, d]
//         ),
//         1e-3
//     ))
// }

// #[test]
// pub fn test_load_safetensors() {
//     use crate::tensor::float_eq;
//     use std::path::PathBuf;
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let model = Llama::from_safetensors(model_dir);
//     assert_eq!(model.vocab, 2048);
//     assert_eq!(model.n_layers, 2);
//     assert_eq!(model.n_q_h, 8);
//     assert_eq!(model.n_kv_h, 4);
//     assert_eq!(model.d, 128);
//     assert_eq!(model.dqkv, 16);
//     assert_eq!(model.di, 384);

//     assert!(float_eq(
//         &model.params.embedding_table.data()[50],
//         &0.14453125,
//         1e-6
//     ));
//     assert_eq!(
//         model.params.lm_head.data()[10],
//         model.params.embedding_table.data()[10]
//     );
//     assert!(float_eq(
//         &model.params.rms_att_w[0].data()[10],
//         &0.18652344,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_ffn_w[1].data()[10],
//         &0.32421875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_out_w.data()[100],
//         &0.73046875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.w_down[0].data()[100],
//         &-0.0625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
//     assert!(float_eq(
//         &model.params.w_gate[1].data()[100],
//         &0.296875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wq[1].data()[100],
//         &0.032226563,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wk[1].data()[100],
//         &-0.21386719,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wv[0].data()[100],
//         &0.041015625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
// }

// #[test]
// pub fn test_self_attention() {
//     let seq_len = 4;
//     let n_kv_h = 2;
//     let n_groups = 2;
//     let dqkv = 1;
//     let total_seq_len = 8;
//     let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);
//     let mut att_scores = Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
//     let q = Tensor::<f32>::new(
//         vec![
//             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
//         ],
//         &vec![seq_len, n_kv_h * n_groups * dqkv],
//     );
//     let k = Tensor::<f32>::new(
//         vec![
//             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
//         ],
//         &vec![total_seq_len, n_kv_h * dqkv],
//     );
//     let v = Tensor::<f32>::new(
//         vec![
//             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
//         ],
//         &vec![total_seq_len, n_kv_h * dqkv],
//     );

//     self_attention(
//         &mut hidden_states,
//         &mut att_scores,
//         &q,
//         &k,
//         &v,
//         n_kv_h,
//         n_groups,
//         seq_len,
//         total_seq_len,
//         dqkv,
//     );

//     println!("hidden_states: {:?}", hidden_states.data());

//     assert!(hidden_states.close_to(
//         &Tensor::<f32>::new(
//             vec![
//                 0.6800, 1.3600, 2.4480, 3.2640, 3.4000, 4.0800, 5.7120, 6.5280, 6.1200, 6.8000,
//                 8.9760, 9.7920, 8.8400, 9.5200, 12.2400, 13.0560
//             ],
//             &vec![n_kv_h, n_groups, seq_len, seq_len]
//         ),
//         1e-6
//     ));
// }
