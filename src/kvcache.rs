use std::{usize, vec};

use crate::tensor::Tensor;
pub struct KVCache<'a, T> {
    k_cache: Vec<Tensor<'a, T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<'a, T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<'a, T: Default + Copy> KVCache<'a, T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize, device: &'a infinicore::Device, stream: &'a infinicore::Stream) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim], device, stream))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim], device, stream))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<'a, T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<'a, T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize){
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}
