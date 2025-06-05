use std::any::type_name;
use std::ptr::null;

use crate::{tensor::Tensor, DeviceType};
use digit_layout::types;
use infinicore::{infini, AsRaw};

pub fn convert_type<T: Copy + Clone + Default>() -> digit_layout::DigitLayout {
    match type_name::<T>() {
        "f32" => types::F32,
        "f16" => types::F16,
        "u32" => types::U32,
        "i32" => types::I32,
        "u8" => types::U8,
        "i8" => types::I8,
        _ => panic!("Unsupported type: {}", type_name::<T>()),
    }
}

pub fn add<T: Copy + Clone + Default>(y: &mut Tensor<T>, x: &Tensor<T>, device: &infinicore::Device) {
    let y_shape = y.shape();
    let x_shape = x.shape();
    assert!(y_shape.len() == x_shape.len());
    let n_bytes = convert_type::<T>().nbytes() as isize;
    let y_strides = y.strides().iter().map(|&s| s * n_bytes).collect::<Vec<_>>();
    let x_strides = x.strides().iter().map(|&s| s * n_bytes).collect::<Vec<_>>();
    let y_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        y_shape.iter().copied(),
        y_strides.iter().copied(),
    );
    let x_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        x_shape.iter().copied(),
        x_strides.iter().copied(),
    );

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();
    infini!(infiniopCreateAddDescriptor(
        handle.as_raw(),
        &mut desc,
        y_desc.as_raw(),
        x_desc.as_raw(),
        y_desc.as_raw()
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetAddWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopAdd(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                y.data_mut().as_mut_ptr() as *mut _,
                x.data().as_ptr() as *const _,
                y.data().as_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopAdd(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                y.dev_blob_ptr_mut() as *mut _,
                x.dev_blob_ptr() as *const _,
                y.dev_blob_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
    }
    infini!(infiniopDestroyAddDescriptor(desc));
}
// get (row) vectors from a 2D table given a list of indices
pub fn gather<T: Copy + Clone + Default>(
    y: &mut Tensor<T>,
    indices: &Tensor<u32>,
    table: &Tensor<T>,
    device: &infinicore::Device,
) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope<
    T: Copy
        + Clone
        + Default
        + Into<f32>
        + From<f32>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::fmt::Debug,
>(
    y: &mut Tensor<T>,
    start_pos: usize,
    theta: f32,
) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * T::from(cos) - b * T::from(sin);
                data[tok * n_heads * d + head * d + i + d / 2] =
                    b * T::from(cos) + a * T::from(sin);
            }
        }
    }
}

pub fn new_rope<T: Copy + Clone + Default + Into<f32> + From<f32> + std::fmt::Debug>(
    y: &mut Tensor<T>,
    start_pos: usize,
    theta: f32,
) {
    let nbytes = convert_type::<T>().nbytes() as isize;
    let shape = y.shape().to_vec();
    let strides = y.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];

    println!("y: {:?}", y.data());
    y.reshape(&vec![seq_len, n_heads, 2, d / 2]).permute(&vec![0, 1, 3, 2]).reshape(&shape);
    println!("y: {:?}", y.data());
    
    assert!(shape.len() == 3);

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    // 创建tensor描述符
    let y_desc =
        infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), strides.iter().copied());

    
    // 创建位置ID张量
    let pos_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
    let pos_desc = infinicore::Tensor::new(
        types::U32,
        std::iter::once(shape[0]),
        std::iter::once(convert_type::<u32>().nbytes() as isize),
    );

    // 创建sin和cos表
    let mut sin_table: Vec<T> = vec![T::from(0.0); seq_len * d / 2];
    let mut cos_table: Vec<T> = vec![T::from(0.0); seq_len * d / 2];

    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for i in 0..d / 2 {
            let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
            let (sin, cos) = freq.sin_cos();
            sin_table[tok * (d / 2) + i] = T::from(sin);
            cos_table[tok * (d / 2) + i] = T::from(cos);
        }
    }

    let sin_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        [seq_len, d / 2].iter().copied(),
        [(d / 2) as isize * nbytes, nbytes].iter().copied(),
    );
    let cos_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        [seq_len, d / 2].iter().copied(),
        [(d / 2) as isize * nbytes, nbytes].iter().copied(),
    );

    // 创建RoPE描述符
    infini!(infiniopCreateRoPEDescriptor(
        handle.as_raw(),
        &mut desc,
        y_desc.as_raw(),
        y_desc.as_raw(),
        pos_desc.as_raw(),
        sin_desc.as_raw(),
        cos_desc.as_raw()
    ));

    // 获取工作空间大小
    let mut workspace_size: usize = 0;
    infini!(infiniopGetRoPEWorkspaceSize(desc, &mut workspace_size));
    let mut workspace = vec![0u8; workspace_size];

    // 执行RoPE操作
    infini!(infiniopRoPE(
        desc,
        workspace.as_mut_ptr() as *mut _,
        workspace_size,
        y.data_mut().as_mut_ptr() as *mut _,
        y.data().as_ptr() as *const _,
        pos_ids.as_ptr() as *const _,
        sin_table.as_ptr() as *const _,
        cos_table.as_ptr() as *const _,
        std::ptr::null_mut()
    ));

    y.reshape(&vec![seq_len, n_heads, d / 2, 2]).permute(&vec![0, 1, 3, 2]).reshape(&shape);

    // 清理描述符
    infini!(infiniopDestroyRoPEDescriptor(desc));
}

// // softmax(x) = exp(x - max) / sum(exp(x - max))
// // y = softmax(mask(x))
// pub fn masked_softmax<T: Copy + Clone + Default + Into<f32> + From<f32> + std::cmp::PartialOrd>(
//     y: &mut Tensor<T>,
// ) {
//     let ndim = y.shape().len();
//     assert!(ndim >= 2);
//     let seq_len = y.shape()[ndim - 2];
//     let total_seq_len = y.shape()[ndim - 1];
//     let batch = y.size() / (seq_len * total_seq_len);
//     let data = unsafe { y.data_mut() };
//     for b in 0..batch {
//         let base = b * seq_len * total_seq_len;
//         for i in 0..seq_len {
//             let offset = base + i * total_seq_len;
//             let boundary = total_seq_len - seq_len + i + 1;

//             let max: T = data[offset..offset + boundary]
//                 .iter()
//                 .fold(data[offset], |a: T, b: &T| if a > *b { a } else { *b });

//             let sum: f32 = (0..boundary)
//                 .map(|j| {
//                     let e: f32 = ((data[offset + j].into() - max.into()).exp());
//                     data[offset + j] = e.into();
//                     e
//                 })
//                 .sum();

//             (0..boundary).for_each(|j| data[offset + j] = (data[offset + j].into() / sum).into());
//             (boundary..total_seq_len).for_each(|j| data[offset + j] = T::default());
//         }
//     }
// }

pub fn causal_softmax<T: Copy + Clone + Default>(y: &mut Tensor<T>, device: &infinicore::Device) {
    let shape = y.shape();
    let nbytes = convert_type::<T>().nbytes() as isize;
    let strides = y.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    let x_desc =
        infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), strides.iter().copied());
    let y_desc =
        infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), strides.iter().copied());

    infini!(infiniopCreateCausalSoftmaxDescriptor(
        handle.as_raw(),
        &mut desc,
        y_desc.as_raw(),
        x_desc.as_raw()
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetCausalSoftmaxWorkspaceSize(
        desc,
        &mut workspace_size
    ));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopCausalSoftmax(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                y.data_mut().as_mut_ptr() as *mut _,
                y.data().as_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopCausalSoftmax(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                y.dev_blob_ptr_mut() as *mut _,
                y.dev_blob_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
    }

    infini!(infiniopDestroyCausalSoftmaxDescriptor(desc));
}

pub fn rms_norm<T: Copy + Clone + Default>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
    w: &Tensor<T>,
    epsilon: f32,
    device: &infinicore::Device,
) {
    let x_shape = x.shape();
    let y_shape = y.shape();
    let w_shape = w.shape();
    assert!(x_shape.len() == 2);
    assert!(y_shape.len() == 2);
    assert!(w_shape.len() == 1);
    assert!(x_shape[1] == y_shape[1]);
    assert!(x_shape[1] == w_shape[0]);

    let nbytes = convert_type::<T>().nbytes() as isize;
    let x_strides = x.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let y_strides = y.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let w_strides = w.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    let x_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        x_shape.iter().copied(),
        x_strides.iter().copied(),
    );
    let y_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        y_shape.iter().copied(),
        y_strides.iter().copied(),
    );
    let w_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        w_shape.iter().copied(),
        w_strides.iter().copied(),
    );

    infini!(infiniopCreateRMSNormDescriptor(
        handle.as_raw(),
        &mut desc,
        x_desc.as_raw(),
        y_desc.as_raw(),
        w_desc.as_raw(),
        epsilon
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetRMSNormWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopRMSNorm(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                y.data_mut().as_mut_ptr() as *mut _,
                x.data().as_ptr() as *const _,
                w.data().as_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopRMSNorm(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                y.dev_blob_ptr_mut() as *mut _,
                x.dev_blob_ptr() as *const _,
                w.dev_blob_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
    }

    infini!(infiniopDestroyRMSNormDescriptor(desc));
}

pub fn swiglu<T: Copy + Clone + Default>(y: &mut Tensor<T>, x: &Tensor<T>, device: &infinicore::Device) {
    let y_shape = y.shape();
    let x_shape = x.shape();

    let nbytes = convert_type::<T>().nbytes() as isize;
    let y_strides = y.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let x_strides = x.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let y_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        y_shape.iter().copied(),
        y_strides.iter().copied(),
    );
    let x_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        x_shape.iter().copied(),
        x_strides.iter().copied(),
    );

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    infini!(infiniopCreateSwiGLUDescriptor(
        handle.as_raw(),
        &mut desc,
        y_desc.as_raw(),
        y_desc.as_raw(),
        x_desc.as_raw(),
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetSwiGLUWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopSwiGLU(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                y.data_mut().as_mut_ptr() as *mut _,
                y.data().as_ptr() as *const _,
                x.data().as_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopSwiGLU(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                y.dev_blob_ptr_mut() as *mut _,
                y.dev_blob_ptr() as *const _,
                x.dev_blob_ptr() as *const _,
                std::ptr::null_mut()
            ));
        }
    }

    infini!(infiniopDestroySwiGLUDescriptor(desc));
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T: Copy + Clone + Default + Into<f32>>(
    c: &mut Tensor<T>,
    beta: f32,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f32,
    device: &infinicore::Device,
) {
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(c_shape.len(), 2);
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    assert_eq!(a_shape[1], b_shape[1]);
    assert_eq!(c_shape[0], a_shape[0]);
    assert_eq!(c_shape[1], b_shape[0]);

    let nbytes = convert_type::<T>().nbytes() as isize;
    let c_strides = c.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let a_strides = a.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let b_transposed_shape = [b_shape[1], b_shape[0]];
    let b_original_strides = b.strides();
    let b_transposed_strides = [
        b_original_strides[1] * nbytes,
        b_original_strides[0] * nbytes,
    ];

    let c_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        c_shape.iter().copied(),
        c_strides.iter().copied(),
    );
    let a_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        a_shape.iter().copied(),
        a_strides.iter().copied(),
    );
    let b_desc = infinicore::Tensor::new(
        convert_type::<T>(),
        b_transposed_shape.iter().copied(),
        b_transposed_strides.iter().copied(),
    );

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    infini!(infiniopCreateGemmDescriptor(
        handle.as_raw(),
        &mut desc,
        c_desc.as_raw(),
        a_desc.as_raw(),
        b_desc.as_raw(),
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetGemmWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopGemm(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                c.data_mut().as_mut_ptr() as *mut _,
                a.data().as_ptr() as *const _,
                b.data().as_ptr() as *const _,
                alpha,
                beta,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopGemm(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                c.dev_blob_ptr_mut() as *mut _,
                a.dev_blob_ptr() as *const _,
                b.dev_blob_ptr() as *const _,
                alpha,
                beta,
                std::ptr::null_mut()
            ));
            device.synchronize();
        }
    }

    infini!(infiniopDestroyGemmDescriptor(desc));
}

pub fn matmul<T: Copy + Clone + Default + Into<f32>>(
    c: &mut Tensor<T>,
    beta: f32,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: f32,
    device: &infinicore::Device,
) {
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(c_shape.len(), 2);
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    assert_eq!(a_shape[1], b_shape[0]);
    assert_eq!(c_shape[0], a_shape[0]);
    assert_eq!(c_shape[1], b_shape[1]);

    let nbytes = types::F32.nbytes() as isize;
    let c_strides = c.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let a_strides = a.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();
    let b_strides = b.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let c_desc = infinicore::Tensor::new(
        types::F32,
        c_shape.iter().copied(),
        c_strides.iter().copied(),
    );
    let a_desc = infinicore::Tensor::new(
        types::F32,
        a_shape.iter().copied(),
        a_strides.iter().copied(),
    );
    let b_desc = infinicore::Tensor::new(
        types::F32,
        b_shape.iter().copied(),
        b_strides.iter().copied(),
    );

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    infini!(infiniopCreateGemmDescriptor(
        handle.as_raw(),
        &mut desc,
        c_desc.as_raw(),
        a_desc.as_raw(),
        b_desc.as_raw(),
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetGemmWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopGemm(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                c.data_mut().as_mut_ptr() as *mut _,
                a.data().as_ptr() as *const _,
                b.data().as_ptr() as *const _,
                alpha.into(),
                beta.into(),
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopGemm(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                c.dev_blob_ptr_mut() as *mut _,
                a.dev_blob_ptr() as *const _,
                b.dev_blob_ptr() as *const _,
                alpha.into(),
                beta.into(),
                std::ptr::null_mut()
            ));
        }
    }

    infini!(infiniopDestroyGemmDescriptor(desc));
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T: Copy + Clone + Default + std::ops::Mul<Output = T> + std::ops::AddAssign>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    device: &infinicore::Device,
) -> T {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = T::default();
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// // Sample a index from a tensor (treated as a probability vector)
// pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
//     assert!(x.shape()[x.shape().len() - 1] == x.size());
//     if temperature <= 0. || top_k < 2 || top_p <= 0. {
//         return x
//             .data()
//             .iter()
//             .enumerate()
//             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//             .unwrap()
//             .0 as _;
//     }

//     #[derive(Clone, Copy, PartialEq, Debug)]
//     struct Probability {
//         val: f32,
//         tok: u32,
//     }
//     impl Eq for Probability {}
//     impl PartialOrd for Probability {
//         #[inline]
//         fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//             Some(self.cmp(other))
//         }
//     }
//     impl Ord for Probability {
//         #[inline]
//         fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//             match self.val.total_cmp(&other.val) {
//                 std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
//                 ord => ord.reverse(),
//             }
//         }
//     }
//     impl From<(usize, &f32)> for Probability {
//         #[inline]
//         fn from((i, p): (usize, &f32)) -> Self {
//             Self {
//                 val: p.clone(),
//                 tok: i as _,
//             }
//         }
//     }

//     // sort
//     let mut logits = x
//         .data()
//         .iter()
//         .enumerate()
//         .map(Probability::from)
//         .collect::<Vec<_>>();
//     logits.sort_unstable();
//     let max = core::mem::replace(&mut logits[0].val, 1.);
//     // softmax & sum
//     for i in 1..logits.len() {
//         logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
//     }
//     // topk & topp & random
//     let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
//     let pp = logits[logits.len() - 1].val * top_p;
//     let plimit = rand::random::<f32>() * f32::min(pk, pp);
//     // sample
//     logits.iter().find(|p| p.val >= plimit).unwrap().tok
// }

pub fn random_sample<T: Copy + Clone + Default>(
    x: &Tensor<T>,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    device: &infinicore::Device,
) -> u32 {
    let shape = x.shape();
    let nbytes = convert_type::<T>().nbytes() as isize;
    let strides = x.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>();

    let mut handle = infinicore::Handle::new();
    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    let x_desc =
        infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), strides.iter().copied());

    let mut indices = Tensor::<u32>::new(vec![0], &vec![], x.device());
    let indices_desc = infinicore::Tensor::new(types::U32, std::iter::empty(), std::iter::empty());

    infini!(infiniopCreateRandomSampleDescriptor(
        handle.as_raw(),
        &mut desc,
        indices_desc.as_raw(),
        x_desc.as_raw()
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetRandomSampleWorkspaceSize(
        desc,
        &mut workspace_size
    ));

    let random_val = rand::random::<f32>();

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopRandomSample(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                indices.data_mut().as_mut_ptr() as *mut _,
                x.data().as_ptr() as *const _,
                random_val,
                top_p,
                top_k as i32,
                temperature,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopRandomSample(
                desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                indices.dev_blob_ptr_mut() as *mut _,
                x.dev_blob_ptr() as *const _,
                random_val,
                top_p,
                top_k as i32,
                temperature,
                std::ptr::null_mut()
            ));
        }
    }
    indices.sync_data(device.get(), DeviceType::CPU);

    infini!(infiniopDestroyRandomSampleDescriptor(desc));

    indices.data()[0]
}

pub fn self_attention<T: Copy + Clone + Default + Into<f32> + std::fmt::Debug>(
    hidden_states: &mut Tensor<T>,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    k_cache: &mut Tensor<T>,
    v_cache: &mut Tensor<T>,
    pos: usize,
    device: &infinicore::Device,
) {
    let handle = infinicore::Handle::new();
    let nbytes = convert_type::<T>().nbytes() as isize;

    let q_desc = infinicore::Tensor::new(convert_type::<T>(), q.shape().iter().copied(), q.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());
    let k_desc = infinicore::Tensor::new(convert_type::<T>(), k.shape().iter().copied(), k.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());
    let v_desc = infinicore::Tensor::new(convert_type::<T>(), v.shape().iter().copied(), v.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());

    let k_cache_desc = infinicore::Tensor::new(convert_type::<T>(), k_cache.shape().iter().copied(), k_cache.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());
    let v_cache_desc = infinicore::Tensor::new(convert_type::<T>(), v_cache.shape().iter().copied(), v_cache.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());

    let hidden_desc = infinicore::Tensor::new(convert_type::<T>(), hidden_states.shape().iter().copied(), hidden_states.strides().iter().map(|&s| s * nbytes).collect::<Vec<_>>());

    let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

    infini!(infiniopCreateAttentionDescriptor(
        handle.as_raw(),
        &mut desc,
        hidden_desc.as_raw(),
        q_desc.as_raw(),
        k_desc.as_raw(),
        v_desc.as_raw(),
        k_cache_desc.as_raw(),
        v_cache_desc.as_raw(),
        pos,
    ));

    let mut workspace_size: usize = 0;
    infini!(infiniopGetAttentionWorkspaceSize(desc, &mut workspace_size));

    match device.get() {
        DeviceType::CPU => {
            let mut workspace = vec![0u8; workspace_size];
            infini!(infiniopAttention(
                desc,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                hidden_states.data_mut().as_mut_ptr() as *mut _,
                q.data().as_ptr() as *const _,
                k.data().as_ptr() as *const _,
                v.data().as_ptr() as *const _,
                k_cache.data_mut().as_mut_ptr() as *mut _,
                v_cache.data_mut().as_mut_ptr() as *mut _,
                std::ptr::null_mut()
            ));
        }
        DeviceType::CUDA => {
            let mut workspace = device.malloc::<u8>(workspace_size);
            infini!(infiniopAttention(desc,
                workspace.as_raw() as *mut _,
                workspace_size,
                hidden_states.dev_blob_ptr_mut() as *mut _,
                q.dev_blob_ptr() as *const _,
                k.dev_blob_ptr() as *const _,
                v.dev_blob_ptr() as *const _,
                k_cache.dev_blob_ptr_mut() as *mut _,
                v_cache.dev_blob_ptr_mut() as *mut _,
                std::ptr::null_mut()
            ));
        }
    }

    infini!(infiniopDestroyAttentionDescriptor(desc));

}

// #[test]
// fn test_silu() {
//     let device = infinicore::Device::default();
//     let mut y = Tensor::<f32>::new(vec![2., 3., 4.], vec![1, 3], &device);
//     let x = Tensor::<f32>::new(vec![1., 2., 3.], vec![1, 3], &device);
//     swiglu(&mut y, &x, &device);
//     println!("y: {:?}", y.data());
//     assert!(y.close_to(
//         &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], vec![1, 3], &device),
//         1e-3
//     ));
// }

// #[test]
// fn test_rms_norm() {
//     let device = infinicore::Device::default();
//     let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], vec![2, 2], &device);
//     let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], vec![2, 2], &device);
//     let w = Tensor::<f32>::new(vec![1., 2.], vec![2], &device);
//     rms_norm(&mut y, &x, &w, 1e-6, &device);
//     assert!(y.close_to(
//         &Tensor::<f32>::new(
//             vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
//             vec![2, 2],
//             &device
//         ),
//         1e-3
//     ));
// }

// #[test]
// fn test_matmul_transb() {
//     let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
//     let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
//     let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
//     matmul_transb(&mut c, 1., &a, &b, 1.);
//     assert!(c.close_to(
//         &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
//         1e-3
//     ));
// }

// #[test]
// fn test_causal_softmax() {
//     let device = infinicore::Device::default();
//     let mut ans = Tensor::<f32>::new(
//         vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
//         vec![3, 3],
//         &device
//     );
//     let mut y = Tensor::<f32>::new(
//         vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
//         vec![3, 3],
//         &device
//     );
//     causal_softmax(&mut y, &device);;
//     println!("y: {:?}", y.data());
//     println!("ans: {:?}", ans.data());
// }

#[test]
fn test_rope() {
    use rand::prelude::*;

    // 测试参数
    let seq_len = 4;
    let n_heads = 2;
    let d = 8; // 必须是偶数，因为rope算法将其分为两半
    let start_pos = 0;
    let theta = 10000.0;

    // 生成随机数据
    // let mut rng = rand::thread_rng();
    // let mut random_data = Vec::with_capacity(seq_len * n_heads * d);
    // for _ in 0..(seq_len * n_heads * d) {
    //     random_data.push(rng.gen_range(-1.0..1.0));
    // }

    // println!("random_data: {:?}", random_data);

    let random_data = vec![
        0.6274, 0.0861, 0.2783, 0.2329, 0.0113, 0.7522, 0.4644, 0.4073, 0.2808,
        0.3611, 0.9342, 0.8281, 0.6032, 0.3108, 0.4151, 0.3991, 0.7340, 0.9061,
        0.0389, 0.5242, 0.3819, 0.7386, 0.3717, 0.2770, 0.3393, 0.1041, 0.6288,
        0.9183, 0.8932, 0.3034, 0.3706, 0.0547, 0.4740, 0.2088, 0.8398, 0.0716,
        0.9928, 0.7421, 0.2815, 0.4603, 0.5274, 0.7123, 0.7616, 0.0950, 0.0275,
        0.9204, 0.9939, 0.5520, 0.1029, 0.6280, 0.1588, 0.4183, 0.5541, 0.7679,
        0.4762, 0.5599, 0.6501, 0.6056, 0.8263, 0.5505, 0.1370, 0.8323, 0.0546,
        0.6466
    ];
    
    let mut device = infinicore::Device::default();

    // 创建相同的输入tensor，分别用于原始rope和新rope
    let mut original_tensor = Tensor::<f32>::new(random_data.clone(), &vec![seq_len, n_heads, d], &device);
    let mut new_tensor = Tensor::<f32>::new(random_data.clone(), &vec![seq_len, n_heads, d], &device);

    // 执行原始rope操作
    rope(&mut original_tensor, start_pos, theta);

    // println!("original_tensor: {:?}", original_tensor.data());

    // 执行新rope操作（假设已实现）
    new_rope(&mut new_tensor, start_pos, theta);

    // println!("new_tensor: {:?}", new_tensor.data());

    assert!(new_tensor.close_to(&original_tensor, 1e-6));
}

#[test]
pub fn test_self_attention() {
    let mut device = infinicore::Device::default();
    let seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 2;
    let dqkv = 1;
    let total_seq_len = 8;
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups, dqkv], &device);
    let q = Tensor::<f32>::new(
        vec![
            0.07501934, 0.03810351, 0.07624029, 0.02104790, 0.02756642, 0.06471346,
        0.01695717, 0.06440213, 0.04326092, 0.09738450, 0.05576990, 0.07929533,
        0.06970619, 0.00332901, 0.08680507, 0.09460428
        ],
        &vec![n_kv_h * n_groups, seq_len, dqkv],
        &device
    );
    let mut k = Tensor::<f32>::new(
        vec![
            0.03173396, 0.07719559, 0.09175372, 0.04245919, 0.04219837, 0.01831813,
        0.05825133, 0.04863298
        ],
        &vec![n_kv_h, seq_len, dqkv],
        &device
    );
    let mut v = Tensor::<f32>::new(
        vec![
            0.02179549, 0.01165751, 0.03982254, 0.08162189, 0.04339119, 0.00063249,
        0.05911496, 0.05535178
        ],
        &vec![n_kv_h, seq_len, dqkv],
        &device
    );
    let mut k_cache = Tensor::<f32>::new(
        vec![
            0.09918365, 0.01031387, 0.07083724, 0.01287274, 0.09505566, 0.03446914,
        0.03289497, 0.08557402, 0.02675064, 0.04614840, 0.07013312, 0.08869162,
        0.07064439, 0.00436187, 0.02215201, 0.05467457
            ],
            &vec![n_kv_h, total_seq_len, dqkv],
            &device
    );
    let mut v_cache = Tensor::<f32>::new(
        vec![
            0.00567356, 0.09062338, 0.05695992, 0.06159676, 0.01643596, 0.02609916,
        0.06027269, 0.03515903, 0.05011621, 0.04568907, 0.02459981, 0.00277210,
        0.08462793, 0.02514033, 0.09587718, 0.05959567
        ],
        &vec![n_kv_h, total_seq_len, dqkv], &device);
    let pos = 0;
    self_attention(
        &mut hidden_states,
        &q,
        &k,
        &v,
        &mut k_cache,
        &mut v_cache,
        pos,
        &device
    );

    // println!("hidden_states: {:?}", hidden_states.data());

    assert!(hidden_states.close_to(
        &Tensor::<f32>::new(
            vec![
                0.02179549, 0.02179549, 0.04339119, 0.04339119, 0.01672211, 0.01671904,
        0.02203670, 0.02201269, 0.02443392, 0.02442713, 0.03440190, 0.03441434,
        0.03872065, 0.03871301, 0.03964926, 0.03965440
            ],
            &vec![seq_len, n_kv_h * n_groups, dqkv],
            &device
        ),
        1e-6
    ));
}
