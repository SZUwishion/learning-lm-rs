use std::{alloc::Layout, ptr, slice, sync::Arc, vec};

use half::f16;
use infinicore::{infini, AsRaw, DeviceType};

use crate::operators::convert_type;
pub struct Tensor<'a, T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    length: usize,
    device: &'a infinicore::Device,
    dev_blob: Option<infinicore::DevBlob>,
}

impl<'a, T: Copy + Clone + Default> Tensor<'a, T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>, device: &'a infinicore::Device) -> Self {
        let length = data.len();
        let dev_blob = match device.get() {
            infinicore::DeviceType::CPU => None,
            infinicore::DeviceType::CUDA => Some(device.from_host(&data)),
        };
        let mut strides = vec![1isize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            strides: strides.clone(),
            offset: 0,
            length,
            device,
            dev_blob,
        }
    }

    pub fn default(shape: &Vec<usize>, device: &'a infinicore::Device) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape, device)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub unsafe fn data_ptr(&self) -> *const T {
        self.data.as_ptr().add(self.offset)
    }

    pub fn device(&self) -> &'a infinicore::Device {
        self.device
    }

    // pub fn dev_blob(&self) -> &infinicore::DevBlob {
    //     self.dev_blob.as_ref().unwrap()
    // }

    pub fn dev_blob_ptr(&self) -> *const T {
        unsafe { self.dev_blob.as_ref().unwrap().as_raw().add(self.offset) as *const T }
    }

    pub fn dev_blob_ptr_mut(&mut self) -> *mut T {
        unsafe { self.dev_blob.as_mut().unwrap().as_raw().add(self.offset) as *mut T }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    pub fn strides(&self) -> &Vec<isize> {
        &self.strides
    }

    pub fn compact_strides(&self) -> Vec<isize> {
        let mut compact_strides = vec![1isize; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            compact_strides[i] = compact_strides[i + 1] * self.shape[i + 1] as isize;
        }
        compact_strides
    }

    pub fn is_compact(&self) -> bool {
        self.strides == self.compact_strides()
    }

    pub fn compact(&mut self) {
        if self.is_compact() {
            return;
        }

        // let shape = self.shape.clone();
        // let src_strides = self.strides.clone();
        // let mut dst_strides = self.compact_strides();

        // let nbytes = convert_type::<T>().nbytes() as isize;

        // let src_desc = infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), src_strides.iter().map(|&s| s * nbytes).collect::<Vec<_>>());
        // let dst_desc = infinicore::Tensor::new(convert_type::<T>(), shape.iter().copied(), dst_strides.iter().map(|&s| s * nbytes).collect::<Vec<_>>());

        // let handle= infinicore::Handle::new();
        // let mut desc: *mut infinicore::bindings::InfiniopDescriptor = std::ptr::null_mut();

        // infini!(infiniopCreateRearrangeDescriptor(
        //     handle.as_raw(),
        //     &mut desc,
        //     dst_desc.as_raw(),
        //     src_desc.as_raw()
        // ));

        // match self.device.get() {
        //     infinicore::DeviceType::CPU => {
        //         infini!(infiniopRearrange(
        //             desc,
        //             self.data_mut().as_mut_ptr() as *mut _,
        //             self.data().as_ptr() as *const _,
        //             std::ptr::null_mut()
        //         ));
        //     }
        //     infinicore::DeviceType::CUDA => {
        //         infini!(infiniopRearrange(desc,
        //             self.dev_blob_ptr_mut() as *mut _,
        //             self.dev_blob_ptr() as *const _,
        //             std::ptr::null_mut()
        //         ));
        //     }
        // }

        let mut new_strides = vec![1isize; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * self.shape[i + 1] as isize;
        }
        let mut new_data = vec![T::default(); self.length];

        let total_elements = self.length;
        let mut indices = vec![0usize; self.shape.len()];
        for i in 0..total_elements {
            let mut src_offset = self.offset;
            let mut dst_offset = 0;
            for j in 0..self.shape.len() {
                src_offset += indices[j] * self.strides[j] as usize;
                dst_offset += indices[j] * new_strides[j] as usize;
            }
            new_data[dst_offset] = self.data[src_offset];
            for j in (0..self.shape.len()).rev() {
                indices[j] += 1;
                if indices[j] < self.shape[j] {
                    break;
                }
                indices[j] = 0;
            }
        }

        self.strides = new_strides.clone();
        unsafe {
            self.data_mut().copy_from_slice(&new_data);
        }

        if self.device.get() == DeviceType::CUDA {
            self.sync_data(DeviceType::CPU, DeviceType::CUDA);
        }
    }

    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }

        // self.compact(); // reshape强制使用新的连续的strides，所以需要先compact，防止变换失效

        self.shape = new_shape.clone();
        let mut new_strides = vec![1isize; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1] as isize;
        }
        self.strides = new_strides.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        let mut new_strides = vec![1isize; shape.len()];
        if shape.len() == self.shape.len() {
            new_strides = self.strides.clone();
        } else {
            for i in (0..shape.len() - 1).rev() {
                new_strides[i] = new_strides[i + 1] * shape[i + 1] as isize;
            }
        }
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            strides: new_strides,
            offset: self.offset + start,
            length: new_length,
            device: self.device,
            dev_blob: match self.device.get() {
                infinicore::DeviceType::CPU => None,
                infinicore::DeviceType::CUDA => self.dev_blob.clone(),
            },
        }
    }

    pub fn permute(&mut self, new_axes: &Vec<usize>) -> &mut Self {
        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.shape.len());

        for &idx in new_axes {
            new_shape.push(self.shape[idx]);
            new_strides.push(self.strides[idx]);
        }

        self.shape.copy_from_slice(&new_shape);
        self.strides.copy_from_slice(&new_strides);
        self.compact();
        self
    }

    #[inline]
    pub fn sync_data(&mut self, src: DeviceType, dst: DeviceType) {
        match (src, dst) {
            (DeviceType::CPU, DeviceType::CPU) => {}
            (DeviceType::CPU, DeviceType::CUDA) => {
                let data_ptr = unsafe { self.data_ptr() };
                let data_len = self.length;
                let slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
                self.device
                    .memcpy_h2d(self.dev_blob.as_mut().unwrap(), slice);
            }
            (DeviceType::CUDA, DeviceType::CPU) => {
                let blob = self.dev_blob.as_ref().unwrap();
                let data_ptr = unsafe { self.data_ptr() };
                let data_len = self.length;
                unsafe {
                    let mut_slice = slice::from_raw_parts_mut(data_ptr as *mut T, data_len);
                    self.device.memcpy_d2h(mut_slice, blob);
                }
            }
            (DeviceType::CUDA, DeviceType::CUDA) => {}
        }
    }
}

// Some helper functions for testing and debugging
impl<'a> Tensor<'a, f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}
#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
