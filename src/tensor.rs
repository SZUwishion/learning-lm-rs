use std::{alloc::Layout, slice, sync::Arc, vec};

use infinicore::{AsRaw, DeviceType};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
    dev_blob: Option<infinicore::DevBlob>,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>, device: &infinicore::Device) -> Self {
        let length = data.len();
        let dev_blob = match device.get() {
            infinicore::DeviceType::CPU => None,
            infinicore::DeviceType::CUDA => Some(device.from_host(&data)),
        };
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length,
            dev_blob,
        }
    }

    pub fn default(shape: &Vec<usize>, device: &infinicore::Device) -> Self {
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

    pub fn dev_blob(&self) -> &infinicore::DevBlob {
        self.dev_blob.as_ref().unwrap()
    }

    pub fn dev_blob_ptr(&self) -> *const T {
        unsafe {self.dev_blob.as_ref().unwrap().as_raw() as *const T}
    }

    pub fn dev_blob_ptr_mut(&mut self) -> *mut T {
        unsafe {self.dev_blob.as_mut().unwrap().as_raw() as *mut T}
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    pub fn strides(&self) -> Vec<isize> {
        let mut strides = vec![1isize; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1] as isize;
        }
        strides
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>, device: &infinicore::Device) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
            dev_blob: match device.get() {
                infinicore::DeviceType::CPU => None,
                infinicore::DeviceType::CUDA => {
                    let layout = Layout::array::<T>(new_length).unwrap();
                    let len = layout.size();
                    Some(device.from_host(&self.data()[start..][..new_length]))
                }
            },
        }
    }

    #[inline]
    pub fn sync_data(&mut self, src: DeviceType, dst: DeviceType, device: &infinicore::Device) {
        match (src, dst) {
            (DeviceType::CPU, DeviceType::CPU) => {}
            (DeviceType::CPU, DeviceType::CUDA) => {
                let data_ptr = unsafe { self.data_ptr() };
                let data_len = self.length;
                let slice = unsafe { slice::from_raw_parts(data_ptr, data_len) };
                device.memcpy_h2d(self.dev_blob.as_mut().unwrap(), slice);
            }
            (DeviceType::CUDA, DeviceType::CPU) => {
                let blob = self.dev_blob.as_ref().unwrap();
                let data_ptr = unsafe { self.data_ptr() };
                let data_len = self.length;
                unsafe {
                    let mut_slice = slice::from_raw_parts_mut(data_ptr as *mut T, data_len);
                    device.memcpy_d2h(mut_slice, blob);
                }
            }
            (DeviceType::CUDA, DeviceType::CUDA) => {}
        }
    }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
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
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
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
