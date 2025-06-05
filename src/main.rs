mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

use clap::{Parser, command, arg};
use infinicore::{Device, DeviceType};
use half::f16;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    // 定义参数
    #[arg(short, long, default_value = "generate")]
    mode: String,

    #[arg(short, long, default_value = "cpu")]
    device: String,

    #[arg(short, long, default_value = "f32")]
    dtype: String,
}

fn main() {
    let args = Args::parse();
    let mut device = infinicore::Device::default();
    match args.device.as_str() {
        "cpu" => device.set(DeviceType::CPU, 0),
        "cuda" => device.set(DeviceType::CUDA, 0),
        _ => {
            println!("Invalid device");
            return;
        }
    };
    device.set_device();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = match args.dtype.as_str() {
        // "f16" => model::Llama::<f16>::from_safetensors(&model_dir, true, &device),
        "f32" => model::Llama::<f32>::from_safetensors(&model_dir, false, &device),
        _ => {
            println!("Invalid dtype");
            return;
        }
    };
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    match args.mode.as_str() {
        "chat" => {
            llama.chat(&tokenizer, 50, 0.8, 30, 1., &device);
        }
        "generate" => {
            let input = "Once upon a time";
            let binding = tokenizer.encode(input, true).unwrap();
            let input_ids = binding.get_ids();
            let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1., &device);
            println!("{}", tokenizer.decode(&output_ids, true).unwrap());
        }
        _ => {
            println!("Invalid mode");
        }
    }
}
