//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, Special};

struct Args {
    /// The path to the model
    model: Model,
    /// The prompt
    prompt: String,
    /// Whether to normalise the produced embeddings
    normalise: bool,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
}

enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
        }
    }
}
fn main() -> Result<()> {
    let Args {
        model,
        prompt,
        normalise,
    } = Args {
        model: Model::Local {
            path: "/Users/bochao/Downloads/bge-m3-q8_0.gguf".into(),
            //path: "/Users/bochao/Downloads/Qwen3-Embedding-4B-Q4_K_M.gguf".into(),
        },
        prompt: "国际社会对此反应呈现两极分化。俄罗斯卫星通讯社援引克里姆林宫消息人士称".to_string(),
        normalise: true,
    };

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model_path = model.get_or_load()?;
    let model = LlamaModel::load_from_file(&backend, model_path.clone(), &model_params)?;

    // 检测模型类型
    let is_encoder_model = is_encoder_model(&model_path);
    eprintln!("模型类型: {}", if is_encoder_model { "Encoder (BGE/BERT)" } else { "Decoder (Qwen/LLaMA)" });

    // 根据模型类型设置不同的参数
    let ctx_params = if is_encoder_model {
        // BGE-M3等编码器模型的配置
        LlamaContextParams::default()
            .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
            .with_n_ctx(NonZeroU32::new(8192))
            .with_n_ubatch(8192)  // 编码器模型需要更大的ubatch
            .with_pooling_type(LlamaPoolingType::Mean)  // 编码器通常用Mean pooling
            .with_embeddings(true)
    } else {
        // Qwen等解码器模型的配置
        LlamaContextParams::default()
            .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
            .with_n_ctx(NonZeroU32::new(8192))
            .with_n_ubatch(1024)
            .with_pooling_type(LlamaPoolingType::Last)  // 解码器通常用Last pooling
            .with_embeddings(true)
    };

    for i in 0..5 {
        let mut ctx = model.new_context(&backend, ctx_params.clone())?;

        let prompt_lines = prompt.lines();
        let tokens_lines_list = prompt_lines
            .map(|line| model.str_to_token(line, AddBos::Always))
            .collect::<Result<Vec<_>, _>>()?;

        let n_ctx = ctx.n_ctx() as usize;
        let n_ctx_train = model.n_ctx_train();

        eprintln!("n_ctx = {n_ctx}, n_ctx_train = {n_ctx_train}");

        if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
            bail!("One of the provided prompts exceeds the size of the context window");
        }

        std::io::stderr().flush()?;

        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut max_seq_id_batch = 0;
        let mut output = Vec::with_capacity(tokens_lines_list.len());

        let t_main_start = ggml_time_us();

        for tokens in &tokens_lines_list {
            println!("batch.n_tokens(): {}", batch.n_tokens());
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
                // 根据模型类型选择不同的处理方法
                if is_encoder_model {
                    batch_encode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, normalise)?;
                } else {
                    batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, normalise)?;
                }
                max_seq_id_batch = 0;
            }

            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        
        // Handle final batch
        if is_encoder_model {
            batch_encode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, normalise)?;
        } else {
            batch_decode(&mut ctx, &mut batch, max_seq_id_batch, &mut output, normalise)?;
        }

        let t_main_end = ggml_time_us();

        for (i, embeddings) in output.iter().enumerate() {
            eprintln!("Embeddings {i}: 前5个值: {:?}", &embeddings[..5.min(embeddings.len())]);
            eprintln!("Embeddings len {i}: {}", embeddings.len());
            eprintln!();
        }

        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
        eprintln!(
            "Created embeddings for {} tokens in {:.2} s, speed {:.2} t/s\n",
            total_tokens,
            duration.as_secs_f32(),
            total_tokens as f32 / duration.as_secs_f32()
        );

        println!("{}", ctx.timings());
    }

    Ok(())
}

// 新增：检测是否为编码器模型
fn is_encoder_model(model_path: &std::path::Path) -> bool {
    let path_str = model_path.to_string_lossy().to_lowercase();
    
    // 编码器模型标识
    let encoder_indicators = [
        "bge",
        "e5", 
        "gte",
        "bert",
        "sentence",
        "instructor",
    ];
    
    encoder_indicators.iter().any(|&indicator| path_str.contains(indicator))
}

// 新增：专门处理编码器模型的函数
fn batch_encode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    
    // 对于编码器模型，使用encode而不是decode
    match ctx.encode(batch) {
        Ok(_) => {
            eprintln!("✅ 编码成功");
        }
        Err(e) => {
            eprintln!("❌ 编码失败，尝试decode: {}", e);
            // 如果encode失败，尝试decode作为fallback
            ctx.decode(batch).with_context(|| "encode和decode都失败了")?;
        }
    }

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();
    Ok(())
}

// 原有的decode函数，用于解码器模型
fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();
    Ok(())
}

// ...existing normalize function...

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}
