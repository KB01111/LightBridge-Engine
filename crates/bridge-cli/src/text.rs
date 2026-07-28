use std::fmt::Write;

use crate::{Aggregate, InspectionReport};

pub fn render_text(report: &InspectionReport) -> String {
    let mut output = String::new();

    writeln!(output, "Model").unwrap();
    writeln!(output, "  architecture: {}", report.general.architecture).unwrap();
    optional_line(&mut output, "name", report.general.name.as_deref());
    optional_line(&mut output, "license", report.general.license.as_deref());
    optional_line(&mut output, "size label", report.general.size_label.as_deref());
    optional_number(
        &mut output,
        "quantization version",
        report.general.quantization_version,
    );
    optional_number(&mut output, "file type", report.general.file_type);

    writeln!(output, "\nFiles").unwrap();
    for file in &report.files {
        writeln!(output, "  {}", file.path).unwrap();
        writeln!(output, "    shard ordinal: {}", file.ordinal).unwrap();
        writeln!(output, "    shard count: {}", file.count).unwrap();
        writeln!(output, "    version: {}", file.version).unwrap();
        writeln!(output, "    endianness: {}", file.endianness).unwrap();
        writeln!(output, "    metadata count: {}", file.metadata_count).unwrap();
        writeln!(output, "    alignment: {}", format_bytes(file.alignment)).unwrap();
        writeln!(output, "    logical size: {}", format_bytes(file.logical_size)).unwrap();
        writeln!(output, "    data offset: {}", format_bytes(file.data_offset)).unwrap();
    }

    writeln!(output, "\nGGUF").unwrap();
    writeln!(output, "  version: {}", report.gguf.version).unwrap();
    writeln!(output, "  endianness: {}", report.gguf.endianness).unwrap();
    writeln!(
        output,
        "  authoritative metadata count: {}",
        report.gguf.authoritative_metadata_count
    )
    .unwrap();
    writeln!(output, "  tensor count: {}", report.gguf.tensor_count).unwrap();
    writeln!(output, "  alignment: {}", format_bytes(report.gguf.alignment)).unwrap();
    writeln!(
        output,
        "  encoded tensor bytes: {}",
        format_bytes(report.gguf.encoded_tensor_bytes)
    )
    .unwrap();

    writeln!(output, "\nHy3").unwrap();
    writeln!(output, "  block count: {}", report.hy3.block_count).unwrap();
    writeln!(output, "  context length: {}", report.hy3.context_length).unwrap();
    writeln!(output, "  embedding length: {}", report.hy3.embedding_length).unwrap();
    writeln!(output, "  dense FFN length: {}", report.hy3.dense_ffn_length).unwrap();
    writeln!(output, "  expert FFN length: {}", report.hy3.expert_ffn_length).unwrap();
    writeln!(
        output,
        "  shared expert FFN length: {}",
        report.hy3.shared_expert_ffn_length
    )
    .unwrap();
    writeln!(output, "  attention heads: {}", report.hy3.attention_head_count).unwrap();
    writeln!(
        output,
        "  attention KV heads: {}",
        report.hy3.attention_kv_head_count
    )
    .unwrap();
    writeln!(output, "  key length: {}", report.hy3.key_length).unwrap();
    writeln!(output, "  value length: {}", report.hy3.value_length).unwrap();
    writeln!(output, "  RMS epsilon: {}", report.hy3.rms_epsilon).unwrap();
    writeln!(output, "  experts: {}", report.hy3.expert_count).unwrap();
    writeln!(output, "  top-k: {}", report.hy3.expert_used_count).unwrap();
    writeln!(
        output,
        "  expert weights norm: {}",
        report.hy3.expert_weights_norm
    )
    .unwrap();
    writeln!(
        output,
        "  expert gating function: {}",
        report.hy3.expert_gating_func
    )
    .unwrap();
    writeln!(
        output,
        "  expert weights scale: {}",
        report.hy3.expert_weights_scale
    )
    .unwrap();
    writeln!(output, "  RoPE base: {}", report.hy3.rope_base).unwrap();
    writeln!(output, "  RoPE scaling type: {}", report.hy3.rope_scaling_type).unwrap();
    writeln!(output, "  YaRN factor: {}", report.hy3.yarn_factor).unwrap();
    writeln!(
        output,
        "  YaRN original context: {}",
        report.hy3.yarn_original_context
    )
    .unwrap();
    writeln!(output, "  vocabulary size: {}", report.hy3.vocabulary_size).unwrap();
    writeln!(output, "  MTP: {}", report.hy3.has_mtp).unwrap();

    writeln!(output, "\nTokenizer").unwrap();
    optional_line(&mut output, "model", report.tokenizer.model.as_deref());
    optional_line(
        &mut output,
        "pretokenizer",
        report.tokenizer.pretokenizer.as_deref(),
    );
    writeln!(output, "  token count: {}", report.tokenizer.token_count).unwrap();
    optional_u64(&mut output, "merge count", report.tokenizer.merge_count);
    optional_u64(&mut output, "token type count", report.tokenizer.token_type_count);
    optional_number(&mut output, "BOS token ID", report.tokenizer.bos_token_id);
    optional_number(&mut output, "EOS token ID", report.tokenizer.eos_token_id);
    optional_number(&mut output, "padding token ID", report.tokenizer.padding_token_id);
    optional_number(
        &mut output,
        "separator token ID",
        report.tokenizer.separator_token_id,
    );
    writeln!(
        output,
        "  chat template present: {}",
        report.tokenizer.has_chat_template
    )
    .unwrap();

    writeln!(output, "\nTensor types").unwrap();
    for (name, aggregate) in &report.types {
        aggregate_line(&mut output, name, aggregate);
    }

    writeln!(output, "\nTensor roles").unwrap();
    aggregate_line(
        &mut output,
        "dense_layer_0_ffn",
        &report.tensors.dense_layer_0_ffn,
    );
    aggregate_line(&mut output, "routed_experts", &report.tensors.routed_experts);
    aggregate_line(&mut output, "shared_experts", &report.tensors.shared_experts);
    aggregate_line(&mut output, "attention", &report.tensors.attention);
    aggregate_line(&mut output, "routers", &report.tensors.routers);
    aggregate_line(&mut output, "embeddings", &report.tensors.embeddings);
    aggregate_line(&mut output, "norms", &report.tensors.norms);
    aggregate_line(&mut output, "output_total", &report.tensors.output);
    for (name, aggregate) in &report.roles {
        aggregate_line(&mut output, name, aggregate);
    }

    writeln!(output, "\nLayers").unwrap();
    for (layer, aggregate) in &report.layers {
        aggregate_line(&mut output, &layer.to_string(), aggregate);
    }

    writeln!(output, "\nExpert storage").unwrap();
    writeln!(output, "  expert count: {}", report.expert_storage.expert_count).unwrap();
    aggregate_line(&mut output, "routed banks", &report.expert_storage.routed_banks);
    aggregate_line(
        &mut output,
        "shared experts",
        &report.expert_storage.shared_experts,
    );
    for (key, projection) in &report.expert_storage.routed_projections {
        writeln!(output, "  {key}:").unwrap();
        writeln!(output, "    tensor count: {}", projection.tensor_count).unwrap();
        writeln!(output, "    expert count: {}", projection.expert_count).unwrap();
        writeln!(
            output,
            "    slab logical elements: {}",
            projection.slab_logical_elements
        )
        .unwrap();
        writeln!(output, "    slab bytes: {}", format_bytes(projection.slab_bytes)).unwrap();
    }

    writeln!(output, "\nExecution status").unwrap();
    writeln!(
        output,
        "  unsupported execution types: {}",
        report.unsupported_execution_types.join(", ")
    )
    .unwrap();

    if !report.warnings.is_empty() {
        writeln!(output, "\nWarnings").unwrap();
        for warning in &report.warnings {
            writeln!(output, "  - {warning}").unwrap();
        }
    }

    output
}

fn aggregate_line(output: &mut String, name: &str, aggregate: &Aggregate) {
    writeln!(
        output,
        "  {name}: {} tensors / {} logical elements / {}",
        aggregate.count,
        aggregate.logical_elements,
        format_bytes(aggregate.encoded_bytes)
    )
    .unwrap();
}

fn optional_line(output: &mut String, name: &str, value: Option<&str>) {
    writeln!(output, "  {name}: {}", value.unwrap_or("(not provided)")).unwrap();
}

fn optional_number(output: &mut String, name: &str, value: Option<u32>) {
    match value {
        Some(value) => writeln!(output, "  {name}: {value}").unwrap(),
        None => writeln!(output, "  {name}: (not provided)").unwrap(),
    }
}

fn optional_u64(output: &mut String, name: &str, value: Option<u64>) {
    match value {
        Some(value) => writeln!(output, "  {name}: {value}").unwrap(),
        None => writeln!(output, "  {name}: (not provided)").unwrap(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("TiB", 1_u64 << 40),
        ("GiB", 1_u64 << 30),
        ("MiB", 1_u64 << 20),
        ("KiB", 1_u64 << 10),
    ];
    let human = UNITS
        .into_iter()
        .find(|(_, threshold)| bytes >= *threshold)
        .map_or_else(
            || format!("{bytes} B"),
            |(unit, scale)| format!("{:.2} {unit}", bytes as f64 / scale as f64),
        );
    format!("{bytes} bytes ({human})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_format_is_locale_independent_and_iec_scaled() {
        assert_eq!(format_bytes(0), "0 bytes (0 B)");
        assert_eq!(format_bytes(1_536), "1536 bytes (1.50 KiB)");
        assert_eq!(format_bytes(96_014_150_912), "96014150912 bytes (89.42 GiB)");
    }
}
