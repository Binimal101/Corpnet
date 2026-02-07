# requires sentence_transformers>=3.2.0
from sentence_transformers import SentenceTransformer, export_optimized_onnx_model, export_dynamic_quantized_onnx_model

# The model to export to ONNX (+ optimize, quantize), OpenVINO
model_id = "mixedbread-ai/mxbai-embed-large-v1"
# Where to save the exported models locally
output_dir = model_id.replace("/", "-")

onnx_model = SentenceTransformer(model_id, backend="onnx", model_kwargs={"export": True})
onnx_model.save_pretrained(output_dir)

for optimization_config in ["O1", "O2", "O3", "O4"]:
    export_optimized_onnx_model(
        onnx_model,
        optimization_config=optimization_config,
        model_name_or_path=output_dir,
    )

for quantization_config in ['arm64', 'avx2', 'avx512', 'avx512_vnni']:
    export_dynamic_quantized_onnx_model(
        onnx_model,
        quantization_config=quantization_config,
        model_name_or_path=output_dir,
    )

openvino_model = SentenceTransformer(model_id, backend="openvino")
openvino_model.save_pretrained(output_dir)
export_to_hub.py