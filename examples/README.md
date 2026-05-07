# airML Examples

Run any example with:

    cargo run --example <name> -- <args>

Each example is self-contained and < 100 LOC.

## 01_image_classify
Image classification on an ONNX model.

    cargo run --example 01_image_classify --features=coreml -- model.onnx cat.jpg

Loads an ONNX image classification model, resizes the input image to 224x224,
applies ImageNet normalisation, runs inference, and prints the top-5 class
indices with softmax probabilities.

## 02_text_embed
Text embedding inference (synthetic tokens for demo).

    cargo run --example 02_text_embed -- model.onnx 16

Loads a BERT-style ONNX embedding model and feeds it synthetic CLS/SEP token
sequences.  Prints the first 8 dimensions of the output embedding.  Switch to
real tokenisation by adding the `nlp` feature and a tokenizer file.

## 03_embedded_model
Loading a model from in-memory bytes (template for `include_bytes!` embedding).

    cargo run --example 03_embedded_model -- model.onnx

Demonstrates the `airml-embed` API: reads model bytes from a file path at
runtime (a drop-in stand-in for compile-time `include_bytes!`), wraps them in
`EmbeddedModel`, converts to an `InferenceEngine`, and prints model metadata
(input/output names, shapes, dtypes).

## 04_embedded_compiletime
Canonical compile-time model embedding with the `embed_model!` macro.

    cargo run --example 04_embedded_compiletime

Uses `embed_model!(IDENTITY, "../../models/synthetic-identity.onnx")` to bake
a 97-byte synthetic ONNX Identity model into the binary at compile time via
`include_bytes!`.  No filesystem access at runtime.  This is the recommended
pattern for shipping a fixed model alongside a binary.
