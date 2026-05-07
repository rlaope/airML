# airml-fuzz

Fuzz targets for airML. Run with:

    cargo install cargo-fuzz
    cd fuzz && cargo +nightly fuzz run graph_parser

Targets:
- graph_parser: feeds arbitrary bytes into airml_tune::histogram_from_bytes.
                Must never panic. Ok results are validated only for type, not content.
