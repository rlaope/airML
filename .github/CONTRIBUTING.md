# Contributing to airML

Thank you for your interest in contributing to airML!

## Getting Started

### Prerequisites

- Rust 1.75 or later
- ONNX Runtime (for testing)

### Building

```bash
# Clone the repository
git clone https://github.com/airml/airml.git
cd airml

# Build in debug mode
cargo build

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run clippy
cargo clippy --all-features

# Format code
cargo fmt
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** for your changes
3. **Make your changes** with clear, atomic commits
4. **Write tests** for new functionality
5. **Run tests and lints** before submitting
6. **Submit a pull request**

## Code Style

- Follow Rust standard style (enforced by `rustfmt`)
- Use meaningful variable and function names
- Write doc comments for public APIs
- Keep functions focused and small

## Commit Messages

Use conventional commits format:

```
type(scope): short description

Longer description if needed.
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add support for dynamic input shapes
fix(preprocess): handle images with alpha channel
docs(readme): add installation instructions
```

## Pull Request Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new functionality
- Ensure CI passes
- Request review from maintainers

## Architecture Overview

```
airML/
├── crates/
│   ├── airml-core/        # Core inference engine
│   │   ├── engine.rs      # InferenceEngine
│   │   ├── session.rs     # Session configuration
│   │   └── error.rs       # Error types
│   │
│   ├── airml-preprocess/  # Input preprocessing
│   │   ├── image.rs       # Image preprocessing
│   │   └── text.rs        # Text tokenization
│   │
│   ├── airml-providers/   # Execution providers
│   │   ├── cpu.rs         # CPU provider
│   │   └── coreml.rs      # CoreML provider
│   │
│   └── airml-embed/       # Model embedding
│
└── src/                   # CLI binary
    ├── main.rs            # Entry point
    ├── cli.rs             # Clap definitions
    └── commands/          # Command implementations
```

## Adding a New Execution Provider

1. Create a new file in `crates/airml-providers/src/`
2. Implement the provider configuration
3. Add to feature flags in `Cargo.toml`
4. Export in `lib.rs`
5. Add to auto-selection logic

## Adding a New Preprocessing Mode

1. Add configuration in `crates/airml-preprocess/src/image.rs`
2. Implement the preprocessing logic
3. Add a constructor method (e.g., `ImagePreprocessor::new_mode()`)
4. Add to CLI preset options

## Questions?

Open a [Discussion](https://github.com/airml/airml/discussions) for questions or ideas.
