//! Serve command implementation
//!
//! Exposes an OpenAI-compatible embeddings HTTP API via `airml serve`.
//! The actual inference implementation is gated behind the `nlp` feature.

use anyhow::Result;

use crate::cli::ServeArgs;

// ---------------------------------------------------------------------------
// Feature-gated implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "nlp")]
mod inner {
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::Arc;

    use anyhow::{Context, Result};
    use axum::extract::{Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use serde::{Deserialize, Serialize};
    use tokio::sync::{Mutex, RwLock};
    use tower_http::cors::CorsLayer;
    use tracing::info;

    use airml_core::{InferenceEngine, SessionConfig};
    use airml_hub::{Fetcher, ModelCache, ModelUri};
    use airml_preprocess::TextPreprocessor;
    use airml_providers::auto_select_providers;

    use crate::cli::ServeArgs;

    // -----------------------------------------------------------------------
    // Shared state
    // -----------------------------------------------------------------------

    /// Per-model cached inference state.
    struct ModelState {
        engine: Mutex<InferenceEngine>,
        tokenizer: TextPreprocessor,
    }

    /// Application state shared across all request handlers.
    struct AppState {
        hub: Fetcher,
        /// Map from resolved model path (string) -> loaded engine + tokenizer.
        sessions: RwLock<HashMap<String, Arc<ModelState>>>,
        default_model: Option<String>,
        auth_token: Option<String>,
    }

    // -----------------------------------------------------------------------
    // OpenAI-compatible request / response types
    // -----------------------------------------------------------------------

    #[derive(Deserialize)]
    struct EmbeddingsRequest {
        model: Option<String>,
        input: StringOrArray,
        #[serde(default)]
        encoding_format: Option<String>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrArray {
        Single(String),
        Multiple(Vec<String>),
    }

    impl StringOrArray {
        fn into_vec(self) -> Vec<String> {
            match self {
                StringOrArray::Single(s) => vec![s],
                StringOrArray::Multiple(v) => v,
            }
        }
    }

    #[derive(Serialize)]
    struct EmbeddingObject {
        object: &'static str,
        embedding: Vec<f32>,
        index: usize,
    }

    #[derive(Serialize)]
    struct UsageInfo {
        prompt_tokens: usize,
        total_tokens: usize,
    }

    #[derive(Serialize)]
    struct EmbeddingsResponse {
        object: &'static str,
        data: Vec<EmbeddingObject>,
        model: String,
        usage: UsageInfo,
    }

    #[derive(Serialize)]
    struct ModelObject {
        id: String,
        object: &'static str,
        owned_by: &'static str,
    }

    #[derive(Serialize)]
    struct ModelsResponse {
        object: &'static str,
        data: Vec<ModelObject>,
    }

    #[derive(Serialize)]
    struct HealthResponse {
        status: &'static str,
    }

    #[derive(Serialize)]
    struct ModelInfoResponse {
        model: String,
        status: &'static str,
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    struct AppError {
        status: StatusCode,
        message: String,
        error_type: &'static str,
    }

    impl AppError {
        fn bad_request(msg: impl Into<String>) -> Self {
            Self {
                status: StatusCode::BAD_REQUEST,
                message: msg.into(),
                error_type: "invalid_request_error",
            }
        }

        fn internal(msg: impl Into<String>) -> Self {
            Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: msg.into(),
                error_type: "server_error",
            }
        }

        fn unauthorized() -> Self {
            Self {
                status: StatusCode::UNAUTHORIZED,
                message: "Invalid or missing Bearer token".to_owned(),
                error_type: "authentication_error",
            }
        }
    }

    #[derive(Serialize)]
    struct ErrorDetail {
        message: String,
        r#type: &'static str,
    }

    #[derive(Serialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }

    impl IntoResponse for AppError {
        fn into_response(self) -> axum::response::Response {
            let body = ErrorBody {
                error: ErrorDetail {
                    message: self.message,
                    r#type: self.error_type,
                },
            };
            (self.status, Json(body)).into_response()
        }
    }

    // -----------------------------------------------------------------------
    // Auth helper
    // -----------------------------------------------------------------------

    fn check_auth(headers: &HeaderMap, state: &AppState) -> Result<(), AppError> {
        let Some(expected) = &state.auth_token else {
            return Ok(());
        };

        let bearer = headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "));

        match bearer {
            Some(token) if token == expected => Ok(()),
            _ => Err(AppError::unauthorized()),
        }
    }

    // -----------------------------------------------------------------------
    // Handlers
    // -----------------------------------------------------------------------

    async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    /// GET /metrics — Prometheus text format.
    async fn metrics() -> impl IntoResponse {
        match crate::metrics::render() {
            Ok(body) => (
                StatusCode::OK,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "text/plain; version=0.0.4",
                )],
                body,
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("metrics error: {e}"),
            )
                .into_response(),
        }
    }

    async fn list_models(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
    ) -> Result<Json<ModelsResponse>, AppError> {
        check_auth(&headers, &state)?;

        let registry_models: Vec<ModelObject> = airml_hub::registry::all()
            .iter()
            .map(|e| ModelObject {
                id: e.id.to_owned(),
                object: "model",
                owned_by: "airml",
            })
            .collect();

        Ok(Json(ModelsResponse {
            object: "list",
            data: registry_models,
        }))
    }

    #[derive(Deserialize)]
    struct ModelInfoQuery {
        model: String,
    }

    async fn model_info(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
        Query(params): Query<ModelInfoQuery>,
    ) -> Result<Json<ModelInfoResponse>, AppError> {
        check_auth(&headers, &state)?;

        let uri = ModelUri::parse(&params.model)
            .map_err(|e| AppError::bad_request(format!("Invalid model URI: {e}")))?;

        let cached = match &uri {
            ModelUri::Registry(id) => {
                airml_hub::registry::lookup(id)
                    .map(|entry| {
                        let is_placeholder = entry.sha256 == "PLACEHOLDER_TO_VERIFY";
                        if is_placeholder {
                            false
                        } else {
                            let cache = if let Some(dir) = None::<PathBuf> {
                                ModelCache::new(dir)
                            } else {
                                ModelCache::new(ModelCache::default_root())
                            };
                            cache.contains(entry.sha256)
                        }
                    })
                    .unwrap_or(false)
            }
            ModelUri::LocalPath(p) => p.exists(),
            _ => false,
        };

        Ok(Json(ModelInfoResponse {
            model: params.model,
            status: if cached { "cached" } else { "uncached" },
        }))
    }

    async fn embeddings(
        State(state): State<Arc<AppState>>,
        headers: HeaderMap,
        Json(req): Json<EmbeddingsRequest>,
    ) -> Result<Json<EmbeddingsResponse>, AppError> {
        check_auth(&headers, &state)?;

        // Validate encoding_format
        if let Some(ref fmt) = req.encoding_format {
            if fmt != "float" {
                return Err(AppError::bad_request(format!(
                    "encoding_format '{fmt}' is not supported; only 'float' is accepted"
                )));
            }
        }

        // Resolve model
        let model_id = req
            .model
            .or_else(|| state.default_model.clone())
            .ok_or_else(|| {
                AppError::bad_request("'model' field is required (no default model configured)")
            })?;

        let texts = req.input.into_vec();
        if texts.is_empty() {
            return Err(AppError::bad_request("'input' must not be empty"));
        }

        // Get or create the cached model state
        let model_arc = get_or_load_model(&state, &model_id).await?;

        // Run inference inside spawn_blocking (ORT is blocking)
        let texts_clone = texts.clone();
        let embeddings_result = tokio::task::spawn_blocking(move || {
            let mut engine = model_arc.engine.blocking_lock();
            embed_texts_inner(&mut engine, &model_arc.tokenizer, &texts_clone)
        })
        .await
        .map_err(|e| AppError::internal(format!("Task join error: {e}")))?
        .map_err(|e| AppError::internal(format!("Inference error: {e}")))?;

        let total_tokens: usize = embeddings_result.iter().map(|(_, t)| t).sum();

        let data: Vec<EmbeddingObject> = embeddings_result
            .into_iter()
            .enumerate()
            .map(|(index, (embedding, _))| EmbeddingObject {
                object: "embedding",
                embedding,
                index,
            })
            .collect();

        Ok(Json(EmbeddingsResponse {
            object: "list",
            data,
            model: model_id,
            usage: UsageInfo {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        }))
    }

    // -----------------------------------------------------------------------
    // Model loading helpers
    // -----------------------------------------------------------------------

    async fn get_or_load_model(
        state: &AppState,
        model_id: &str,
    ) -> Result<Arc<ModelState>, AppError> {
        // Fast path: already loaded
        {
            let sessions = state.sessions.read().await;
            if let Some(ms) = sessions.get(model_id) {
                return Ok(Arc::clone(ms));
            }
        }

        // Slow path: resolve + load (blocking I/O)
        let model_id_owned = model_id.to_owned();
        let uri = ModelUri::parse(&model_id_owned)
            .map_err(|e| AppError::bad_request(format!("Invalid model URI: {e}")))?;

        // Fetcher::resolve_to_path is blocking
        let fetcher_ref = &state.hub;
        let model_path = tokio::task::spawn_blocking({
            // We need to send the fetcher into the blocking task.
            // Fetcher is not Clone, so we re-create one here using the same
            // cache root that was configured at startup.
            let uri_clone = uri.clone();
            let cache_root = ModelCache::default_root();
            move || {
                let fetcher = Fetcher::with_cache(ModelCache::new(cache_root));
                fetcher
                    .resolve_to_path(&uri_clone)
                    .map_err(|e| anyhow::anyhow!("Failed to resolve model: {e}"))
            }
        })
        .await
        .map_err(|e| AppError::internal(format!("Task join error: {e}")))?
        .map_err(|e| AppError::internal(e.to_string()))?;

        // Look for tokenizer.json alongside the model file
        let tokenizer_path = model_path
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .filter(|p| p.exists())
            .or_else(|| {
                // Also try the same directory as the model
                model_path
                    .with_file_name("tokenizer.json")
                    .exists()
                    .then(|| model_path.with_file_name("tokenizer.json"))
            });

        let _ = fetcher_ref; // silence unused warning

        let model_path_clone = model_path.clone();
        let model_state = tokio::task::spawn_blocking(move || -> Result<ModelState> {
            let providers = auto_select_providers();
            let config = SessionConfig::new().with_providers(providers);

            let engine = InferenceEngine::from_file_with_config(&model_path_clone, config)
                .with_context(|| format!("Failed to load model from {}", model_path_clone.display()))?;

            // Load tokenizer — try alongside model first, then default path
            let tokenizer = if let Some(tok_path) = tokenizer_path {
                TextPreprocessor::from_file(&tok_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
                    .with_max_length(512)
            } else {
                return Err(anyhow::anyhow!(
                    "No tokenizer.json found alongside model at {}. \
                     Ensure tokenizer.json is in the same directory as the model file.",
                    model_path_clone.display()
                ));
            };

            Ok(ModelState {
                engine: Mutex::new(engine),
                tokenizer,
            })
        })
        .await
        .map_err(|e| AppError::internal(format!("Task join error: {e}")))?
        .map_err(|e| AppError::internal(e.to_string()))?;

        let model_arc = Arc::new(model_state);

        // Insert into cache
        {
            let mut sessions = state.sessions.write().await;
            sessions
                .entry(model_id_owned)
                .or_insert_with(|| Arc::clone(&model_arc));
        }

        Ok(model_arc)
    }

    /// Run tokenization + inference for a batch of texts.
    /// Returns `(embedding_vec, token_count)` per text.
    fn embed_texts_inner(
        engine: &mut InferenceEngine,
        tokenizer: &TextPreprocessor,
        texts: &[String],
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let tokenized = tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

            let token_count = tokenized.attention_mask.iter().filter(|&&m| m == 1).count();
            let (input_ids, attention_mask) = tokenized.to_array();

            let n_inputs = engine.inputs().len();
            let outputs = if n_inputs >= 2 {
                engine
                    .run_multiple(vec![
                        input_ids.into_dyn().mapv(|x| x as f32),
                        attention_mask.clone().into_dyn().mapv(|x| x as f32),
                    ])
                    .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?
            } else {
                engine
                    .run(input_ids.into_dyn().mapv(|x| x as f32))
                    .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?
            };

            let output = outputs
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Model produced no output"))?;

            let shape = output.shape().to_vec();
            let mask_flat: Vec<f32> = attention_mask
                .iter()
                .map(|&m| m as f32)
                .collect();

            let embedding = match shape.len() {
                2 => {
                    // [batch=1, hidden] — direct embedding
                    output.iter().copied().collect()
                }
                3 => {
                    // [batch=1, seq_len, hidden] — mean-pool over real tokens
                    let seq_len = shape[1];
                    let hidden = shape[2];
                    let mut pooled = vec![0.0f32; hidden];
                    let mut real_token_count = 0.0f32;

                    for i in 0..seq_len {
                        let mask_val = if i < mask_flat.len() { mask_flat[i] } else { 0.0 };
                        if mask_val > 0.0 {
                            real_token_count += 1.0;
                            for j in 0..hidden {
                                pooled[j] += output[[0, i, j]];
                            }
                        }
                    }

                    if real_token_count > 0.0 {
                        for v in &mut pooled {
                            *v /= real_token_count;
                        }
                    }
                    pooled
                }
                _ => output.iter().copied().collect(),
            };

            // L2 normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized = if norm > 0.0 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            };

            results.push((normalized, token_count));
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------

    pub fn execute(args: &ServeArgs) -> Result<()> {
        // Tracing is initialised in main() before subcommand dispatch.

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("Failed to build Tokio runtime")?;

        rt.block_on(run_server(args))
    }

    async fn run_server(args: &ServeArgs) -> Result<()> {
        let hub = if let Some(cache_dir) = &args.cache_dir {
            Fetcher::with_cache(ModelCache::new(cache_dir.clone()))
        } else {
            Fetcher::new()
        };

        let state = Arc::new(AppState {
            hub,
            sessions: RwLock::new(HashMap::new()),
            default_model: args.default_model.clone(),
            auth_token: args.auth_token.clone(),
        });

        let app = Router::new()
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/models", get(list_models))
            .route("/v1/embeddings/info", get(model_info))
            .route("/healthz", get(health))
            .route("/metrics", get(metrics))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let addr: SocketAddr = args
            .bind
            .parse()
            .with_context(|| format!("Invalid bind address: '{}'", args.bind))?;

        println!("airml serve listening on http://{addr}");
        println!(
            "  curl -s http://{addr}/v1/embeddings \\\n    \
             -H 'Content-Type: application/json' \\\n    \
             -d '{{\"model\":\"bge-small-en\",\"input\":[\"Hello, world.\"]}}'"
        );

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("Failed to bind to {addr}"))?;

        info!("Listening on {addr}");

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        info!("Server shut down cleanly");
        Ok(())
    }

    async fn shutdown_signal() {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for ctrl_c signal");
        info!("Received shutdown signal");
    }
}

// ---------------------------------------------------------------------------
// Public entry point — always compiled; implementation gated on `nlp`
// ---------------------------------------------------------------------------

/// Execute the serve command.
pub fn execute(args: &ServeArgs) -> Result<()> {
    #[cfg(feature = "nlp")]
    {
        inner::execute(args)
    }

    #[cfg(not(feature = "nlp"))]
    {
        let _ = args;
        anyhow::bail!(
            "`airml serve` requires the `nlp` feature.\n\
             Rebuild with: cargo build --features nlp\n\
             Or install with: cargo install airml --features nlp"
        )
    }
}
