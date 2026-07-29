use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bridge_runtime::{
    CancellationToken, CausalModel, ChatCompletion, Hy3ChatEngine, SamplingConfig, StopReason,
};
use bridge_tokenizer::{ChatMessage, ChatTemplateOptions, ReasoningEffort, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

pub const MODEL_ID: &str = "hy3-1m-iq2-m";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_STOP_SEQUENCES: usize = 4;
static NEXT_COMPLETION_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub max_concurrent_requests: usize,
    pub max_request_bytes: usize,
    pub max_new_tokens: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
            max_concurrent_requests: 1,
            max_request_bytes: DEFAULT_REQUEST_BYTES,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl ServerConfig {
    pub fn validate(self) -> Result<(), ServerError> {
        for (field, value) in [
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("max_request_bytes", self.max_request_bytes),
            ("max_new_tokens", self.max_new_tokens),
        ] {
            if value == 0 {
                return Err(ServerError::ZeroLimit { field });
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Hy3ChatEngine>,
    slots: Arc<Semaphore>,
    max_new_tokens: usize,
}

pub fn router(engine: Arc<Hy3ChatEngine>, config: ServerConfig) -> Result<Router, ServerError> {
    config.validate()?;
    let state = AppState {
        engine,
        slots: Arc::new(Semaphore::new(config.max_concurrent_requests)),
        max_new_tokens: config.max_new_tokens,
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/model", get(model_info))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(config.max_request_bytes))
        .with_state(state))
}

pub async fn serve(engine: Hy3ChatEngine, config: ServerConfig) -> Result<(), ServerError> {
    serve_shared(Arc::new(engine), config).await
}

pub async fn serve_shared(engine: Arc<Hy3ChatEngine>, config: ServerConfig) -> Result<(), ServerError> {
    let app = router(engine, config)?;
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .map_err(|source| ServerError::Io {
            operation: "bind server listener",
            source,
        })?;
    tracing::info!(address = %config.bind, "LightBridge server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|source| ServerError::Io {
            operation: "serve HTTP requests",
            source,
        })
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to install Ctrl+C handler");
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "ready": true,
        "model": MODEL_ID,
        "backend": state.engine.model().backend_name(),
        "cpu_threads": state.engine.model().cpu_threads(),
        "context_length": state.engine.model().context_length(),
    }))
}

async fn models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: MODEL_ID,
            object: "model",
            owned_by: "lightbridge",
        }],
    })
}

async fn model_info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let model = state.engine.model();
    let cache = model
        .cache_stats()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(json!({
        "id": MODEL_ID,
        "architecture": "hy_v3",
        "backend": model.backend_name(),
        "cpu_threads": model.cpu_threads(),
        "context_length": model.context_length(),
        "checkpoint_context_length": model.config().context_length,
        "vocabulary_size": model.config().vocabulary_size,
        "block_count": model.config().block_count,
        "expert_count": model.config().expert_count,
        "expert_used_count": model.config().expert_used_count,
        "resident_weight_bytes": model.resident_weight_bytes(),
        "source_paths": model.source_paths(),
        "source_sha256": model.source_sha256(),
        "expert_cache": cache,
        "capabilities": {
            "cpu_simd_active": model.cpu_simd_active(),
            "parallel_expert_prefetch": model.parallel_expert_prefetch(),
            "persistent_sessions": true,
            "grouped_prefill": false,
            "cuda": false,
            "mtp": false,
        },
    })))
}

async fn tokenize(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let tokenizer = state.engine.tokenizer();
    let token_ids = match (request.text, request.messages) {
        (Some(text), None) => tokenizer
            .encode(&text)
            .map_err(|error| ApiError::bad_request(error.to_string()))?,
        (None, Some(messages)) => {
            let messages = convert_messages(messages)?;
            let template = template_options(request.reasoning_effort, request.tools)?;
            tokenizer
                .format_and_encode(&messages, &template)
                .map_err(|error| ApiError::bad_request(error.to_string()))?
        }
        _ => {
            return Err(ApiError::bad_request(
                "provide exactly one of `text` or `messages`",
            ))
        }
    };
    Ok(Json(TokenizeResponse {
        count: token_ids.len(),
        token_ids,
    }))
}

async fn detokenize(
    State(state): State<AppState>,
    Json(request): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    let text = state
        .engine
        .tokenizer()
        .decode(&request.token_ids, request.skip_special_tokens)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    Ok(Json(DetokenizeResponse { text }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if request.model.as_deref().is_some_and(|model| model != MODEL_ID) {
        return Err(ApiError::not_found(format!(
            "model {:?} is not loaded",
            request.model.as_deref().unwrap_or_default()
        )));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("only `n: 1` is supported"));
    }
    let prepared = PreparedRequest::new(request, state.max_new_tokens, state.engine.tokenizer())?;
    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    let request_id = completion_id();
    if prepared.stream {
        return Ok(stream_completion(state.engine, prepared, permit, request_id));
    }

    let engine = state.engine;
    let completion = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        engine.complete_with_stops(
            &prepared.messages,
            &prepared.template,
            prepared.sampling,
            &prepared.stop_sequences,
            &CancellationToken::new(),
            |_| ControlFlow::Continue(()),
        )
    })
    .await
    .map_err(|error| ApiError::internal(format!("inference worker failed: {error}")))?
    .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(non_streaming_response(request_id, completion)).into_response())
}

fn stream_completion(
    engine: Arc<Hy3ChatEngine>,
    prepared: PreparedRequest,
    permit: tokio::sync::OwnedSemaphorePermit,
    request_id: String,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let include_usage = prepared.include_usage;
        let buffer_structured_output = prepared.buffer_structured_output;
        let mut opening = json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": unix_seconds(),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": null
            }]
        });
        add_null_usage(&mut opening, include_usage);
        if send_event(&sender, &opening).is_err() {
            return;
        }
        let cancellation = CancellationToken::new();
        let result = engine.complete_with_stops(
            &prepared.messages,
            &prepared.template,
            prepared.sampling,
            &prepared.stop_sequences,
            &cancellation,
            |chunk| {
                if buffer_structured_output {
                    return ControlFlow::Continue(());
                }
                let mut payload = json!({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": unix_seconds(),
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": null
                    }]
                });
                add_null_usage(&mut payload, include_usage);
                if send_event(&sender, &payload).is_err() {
                    cancellation.cancel();
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        match result {
            Ok(completion) => {
                let usage = completion_usage(&completion);
                if buffer_structured_output {
                    let mut structured = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": unix_seconds(),
                        "model": MODEL_ID,
                        "choices": [{
                            "index": 0,
                            "delta": assistant_delta(&request_id, &completion.assistant),
                            "finish_reason": null
                        }]
                    });
                    add_null_usage(&mut structured, include_usage);
                    let _ = send_event(&sender, &structured);
                }
                let mut closing = json!({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": unix_seconds(),
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": completion_finish_reason(&completion)
                    }]
                });
                add_null_usage(&mut closing, include_usage);
                let _ = send_event(&sender, &closing);
                if include_usage {
                    let _ = send_event(
                        &sender,
                        &json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": unix_seconds(),
                            "model": MODEL_ID,
                            "choices": [],
                            "usage": usage
                        }),
                    );
                }
            }
            Err(error) => {
                let _ = send_event(
                    &sender,
                    &json!({"error": {"message": error.to_string(), "type": "inference_error"}}),
                );
            }
        }
        let _ = sender.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(receiver))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

fn send_event(sender: &mpsc::Sender<Result<Event, Infallible>>, payload: &Value) -> Result<(), ()> {
    let data = serde_json::to_string(payload).map_err(|_| ())?;
    sender
        .blocking_send(Ok(Event::default().data(data)))
        .map_err(|_| ())
}

fn add_null_usage(payload: &mut Value, include_usage: bool) {
    if include_usage {
        payload["usage"] = Value::Null;
    }
}

fn non_streaming_response(id: String, completion: ChatCompletion) -> ChatCompletionResponse {
    let usage = completion_usage(&completion);
    let finish_reason = completion_finish_reason(&completion);
    let message = assistant_message(&id, &completion.assistant);
    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_seconds(),
        model: MODEL_ID,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
    }
}

fn assistant_message(id: &str, output: &bridge_tokenizer::AssistantOutput) -> AssistantMessage {
    AssistantMessage {
        role: "assistant",
        content: if output.content.is_empty() && !output.tool_calls.is_empty() {
            None
        } else {
            Some(output.content.clone())
        },
        reasoning_content: output.reasoning.clone(),
        tool_calls: response_tool_calls(id, &output.tool_calls),
    }
}

fn assistant_delta(id: &str, output: &bridge_tokenizer::AssistantOutput) -> Value {
    let mut delta = serde_json::Map::new();
    if !output.content.is_empty() {
        delta.insert("content".into(), Value::String(output.content.clone()));
    }
    if let Some(reasoning) = &output.reasoning {
        delta.insert("reasoning_content".into(), Value::String(reasoning.clone()));
    }
    if !output.tool_calls.is_empty() {
        delta.insert(
            "tool_calls".into(),
            Value::Array(
                response_tool_calls(id, &output.tool_calls)
                    .into_iter()
                    .enumerate()
                    .map(|(index, call)| {
                        json!({
                            "index": index,
                            "id": call.id,
                            "type": call.kind,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    Value::Object(delta)
}

fn response_tool_calls(id: &str, calls: &[ToolCall]) -> Vec<AssistantToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| AssistantToolCall {
            id: format!("call_{}_{index}", id.trim_start_matches("chatcmpl-")),
            kind: "function",
            function: AssistantFunctionCall {
                name: call.name.clone(),
                arguments: Value::Object(call.arguments.clone().into_iter().collect()).to_string(),
            },
        })
        .collect()
}

fn completion_usage(completion: &ChatCompletion) -> Usage {
    Usage {
        prompt_tokens: completion.prompt_token_ids.len(),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: completion.cached_prompt_tokens,
        },
        completion_tokens: completion.generation.token_ids.len(),
        total_tokens: completion
            .prompt_token_ids
            .len()
            .saturating_add(completion.generation.token_ids.len()),
    }
}

fn finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::StopToken(_) | StopReason::StopSequence => "stop",
        StopReason::MaxTokens | StopReason::ContextLength => "length",
        StopReason::Cancelled | StopReason::Callback => "cancelled",
    }
}

fn completion_finish_reason(completion: &ChatCompletion) -> &'static str {
    if completion.assistant.tool_calls.is_empty() {
        finish_reason(completion.generation.stop_reason)
    } else {
        "tool_calls"
    }
}

#[derive(Debug)]
struct PreparedRequest {
    messages: Vec<ChatMessage>,
    template: ChatTemplateOptions,
    sampling: SamplingConfig,
    stop_sequences: Vec<String>,
    include_usage: bool,
    buffer_structured_output: bool,
    stream: bool,
}

impl PreparedRequest {
    fn new(
        request: ChatCompletionRequest,
        server_max_new_tokens: usize,
        tokenizer: &bridge_tokenizer::Hy3Tokenizer,
    ) -> Result<Self, ApiError> {
        let max_new_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        if max_new_tokens == 0 || max_new_tokens > server_max_new_tokens {
            return Err(ApiError::bad_request(format!(
                "max tokens must be within 1..={server_max_new_tokens}"
            )));
        }
        if request
            .temperature
            .is_some_and(|temperature| !temperature.is_finite() || !(0.0..=2.0).contains(&temperature))
        {
            return Err(ApiError::bad_request(
                "temperature must be finite and within 0..=2",
            ));
        }
        let messages = convert_messages(request.messages)?;
        if messages.is_empty() {
            return Err(ApiError::bad_request("messages must not be empty"));
        }
        let template = template_options(request.reasoning_effort, request.tools)?;
        let mut sampling = SamplingConfig {
            max_new_tokens,
            temperature: request.temperature.unwrap_or(0.9),
            top_k: request.top_k.unwrap_or(0),
            top_p: request.top_p.unwrap_or(1.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            repeat_last_n: request.repetition_window.unwrap_or(64),
            seed: request.seed.unwrap_or(0),
            ..SamplingConfig::default()
        };
        sampling.emit_stop_token = false;
        sampling
            .validate(tokenizer.vocabulary_size())
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let stop_sequences = request.stop.map(StopSequences::into_vec).unwrap_or_default();
        if stop_sequences.len() > MAX_STOP_SEQUENCES {
            return Err(ApiError::bad_request(format!(
                "at most {MAX_STOP_SEQUENCES} stop sequences are supported"
            )));
        }
        if stop_sequences.iter().any(String::is_empty) {
            return Err(ApiError::bad_request("stop sequences must not be empty"));
        }
        let include_usage = request
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false);
        if include_usage && !request.stream {
            return Err(ApiError::bad_request(
                "stream_options may only be used when stream is true",
            ));
        }
        let buffer_structured_output =
            !template.tools.is_empty() || template.reasoning_effort != ReasoningEffort::NoThink;
        Ok(Self {
            messages,
            template,
            sampling,
            stop_sequences,
            include_usage,
            buffer_structured_output,
            stream: request.stream,
        })
    }
}

fn template_options(
    effort: Option<ReasoningEffort>,
    tools: Vec<Value>,
) -> Result<ChatTemplateOptions, ApiError> {
    if tools.iter().any(|tool| !tool.is_object()) {
        return Err(ApiError::bad_request("every tool must be a JSON object"));
    }
    Ok(ChatTemplateOptions {
        reasoning_effort: effort.unwrap_or_default(),
        tools,
        ..ChatTemplateOptions::default()
    })
}

fn convert_messages(messages: Vec<ApiMessage>) -> Result<Vec<ChatMessage>, ApiError> {
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            let content = message_content(message.content, index)?;
            match message.role.as_str() {
                "system" | "developer" => Ok(ChatMessage::system(content)),
                "user" => Ok(ChatMessage::user(content)),
                "tool" => Ok(ChatMessage::tool(content)),
                "assistant" => {
                    let mut calls = Vec::new();
                    calls
                        .try_reserve_exact(message.tool_calls.len())
                        .map_err(|_| ApiError::internal("failed to reserve tool calls"))?;
                    for (call_index, call) in message.tool_calls.into_iter().enumerate() {
                        if call.kind != "function" {
                            return Err(ApiError::bad_request(format!(
                                "message {index} tool call {call_index} has unsupported type {:?}",
                                call.kind
                            )));
                        }
                        let arguments: Value =
                            serde_json::from_str(&call.function.arguments).map_err(|error| {
                                ApiError::bad_request(format!(
                                    "message {index} tool call {call_index} arguments are not JSON: {error}"
                                ))
                            })?;
                        let Value::Object(arguments) = arguments else {
                            return Err(ApiError::bad_request(format!(
                                "message {index} tool call {call_index} arguments must be a JSON object"
                            )));
                        };
                        calls.push(ToolCall {
                            name: call.function.name,
                            arguments: arguments.into_iter().collect::<BTreeMap<_, _>>(),
                        });
                    }
                    Ok(ChatMessage::Assistant {
                        content,
                        reasoning: message.reasoning_content,
                        tool_calls: calls,
                    })
                }
                role => Err(ApiError::bad_request(format!(
                    "message {index} has unsupported role {role:?}"
                ))),
            }
        })
        .collect()
}

fn message_content(content: Option<Value>, index: usize) -> Result<String, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(content)) => Ok(content),
        Some(Value::Array(parts)) => {
            let mut output = String::new();
            for (part_index, part) in parts.into_iter().enumerate() {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "message {index} content part {part_index} is not a text part"
                    ))
                })?;
                output.push_str(text);
            }
            Ok(output)
        }
        Some(_) => Err(ApiError::bad_request(format!(
            "message {index} content must be a string, null, or text-part array"
        ))),
    }
}

fn completion_id() -> String {
    let ordinal = NEXT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{:x}-{ordinal:x}", unix_nanos())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<ApiMessage>>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub tools: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub token_ids: Vec<u32>,
    pub count: usize,
}

#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    pub token_ids: Vec<u32>,
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_window: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default, alias = "reasoning")]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ApiToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct ApiToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ApiFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct ApiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<AssistantToolCall>,
}

#[derive(Debug, Serialize)]
struct AssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: AssistantFunctionCall,
}

#[derive(Debug, Serialize)]
struct AssistantFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    prompt_tokens_details: PromptTokensDetails,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct PromptTokensDetails {
    cached_tokens: usize,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            kind: "not_found_error",
        }
    }

    fn busy() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "all bounded inference slots are busy".into(),
            kind: "rate_limit_error",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": self.kind,
                    "param": null,
                    "code": null
                }
            })),
        )
            .into_response()
    }
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("server limit {field} must be greater than zero")]
    ZeroLimit { field: &'static str },
    #[error("failed to {operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::Body;
    use bridge_kernels_reference::ReferenceExecutionMode;
    use bridge_runtime::{GenerationOutcome, GenerationStats, Hy3ScalarModel, Hy3ScalarOptions};
    use bridge_test_model::ReducedHy3Model;
    use http::{Method, Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    struct TemporaryModel {
        path: PathBuf,
    }

    impl TemporaryModel {
        fn write(bytes: &[u8]) -> Self {
            let ordinal = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "lightbridge-server-{}-{ordinal}.gguf",
                std::process::id()
            ));
            fs::write(&path, bytes).unwrap();
            Self { path }
        }
    }

    impl Drop for TemporaryModel {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(value) => {
                builder = builder.header("content-type", "application/json");
                Body::from(serde_json::to_vec(&value).unwrap())
            }
            None => Body::empty(),
        };
        builder.body(body).unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn reduced_app() -> (TemporaryModel, Router) {
        let reference = ReducedHy3Model::new().unwrap();
        let profile = reference.profile().unwrap();
        let file = TemporaryModel::write(&reference.gguf_bytes_with_chat_tokenizer().unwrap());
        let parsed = bridge_gguf::open(&file.path).unwrap();
        let tokenizer = bridge_tokenizer::Hy3Tokenizer::from_gguf(&parsed).unwrap();
        assert_eq!(tokenizer.encode("a").unwrap(), [0]);
        let model = Hy3ScalarModel::open_profile_for_testing(
            &file.path,
            &profile,
            Hy3ScalarOptions {
                context_capacity: 64,
                kv_page_tokens: 8,
                expert_cache_bytes: 2 * 1024 * 1024,
                cache_admit_after_requests: 1,
                execution_mode: ReferenceExecutionMode::LlamaQ8K,
                ..Hy3ScalarOptions::default()
            },
        )
        .unwrap();
        let engine = Hy3ChatEngine::from_parts(model, tokenizer).unwrap();
        let app = router(
            Arc::new(engine),
            ServerConfig {
                max_request_bytes: 512,
                max_new_tokens: 4,
                ..ServerConfig::default()
            },
        )
        .unwrap();
        (file, app)
    }

    #[test]
    fn converts_openai_messages_and_tool_calls_to_hy3_chat_semantics() {
        let messages = convert_messages(vec![
            ApiMessage {
                role: "developer".into(),
                content: Some(Value::String("Be exact.".into())),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ApiMessage {
                role: "assistant".into(),
                content: Some(Value::String(String::new())),
                reasoning_content: Some("Need weather.".into()),
                tool_calls: vec![ApiToolCall {
                    id: Some("call_1".into()),
                    kind: "function".into(),
                    function: ApiFunctionCall {
                        name: "weather".into(),
                        arguments: r#"{"city":"Stockholm"}"#.into(),
                    },
                }],
            },
        ])
        .unwrap();
        assert!(matches!(messages[0], ChatMessage::System { .. }));
        let ChatMessage::Assistant { tool_calls, .. } = &messages[1] else {
            panic!("expected assistant message");
        };
        assert_eq!(tool_calls[0].name, "weather");
        assert_eq!(tool_calls[0].arguments["city"], "Stockholm");
    }

    #[test]
    fn request_limits_and_unsupported_shapes_are_rejected() {
        assert!(matches!(
            ServerConfig {
                max_concurrent_requests: 0,
                ..ServerConfig::default()
            }
            .validate(),
            Err(ServerError::ZeroLimit {
                field: "max_concurrent_requests"
            })
        ));
        assert!(convert_messages(vec![ApiMessage {
            role: "image".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
        }])
        .is_err());
    }

    #[test]
    fn structured_assistant_output_serializes_as_openai_tool_calls() {
        let mut arguments = BTreeMap::new();
        arguments.insert("city".into(), json!("Stockholm"));
        let completion = ChatCompletion {
            prompt_token_ids: vec![1, 2],
            cached_prompt_tokens: 0,
            text: String::new(),
            raw_text: String::new(),
            assistant: bridge_tokenizer::AssistantOutput {
                content: String::new(),
                reasoning: Some("Need current weather.".into()),
                tool_calls: vec![ToolCall {
                    name: "weather".into(),
                    arguments,
                }],
            },
            structured_output_error: None,
            generation: GenerationOutcome {
                token_ids: vec![3, 4],
                stop_reason: StopReason::StopToken(2),
                stats: GenerationStats {
                    prompt_tokens: 2,
                    generated_tokens: 2,
                    prefill_duration: Duration::ZERO,
                    decode_duration: Duration::ZERO,
                    total_duration: Duration::ZERO,
                },
            },
        };
        let response = non_streaming_response("chatcmpl-test".into(), completion);
        let response = serde_json::to_value(response).unwrap();
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(response["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            response["choices"][0]["message"]["reasoning_content"],
            "Need current weather."
        );
        let call = &response["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["id"], "call_test_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "weather");
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            json!({"city": "Stockholm"})
        );
    }

    #[tokio::test]
    async fn reduced_model_serves_health_tokenization_and_completion_routes() {
        let (_file, app) = reduced_app();

        let health = app
            .clone()
            .oneshot(request(Method::GET, "/health", None))
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let health = json_body(health).await;
        assert_eq!(health["ready"], true);
        assert_eq!(health["backend"], "scalar_reference_q8_k");
        assert!(health["cpu_threads"].is_null());
        assert_eq!(health["context_length"], 64);

        let model_info = app
            .clone()
            .oneshot(request(Method::GET, "/v1/model", None))
            .await
            .unwrap();
        assert_eq!(model_info.status(), StatusCode::OK);
        let model_info = json_body(model_info).await;
        assert_eq!(model_info["backend"], "scalar_reference_q8_k");
        assert_eq!(model_info["capabilities"]["persistent_sessions"], true);
        assert_eq!(model_info["capabilities"]["grouped_prefill"], false);

        let tokenized = app
            .clone()
            .oneshot(request(Method::POST, "/v1/tokenize", Some(json!({"text": "a"}))))
            .await
            .unwrap();
        assert_eq!(tokenized.status(), StatusCode::OK);
        assert_eq!(json_body(tokenized).await, json!({"token_ids": [0], "count": 1}));

        let detokenized = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/detokenize",
                Some(json!({"token_ids": [0]})),
            ))
            .await
            .unwrap();
        assert_eq!(detokenized.status(), StatusCode::OK);
        assert_eq!(json_body(detokenized).await, json!({"text": "a"}));

        let completion_request = json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "a"}],
            "temperature": 0.0,
            "max_tokens": 4
        });
        let completion = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                Some(completion_request.clone()),
            ))
            .await
            .unwrap();
        assert_eq!(completion.status(), StatusCode::OK);
        let completion = json_body(completion).await;
        assert_eq!(completion["object"], "chat.completion");
        assert_eq!(completion["model"], MODEL_ID);
        assert_eq!(completion["choices"][0]["message"]["role"], "assistant");
        let generated = completion["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            !generated.is_empty(),
            "reduced fixture must produce visible text for stop-route coverage: {completion}"
        );
        assert!(completion["usage"]["prompt_tokens"].as_u64().unwrap() > 0);

        let mut stopped_request = completion_request.clone();
        stopped_request["stop"] = Value::String(generated.clone());
        let stopped = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                Some(stopped_request),
            ))
            .await
            .unwrap();
        assert_eq!(stopped.status(), StatusCode::OK);
        let stopped = json_body(stopped).await;
        assert_eq!(stopped["choices"][0]["message"]["content"], "");
        assert_eq!(stopped["choices"][0]["finish_reason"], "stop");

        let mut streaming_request = completion_request.clone();
        streaming_request["stream"] = Value::Bool(true);
        streaming_request["stop"] = Value::String(generated.clone());
        streaming_request["stream_options"] = json!({"include_usage": true});
        let streaming = app
            .clone()
            .oneshot(request(
                Method::POST,
                "/v1/chat/completions",
                Some(streaming_request),
            ))
            .await
            .unwrap();
        assert_eq!(streaming.status(), StatusCode::OK);
        assert_eq!(streaming.headers()["content-type"], "text/event-stream");
        let stream =
            String::from_utf8(streaming.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
        assert!(stream.contains("\"role\":\"assistant\""));
        assert!(stream.contains("\"finish_reason\":\"stop\""));
        assert!(!stream.contains(&generated));
        assert!(stream.contains("\"usage\":null"));
        assert!(stream.contains("\"choices\":[],\"created\""));
        assert!(stream.contains("\"completion_tokens\":4"));
        assert!(stream.contains("data: [DONE]"));

        for invalid in [
            json!({
                "messages": [{"role": "user", "content": "a"}],
                "stop": ["one", "two", "three", "four", "five"]
            }),
            json!({
                "messages": [{"role": "user", "content": "a"}],
                "stop": ""
            }),
            json!({
                "messages": [{"role": "user", "content": "a"}],
                "temperature": 2.1
            }),
            json!({
                "messages": [{"role": "user", "content": "a"}],
                "stream_options": {"include_usage": true}
            }),
        ] {
            let response = app
                .clone()
                .oneshot(request(Method::POST, "/v1/chat/completions", Some(invalid)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        let oversized = app
            .oneshot(request(
                Method::POST,
                "/v1/tokenize",
                Some(json!({"text": "a".repeat(1_024)})),
            ))
            .await
            .unwrap();
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
