//! End-to-end tests: a real `AgentServer` over an in-memory JSON-RPC pipe,
//! driven by a scripted provider and a fake ACP client.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema as acp;
use claurst_api::{ProviderError, ProviderRequest, StreamEvent};
use claurst_core::types::{ContentBlock, UsageInfo};
use dashmap::DashMap;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::sync::{mpsc, oneshot, Notify};

use crate::connection::{self, Connection};
use crate::runtime::AgentRuntime;
use crate::server::AgentServer;

/// One scripted provider response.
pub(crate) enum Step {
    /// Never produce a response; only a cancel ends the turn.
    Hang,
    /// End the turn with this text.
    Text(&'static str),
    /// Call one tool with this input.
    Tool(&'static str, Value),
    /// Fail the request with a provider error.
    Fail(&'static str),
    /// Wait for the gate to open, then answer with the inner step.
    Gated(Arc<Notify>, Box<Step>),
}

/// Provider double that replays `Step`s and records every request.
pub(crate) struct ScriptedProvider {
    id: claurst_core::provider_id::ProviderId,
    steps: parking_lot::Mutex<VecDeque<Step>>,
    pub requests: parking_lot::Mutex<Vec<ProviderRequest>>,
    pub image_input: bool,
    started: Notify,
}

impl ScriptedProvider {
    pub fn new(steps: Vec<Step>) -> Arc<Self> {
        Self::build(steps, false)
    }

    /// A provider that reports image input support.
    pub fn with_images(steps: Vec<Step>) -> Arc<Self> {
        Self::build(steps, true)
    }

    fn build(steps: Vec<Step>, image_input: bool) -> Arc<Self> {
        Arc::new(Self {
            id: claurst_core::provider_id::ProviderId::new("scripted"),
            steps: parking_lot::Mutex::new(steps.into()),
            requests: parking_lot::Mutex::new(Vec::new()),
            image_input,
            started: Notify::new(),
        })
    }

    /// Wait until the provider has received a request.
    pub async fn wait_started(&self) {
        tokio::time::timeout(Duration::from_secs(10), self.started.notified())
            .await
            .expect("provider never received a request");
    }
}

type EventStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<StreamEvent, ProviderError>> + Send>,
>;

#[async_trait::async_trait]
impl claurst_api::LlmProvider for ScriptedProvider {
    fn id(&self) -> &claurst_core::provider_id::ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "scripted"
    }

    async fn create_message(
        &self,
        _request: ProviderRequest,
    ) -> Result<claurst_api::ProviderResponse, ProviderError> {
        unimplemented!("the query loop streams")
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<EventStream, ProviderError> {
        use claurst_api::provider_types::StopReason;

        self.requests.lock().push(request);
        self.started.notify_one();
        let mut step = self.steps.lock().pop_front().unwrap_or(Step::Text("done"));
        if let Step::Gated(gate, inner) = step {
            gate.notified().await;
            step = *inner;
        }
        let start = Ok(StreamEvent::MessageStart {
            id: uuid::Uuid::new_v4().to_string(),
            model: "scripted-model".to_string(),
            usage: UsageInfo::default(),
        });
        let events = match step {
            Step::Hang => return Ok(Box::pin(futures::stream::pending())),
            Step::Gated(..) => unreachable!("gates are opened above"),
            Step::Fail(message) => {
                return Err(ProviderError::AuthFailed {
                    provider: self.id.clone(),
                    message: message.to_string(),
                })
            }
            Step::Text(text) => vec![
                start,
                Ok(StreamEvent::TextDelta {
                    index: 0,
                    text: text.to_string(),
                }),
                Ok(StreamEvent::MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    usage: Some(text_step_usage()),
                }),
                Ok(StreamEvent::MessageStop),
            ],
            Step::Tool(name, input) => vec![
                start,
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: ContentBlock::ToolUse {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: name.to_string(),
                        input: json!({}),
                        thought_signature: None,
                    },
                }),
                Ok(StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: input.to_string(),
                }),
                Ok(StreamEvent::MessageDelta {
                    stop_reason: Some(StopReason::ToolUse),
                    usage: Some(UsageInfo::default()),
                }),
                Ok(StreamEvent::MessageStop),
            ],
        };
        Ok(Box::pin(futures::stream::iter(events)))
    }

    async fn health_check(&self) -> Result<claurst_api::ProviderStatus, ProviderError> {
        Ok(claurst_api::ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> claurst_api::ProviderCapabilities {
        claurst_api::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: self.image_input,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: claurst_api::SystemPromptStyle::TopLevel,
        }
    }
}

/// Usage every `Step::Text` request reports.
pub(crate) fn text_step_usage() -> UsageInfo {
    UsageInfo {
        input_tokens: 100,
        output_tokens: 20,
        cache_creation_input_tokens: 3,
        cache_read_input_tokens: 7,
    }
}

/// A runtime for `cwd` that routes every request to `provider`.
pub(crate) fn scripted_runtime(cwd: &Path, provider: Arc<ScriptedProvider>) -> AgentRuntime {
    let settings = claurst_core::config::Settings::default();
    let mut config = settings.effective_config();
    config.provider = Some("scripted".to_string());
    config.project_dir = Some(cwd.to_path_buf());

    let mut registry = claurst_api::ProviderRegistry::new();
    registry.register(provider);
    let provider_registry = Arc::new(registry);

    let api_client = Arc::new(
        claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        })
        .expect("build test client"),
    );

    let mut query_config = claurst_query::QueryConfig::from_config(&config);
    query_config.model = "scripted-model".to_string();
    query_config.max_turns = 10;
    query_config.provider_registry = Some(provider_registry.clone());
    let supports_images = crate::runtime::provider_accepts_images(&config, &provider_registry);

    AgentRuntime {
        config,
        settings,
        api_client,
        provider_registry,
        tools: Arc::new(claurst_tools::all_tools()),
        cost_tracker: claurst_core::CostTracker::new(),
        query_config,
        mcp_manager: None,
        working_dir: cwd.to_path_buf(),
        supports_images,
    }
}

/// Fake ACP client wired to a running `AgentServer`.
pub(crate) struct Client {
    writer: tokio::sync::Mutex<DuplexStream>,
    next_id: AtomicI64,
    responses: Arc<DashMap<i64, oneshot::Sender<Value>>>,
    /// Every `session/request_permission` the agent sent.
    pub permission_requests: Arc<parking_lot::Mutex<Vec<Value>>>,
    /// Option ids to answer permission requests with, in order. When empty
    /// the client rejects.
    pub permission_answers: Arc<parking_lot::Mutex<VecDeque<&'static str>>>,
    /// When set, permission requests are recorded but never answered, like
    /// a picker the user has not dismissed yet.
    pub hold_permissions: AtomicBool,
    /// Every `session/update` notification the agent sent.
    pub updates: Arc<parking_lot::Mutex<Vec<Value>>>,
    /// Every `_claurst/session_event` notification the agent sent.
    pub events: Arc<parking_lot::Mutex<Vec<Value>>>,
}

impl Client {
    /// Start a server with `runtime` and connect a client to it.
    pub fn start(runtime: AgentRuntime) -> Arc<Self> {
        let (client_write, server_read) = tokio::io::duplex(1 << 20);
        let (server_write, client_read) = tokio::io::duplex(1 << 20);

        let connection = Connection::new(server_write);
        let server = AgentServer::new(connection.clone());
        server
            .runtimes
            .try_lock()
            .expect("a new server's runtime cache is free")
            .insert(runtime.working_dir.clone(), Arc::new(runtime));
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(connection::run_reader(connection, server_read, tx));
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                server.dispatch(msg);
            }
        });

        let client = Arc::new(Self {
            writer: tokio::sync::Mutex::new(client_write),
            next_id: AtomicI64::new(1),
            responses: Arc::new(DashMap::new()),
            permission_requests: Arc::new(parking_lot::Mutex::new(Vec::new())),
            permission_answers: Arc::new(parking_lot::Mutex::new(VecDeque::new())),
            hold_permissions: AtomicBool::new(false),
            updates: Arc::new(parking_lot::Mutex::new(Vec::new())),
            events: Arc::new(parking_lot::Mutex::new(Vec::new())),
        });
        tokio::spawn(Self::read_loop(client.clone(), client_read));
        client
    }

    async fn read_loop(self: Arc<Self>, reader: DuplexStream) {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let msg: Value = serde_json::from_str(&line).expect("agent wrote invalid JSON");
            match msg.get("method").and_then(Value::as_str) {
                Some("session/request_permission") => {
                    self.permission_requests.lock().push(msg["params"].clone());
                    if self.hold_permissions.load(Ordering::SeqCst) {
                        continue;
                    }
                    let answer = self.permission_answers.lock().pop_front().unwrap_or("reject_once");
                    let reply = json!({
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": { "outcome": { "outcome": "selected", "optionId": answer } },
                    });
                    self.write(&reply).await;
                }
                Some("session/update") => self.updates.lock().push(msg["params"].clone()),
                Some(crate::extensions::SESSION_EVENT) => self.events.lock().push(msg["params"].clone()),
                Some(other) => panic!("unexpected agent message: {other}"),
                None => {
                    let id = msg["id"].as_i64().expect("response id");
                    if let Some((_, tx)) = self.responses.remove(&id) {
                        let _ = tx.send(msg);
                    }
                }
            }
        }
    }

    async fn write(&self, msg: &Value) {
        let mut bytes = serde_json::to_vec(msg).expect("serialize");
        bytes.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(&bytes).await.expect("write to agent");
        writer.flush().await.expect("flush to agent");
    }

    /// Send a request and return the full response envelope.
    pub async fn call(&self, method: &str, params: impl serde::Serialize) -> Value {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.responses.insert(id, tx);
        let params = serde_json::to_value(params).expect("serialize params");
        self.write(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .await;
        tokio::time::timeout(Duration::from_secs(20), rx)
            .await
            .unwrap_or_else(|_| panic!("no response to {method}"))
            .expect("response channel closed")
    }

    pub async fn notify(&self, method: &str, params: impl serde::Serialize) {
        let params = serde_json::to_value(params).expect("serialize params");
        self.write(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await;
    }

    /// `initialize`, opting into the Claurst extensions.
    pub async fn initialize_with_extensions(&self) -> Value {
        self.call(
            "initialize",
            json!({
                "protocolVersion": 1,
                "clientCapabilities": {
                    "_meta": { "claurst": { "sessionEvents": true, "usageUpdates": true } }
                },
            }),
        )
        .await
    }

    pub async fn new_session(&self, cwd: &Path) -> acp::SessionId {
        let response = self
            .call("session/new", acp::NewSessionRequest::new(cwd.to_path_buf()))
            .await;
        let result: acp::NewSessionResponse =
            serde_json::from_value(response["result"].clone()).expect("session/new result");
        result.session_id
    }

    pub async fn prompt(&self, session_id: &acp::SessionId, text: &str) -> Value {
        self.call(
            "session/prompt",
            acp::PromptRequest::new(
                session_id.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(text))],
            ),
        )
        .await
    }
}

/// Parse the stop reason out of a `session/prompt` response.
pub(crate) fn stop_reason(response: &Value) -> acp::StopReason {
    let result: acp::PromptResponse = serde_json::from_value(response["result"].clone())
        .unwrap_or_else(|e| panic!("expected a PromptResponse, got {response}: {e}"));
    result.stop_reason
}

/// Serialises the tests that read or write the global settings, and points
/// the settings directory at a fresh scratch directory for each of them so
/// an "Allow always" rule saved by one test never leaks into another (or
/// into the real user settings).
pub(crate) async fn isolated_home() -> tokio::sync::MutexGuard<'static, ()> {
    static HOME_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let guard = HOME_LOCK.lock().await;
    std::env::set_var("CLAURST_HOME", scratch_dir("home"));
    guard
}

/// Tool-call updates the agent sent, as `(status, tool title)` pairs.
pub(crate) fn tool_statuses(client: &Client) -> Vec<String> {
    client
        .updates
        .lock()
        .iter()
        .filter(|u| u["update"]["sessionUpdate"] == "tool_call_update")
        .map(|u| u["update"]["status"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// A fresh, canonical scratch directory.
pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("claurst-acp-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    std::fs::canonicalize(&dir).expect("canonicalize scratch dir")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_after_cancel_runs_normally() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("cancel");
    let provider = ScriptedProvider::new(vec![Step::Hang, Step::Text("second answer")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let session_id = client.new_session(&cwd).await;

    let first = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "first").await })
    };
    provider.wait_started().await;
    client
        .notify("session/cancel", acp::CancelNotification::new(session_id.clone()))
        .await;
    let first = first.await.expect("first prompt task");
    assert_eq!(stop_reason(&first), acp::StopReason::Cancelled);

    let second = client.prompt(&session_id, "second").await;
    assert_eq!(
        stop_reason(&second),
        acp::StopReason::EndTurn,
        "a cancel must not poison later prompts"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_prompt_on_one_session_is_rejected() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("busy");
    let provider = ScriptedProvider::new(vec![Step::Hang]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let session_id = client.new_session(&cwd).await;

    let first = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "first").await })
    };
    provider.wait_started().await;

    let second = client.prompt(&session_id, "second").await;
    let error: acp::Error =
        serde_json::from_value(second["error"].clone()).expect("busy prompt must be an error");
    assert_eq!(error.code, acp::ErrorCode::InvalidRequest);
    assert!(error.message.contains("already has a prompt running"), "{}", error.message);

    client
        .notify("session/cancel", acp::CancelNotification::new(session_id.clone()))
        .await;
    let first = first.await.expect("first prompt task");
    assert_eq!(stop_reason(&first), acp::StopReason::Cancelled);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn read_inside_workspace_runs_without_asking_the_client() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("read");
    let file = cwd.join("notes.txt");
    std::fs::write(&file, "hello").expect("seed file");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Read", json!({ "file_path": file })),
        Step::Text("read it"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    let session_id = client.new_session(&cwd).await;

    let response = client.prompt(&session_id, "read the notes").await;
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    assert!(
        client.permission_requests.lock().is_empty(),
        "a read inside the workspace is allowed by policy and must not reach the client"
    );
    assert_eq!(tool_statuses(&client), vec!["completed"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn read_outside_workspace_asks_the_client() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("workspace");
    let outside = scratch_dir("outside").join("secret.txt");
    std::fs::write(&outside, "secret").expect("seed file");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Read", json!({ "file_path": outside })),
        Step::Text("read it"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    client.permission_answers.lock().push_back("allow_once");
    let session_id = client.new_session(&cwd).await;

    let response = client.prompt(&session_id, "read the secret").await;
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    let requests = client.permission_requests.lock().clone();
    assert_eq!(requests.len(), 1, "{requests:?}");
    let request: acp::RequestPermissionRequest =
        serde_json::from_value(requests[0].clone()).expect("request_permission params");
    assert_eq!(request.session_id, session_id);
    assert_eq!(tool_statuses(&client), vec!["completed"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn write_asks_once_then_allow_always_is_remembered_and_saved() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("write");
    let target = cwd.join("out.txt");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Write", json!({ "file_path": target, "content": "one" })),
        Step::Text("wrote one"),
        Step::Tool("Write", json!({ "file_path": target, "content": "two" })),
        Step::Text("wrote two"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    client.permission_answers.lock().push_back("allow_always");
    let session_id = client.new_session(&cwd).await;

    let first = client.prompt(&session_id, "write one").await;
    assert_eq!(stop_reason(&first), acp::StopReason::EndTurn);
    assert_eq!(client.permission_requests.lock().len(), 1);
    assert_eq!(std::fs::read_to_string(&target).expect("written"), "one");

    let second = client.prompt(&session_id, "write two").await;
    assert_eq!(stop_reason(&second), acp::StopReason::EndTurn);
    assert_eq!(
        client.permission_requests.lock().len(),
        1,
        "allow-always must not ask again in the same session"
    );
    assert_eq!(std::fs::read_to_string(&target).expect("written"), "two");

    // Saved like the TUI's "always allow in this project": in the project
    // store under the claurst home, not in the global settings.
    let store = claurst_core::permissions::project_approvals_path(
        &claurst_core::config::Settings::config_dir(),
        &claurst_core::permissions::approvals_project_root(&cwd),
    );
    let saved = std::fs::read_to_string(&store).expect("project approvals saved");
    assert!(saved.contains("\"Write\""), "allow-always must be saved for the project: {saved}");
    let global = claurst_core::config::Settings::load_sync().expect("settings readable");
    assert!(global.permission_rules.is_empty(), "{:?}", global.permission_rules);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejected_write_fails_the_tool_and_asks_again_next_time() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("reject");
    let target = cwd.join("out.txt");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Write", json!({ "file_path": target, "content": "one" })),
        Step::Text("could not write"),
        Step::Tool("Write", json!({ "file_path": target, "content": "two" })),
        Step::Text("could not write"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    client
        .permission_answers
        .lock()
        .extend(["reject_once", "reject_once"]);
    let session_id = client.new_session(&cwd).await;

    client.prompt(&session_id, "write one").await;
    client.prompt(&session_id, "write two").await;
    assert_eq!(client.permission_requests.lock().len(), 2);
    assert!(!target.exists(), "a rejected write must not touch the file");
    assert_eq!(tool_statuses(&client), vec!["failed", "failed"]);
}

/// A minimal stdio MCP server: answers `initialize`, lists one `ping` tool,
/// and returns "pong" for every call.
#[cfg(unix)]
const FAKE_MCP_SERVER: &str = r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | grep -o '"id":[0-9]*' | head -n 1 | cut -d: -f2)
  [ -z "$id" ] && continue
  case "$line" in
    *'"method":"initialize"'*)
      version=$(printf '%s' "$line" | grep -o '"protocolVersion":"[^"]*"' | head -n 1 | cut -d'"' -f4)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1.0.0"}}}\n' "$id" "$version" ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"ping","description":"Reply with pong","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id" ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"pong"}],"isError":false}}\n' "$id" ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"method not found"}}\n' "$id" ;;
  esac
done
"#;

#[cfg(unix)]
fn sh_server(
    name: &str,
    script: &str,
    origin: claurst_core::config::McpServerOrigin,
) -> claurst_core::config::McpServerConfig {
    claurst_core::config::McpServerConfig {
        name: name.to_string(),
        command: Some("sh".to_string()),
        args: vec!["-c".to_string(), script.to_string()],
        env: std::collections::HashMap::new(),
        url: None,
        server_type: "stdio".to_string(),
        headers: Default::default(),
        connect_timeout_ms: None,
        catalog_timeout_ms: None,
        origin,
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn trusted_mcp_tools_are_attached_and_callable() {
    use claurst_core::config::{McpServerOrigin, Settings};

    let _home = isolated_home().await;
    let cwd = scratch_dir("mcp");
    let marker = cwd.join("project-server-ran");
    let settings = Settings::default();
    let mut config = settings.effective_config();
    config.mcp_servers = vec![
        sh_server("fake", FAKE_MCP_SERVER, McpServerOrigin::User),
        sh_server(
            "repo",
            &format!("touch '{}'", marker.display()),
            McpServerOrigin::Project,
        ),
    ];

    let manager = crate::runtime::build_mcp_manager(&config, &settings, &cwd)
        .await
        .expect("the user-defined server is trusted");
    let names: Vec<String> = claurst_tools::mcp_tools(&manager)
        .iter()
        .map(|tool| tool.name().to_string())
        .collect();
    assert_eq!(names, vec!["mcp__fake__ping"]);
    assert!(
        !marker.exists(),
        "an untrusted project server must not be launched"
    );

    let provider = ScriptedProvider::new(vec![
        Step::Tool("mcp__fake__ping", json!({})),
        Step::Text("got pong"),
    ]);
    let mut runtime = scripted_runtime(&cwd, provider);
    let mut tools = claurst_tools::all_tools();
    tools.extend(claurst_tools::mcp_tools(&manager));
    runtime.tools = Arc::new(tools);
    runtime.mcp_manager = Some(manager);

    let client = Client::start(runtime);
    client.permission_answers.lock().push_back("allow_once");
    let session_id = client.new_session(&cwd).await;
    let response = client.prompt(&session_id, "ping the server").await;
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    assert_eq!(
        client.permission_requests.lock().len(),
        1,
        "an MCP tool asks once (no double prompt from the central backstop)"
    );
    let outputs: Vec<Value> = client
        .updates
        .lock()
        .iter()
        .filter(|u| u["update"]["sessionUpdate"] == "tool_call_update")
        .map(|u| u["update"]["rawOutput"].clone())
        .collect();
    assert_eq!(outputs, vec![json!("pong")]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_turn_is_a_jsonrpc_error_not_a_refusal() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("error");
    let provider = ScriptedProvider::new(vec![Step::Fail("invalid api key")]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    let session_id = client.new_session(&cwd).await;

    let response = client.prompt(&session_id, "hello").await;
    assert!(response.get("result").is_none(), "{response}");
    let error: acp::Error =
        serde_json::from_value(response["error"].clone()).expect("an ACP error object");
    assert_eq!(error.code, acp::ErrorCode::InternalError);
    assert!(error.message.contains("invalid api key"), "{}", error.message);

    let retry = client.prompt(&session_id, "hello again").await;
    assert_eq!(
        stop_reason(&retry),
        acp::StopReason::EndTurn,
        "a failed turn must leave the session usable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_images_reach_a_provider_that_accepts_them() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("image");
    let provider = ScriptedProvider::with_images(vec![Step::Text("a cat")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let session_id = client.new_session(&cwd).await;

    let response = client
        .call(
            "session/prompt",
            acp::PromptRequest::new(
                session_id.clone(),
                vec![
                    acp::ContentBlock::Text(acp::TextContent::new("what is this?")),
                    acp::ContentBlock::Image(acp::ImageContent::new("aGVsbG8=", "image/png")),
                ],
            ),
        )
        .await;
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);

    let requests = provider.requests.lock();
    let user = requests[0]
        .messages
        .iter()
        .find(|m| m.role == claurst_core::types::Role::User)
        .expect("the request carries the user turn");
    let has_image = user.content_blocks().iter().any(|block| {
        matches!(block, ContentBlock::Image { source } if source.data.as_deref() == Some("aGVsbG8="))
    });
    assert!(has_image, "the image must be passed to the provider: {user:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runtime_uses_settings_and_agents_md_of_the_session_cwd() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("project");
    std::fs::create_dir_all(cwd.join(".claurst")).expect("create .claurst");
    let mut project = claurst_core::config::Settings::default();
    project.config.model = Some("project-pinned-model".to_string());
    std::fs::write(
        cwd.join(".claurst").join("settings.json"),
        serde_json::to_string(&project).expect("serialize project settings"),
    )
    .expect("write project settings");
    std::fs::write(cwd.join("AGENTS.md"), "Project rule: PROJECT-RULE-MARKER").expect("write AGENTS.md");

    let runtime = AgentRuntime::build(cwd.clone()).await.expect("runtime builds");
    assert_eq!(runtime.working_dir, cwd);
    assert_eq!(runtime.query_config.model, "project-pinned-model");
    assert!(
        runtime
            .query_config
            .instruction_files
            .iter()
            .any(|file| file.path == cwd.join("AGENTS.md") && file.content.contains("PROJECT-RULE-MARKER")),
        "AGENTS.md from the session cwd must reach the system prompt"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_while_a_permission_request_is_pending_unwinds_the_tool() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("cancel-pending");
    let target = cwd.join("out.txt");
    let provider = ScriptedProvider::new(vec![Step::Tool(
        "Write",
        json!({ "file_path": target, "content": "never" }),
    )]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    client.hold_permissions.store(true, Ordering::SeqCst);
    let session_id = client.new_session(&cwd).await;

    let prompt = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "write it").await })
    };
    tokio::time::timeout(Duration::from_secs(10), async {
        while client.permission_requests.lock().is_empty() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the write must reach the client as a permission request");

    client
        .notify("session/cancel", acp::CancelNotification::new(session_id.clone()))
        .await;
    // `Client::call` panics after 20 s without a response, so a tool stuck
    // waiting on the unanswered request fails the test.
    let response = prompt.await.expect("prompt task");
    assert_eq!(stop_reason(&response), acp::StopReason::Cancelled);
    assert!(!target.exists(), "a cancelled write must not touch the file");
    assert_eq!(tool_statuses(&client), vec!["failed"]);
    assert_eq!(client.permission_requests.lock().len(), 1);
}

/// `sessionUpdate` kinds the agent sent, in order.
fn update_kinds(client: &Client) -> Vec<String> {
    client
        .updates
        .lock()
        .iter()
        .map(|u| u["update"]["sessionUpdate"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_response_reports_the_turn_usage() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("usage");
    let provider = ScriptedProvider::new(vec![Step::Text("hi")]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    let session_id = client.new_session(&cwd).await;

    let response = client.prompt(&session_id, "hello").await;
    let result = &response["result"];
    assert_eq!(result["usage"]["inputTokens"], 100, "{result}");
    assert_eq!(result["usage"]["outputTokens"], 20);
    assert_eq!(result["usage"]["cachedReadTokens"], 7);
    assert_eq!(result["usage"]["cachedWriteTokens"], 3);
    assert_eq!(result["usage"]["totalTokens"], 130);
    assert_eq!(result["_meta"]["claurst"]["contextUsed"], 130);
    assert!(result["_meta"]["claurst"]["contextSize"].as_u64().unwrap_or(0) > 0, "{result}");

    // A client that did not opt in gets the plain ACP stream.
    assert!(!update_kinds(&client).contains(&"usage_update".to_string()));
    assert!(client.events.lock().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opted_in_client_gets_usage_updates_and_the_turn_cap_event() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("events");
    let file = cwd.join("notes.txt");
    std::fs::write(&file, "hello").expect("seed file");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Read", json!({ "file_path": file })),
        Step::Tool("Read", json!({ "file_path": file })),
        Step::Text("summary"),
    ]);
    let mut runtime = scripted_runtime(&cwd, provider);
    runtime.query_config.max_turns = 1;
    let client = Client::start(runtime);
    client.initialize_with_extensions().await;
    let session_id = client.new_session(&cwd).await;

    client.prompt(&session_id, "read the notes").await;

    let events = client.events.lock().clone();
    assert!(
        events.iter().any(|e| e["sessionId"] == session_id.0.as_ref()
            && e["event"]["type"] == "max_turns"
            && e["event"]["maxTurns"] == 1),
        "{events:?}"
    );
    let usage: Vec<Value> = client
        .updates
        .lock()
        .iter()
        .filter(|u| u["update"]["sessionUpdate"] == "usage_update")
        .cloned()
        .collect();
    assert!(!usage.is_empty(), "no usage_update in {:?}", update_kinds(&client));
    assert!(usage[0]["update"]["size"].as_u64().unwrap_or(0) > 0);
    assert_eq!(usage[0]["update"]["cost"]["currency"], "USD");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_finished_turn_is_saved_to_the_sessions_directory() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("save");
    let provider = ScriptedProvider::new(vec![Step::Text("first answer")]);
    let client = Client::start(scripted_runtime(&cwd, provider));
    let session_id = client.new_session(&cwd).await;

    client.prompt(&session_id, "hello there").await;

    let saved = claurst_core::session_journal::load(
        &claurst_core::history::sessions_dir(),
        session_id.0.as_ref(),
    )
    .expect("the session is saved after its first turn")
    .session;
    assert_eq!(saved.messages.len(), 2, "{:?}", saved.messages);
    assert_eq!(saved.messages[1].get_all_text(), "first answer");
    assert_eq!(saved.title.as_deref(), Some("hello there"));
    assert_eq!(saved.working_dir.as_deref(), Some(cwd.display().to_string().as_str()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_saved_session_is_loaded_replayed_and_continued() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("load");
    // A session saved by an earlier process.
    let mut saved = claurst_core::history::ConversationSession::new("scripted-model".into());
    saved.id = "acp-saved-earlier".into();
    saved.working_dir = Some(cwd.display().to_string());
    saved.messages = vec![
        claurst_core::types::Message::user("hello"),
        claurst_core::types::Message::assistant("first answer"),
    ];
    claurst_core::session_journal::save(&claurst_core::history::sessions_dir(), &mut saved)
        .expect("seed saved session");

    let provider = ScriptedProvider::new(vec![Step::Text("second answer")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let init = client.initialize_with_extensions().await;
    assert_eq!(init["result"]["agentCapabilities"]["loadSession"], true, "{init}");

    let load = client
        .call("session/load", acp::LoadSessionRequest::new("acp-saved-earlier", cwd.clone()))
        .await;
    assert!(load.get("error").is_none(), "{load}");
    let replayed: Vec<(String, String)> = client
        .updates
        .lock()
        .iter()
        .map(|u| {
            (
                u["update"]["sessionUpdate"].as_str().unwrap_or_default().to_string(),
                u["update"]["content"]["text"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    assert_eq!(
        replayed,
        vec![
            ("user_message_chunk".to_string(), "hello".to_string()),
            ("agent_message_chunk".to_string(), "first answer".to_string()),
        ]
    );

    let session_id = acp::SessionId::new("acp-saved-earlier");
    let response = client.prompt(&session_id, "and then?").await;
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    let sent = provider.requests.lock()[0].messages.clone();
    assert_eq!(sent.len(), 3, "the model sees the loaded history: {sent:?}");
    assert_eq!(sent[1].get_all_text(), "first answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn load_refuses_unknown_and_unsafe_session_ids() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("load-bad");
    let client = Client::start(scripted_runtime(&cwd, ScriptedProvider::new(Vec::new())));

    let unknown = client
        .call("session/load", acp::LoadSessionRequest::new("acp-never-saved", cwd.clone()))
        .await;
    let error: acp::Error = serde_json::from_value(unknown["error"].clone()).expect("unknown id is an error");
    assert_eq!(error.code, acp::ErrorCode::ResourceNotFound);

    let unsafe_id = client
        .call("session/load", acp::LoadSessionRequest::new("../../settings", cwd.clone()))
        .await;
    let error: acp::Error = serde_json::from_value(unsafe_id["error"].clone()).expect("unsafe id is an error");
    assert_eq!(error.code, acp::ErrorCode::InvalidParams);
}

/// Texts of the user messages in `request`.
fn user_texts(request: &ProviderRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|m| m.role == claurst_core::types::Role::User)
        .map(|m| m.get_all_text())
        .filter(|t| !t.is_empty())
        .collect()
}

async fn queue(client: &Client, session_id: &acp::SessionId, text: &str) -> Value {
    let response = client
        .call(
            crate::extensions::QUEUE_MESSAGE,
            json!({ "sessionId": session_id, "text": text }),
        )
        .await;
    response["result"].clone()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_queued_during_a_turn_reaches_the_model_at_the_next_step() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("queue-step");
    std::fs::write(cwd.join("a.txt"), "hello").unwrap();
    let gate = Arc::new(Notify::new());
    let provider = ScriptedProvider::new(vec![
        Step::Gated(
            gate.clone(),
            Box::new(Step::Tool("Read", json!({ "file_path": cwd.join("a.txt") }))),
        ),
        Step::Text("read it, and noted the icon"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let init = client.initialize_with_extensions().await;
    assert_eq!(init["result"]["agentCapabilities"]["_meta"]["claurst"]["queueMessages"], true);
    let session_id = client.new_session(&cwd).await;

    let turn = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "read a.txt").await })
    };
    provider.wait_started().await;
    let queued = queue(&client, &session_id, "also add an icon").await;
    assert_eq!(queued["queued"], true, "{queued}");
    gate.notify_one();

    let response = turn.await.expect("prompt task");
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    assert_eq!(response["result"]["_meta"]["claurst"]["undeliveredMessages"], json!([]));
    let requests = provider.requests.lock();
    assert_eq!(requests.len(), 2);
    assert_eq!(user_texts(&requests[1]).last().map(String::as_str), Some("also add an icon"));
    let events = client.events.lock();
    assert!(
        events.iter().any(|e| e["event"]
            == json!({ "type": "input_injected", "messages": ["also add an icon"] })),
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_queued_during_the_final_answer_keeps_the_turn_going() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("queue-final");
    let gate = Arc::new(Notify::new());
    let provider = ScriptedProvider::new(vec![
        Step::Gated(gate.clone(), Box::new(Step::Text("first answer"))),
        Step::Text("answer to the follow-up"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    client.initialize_with_extensions().await;
    let session_id = client.new_session(&cwd).await;

    let turn = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "question").await })
    };
    provider.wait_started().await;
    assert_eq!(queue(&client, &session_id, "follow-up").await["queued"], true);
    gate.notify_one();

    let response = turn.await.expect("prompt task");
    assert_eq!(stop_reason(&response), acp::StopReason::EndTurn);
    let requests = provider.requests.lock();
    assert_eq!(requests.len(), 2, "the queued message is answered in the same turn");
    assert_eq!(user_texts(&requests[1]), vec!["question", "follow-up"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_messages_can_be_taken_back_and_a_cancel_returns_the_rest() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("queue-cancel");
    let provider = ScriptedProvider::new(vec![Step::Hang]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    client.initialize_with_extensions().await;
    let session_id = client.new_session(&cwd).await;

    // Nothing running: the client must send it as a prompt instead.
    assert_eq!(queue(&client, &session_id, "too early").await, json!({ "queued": false }));

    let turn = {
        let client = client.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move { client.prompt(&session_id, "work").await })
    };
    provider.wait_started().await;
    let first = queue(&client, &session_id, "first").await;
    queue(&client, &session_id, "second").await;
    let removed = client
        .call(
            crate::extensions::UNQUEUE_MESSAGE,
            json!({ "sessionId": session_id, "id": first["id"] }),
        )
        .await;
    assert_eq!(removed["result"], json!({ "removed": true }));

    client
        .notify("session/cancel", acp::CancelNotification::new(session_id.clone()))
        .await;
    let response = turn.await.expect("prompt task");
    assert_eq!(stop_reason(&response), acp::StopReason::Cancelled);
    assert_eq!(response["result"]["_meta"]["claurst"]["undeliveredMessages"], json!(["second"]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_instructions_reach_the_system_prompt() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("session-prompt");
    let provider = ScriptedProvider::new(vec![Step::Text("checked")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    client.initialize_with_extensions().await;
    let mut request = acp::NewSessionRequest::new(cwd.clone());
    request.meta = json!({ "claurst": { "appendSystemPrompt": "Verify your work before you stop." } })
        .as_object()
        .cloned();
    let response = client.call("session/new", request).await;
    let session_id: acp::SessionId =
        serde_json::from_value(response["result"]["sessionId"].clone()).expect("session id");

    client.prompt(&session_id, "fix the button").await;
    let system = serde_json::to_string(&provider.requests.lock()[0].system_prompt).unwrap();
    assert!(system.contains("Verify your work before you stop."), "{system}");
}

async fn new_session_with(client: &Client, cwd: &Path, meta: Value) -> Value {
    let mut request = acp::NewSessionRequest::new(cwd.to_path_buf());
    request.meta = meta.as_object().cloned();
    client.call("session/new", request).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_options_deny_tools_and_size_responses() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("session-options");
    let file = cwd.join("notes.txt");
    std::fs::write(&file, "hello").expect("seed file");
    let provider = ScriptedProvider::new(vec![
        Step::Tool("Read", json!({ "file_path": file })),
        Step::Text("could not read"),
    ]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let response = new_session_with(
        &client,
        &cwd,
        json!({ "claurst": { "options": { "disallowedTools": "Read", "maxTokens": 1234 } } }),
    )
    .await;
    let session_id: acp::SessionId =
        serde_json::from_value(response["result"]["sessionId"].clone()).expect("session id");

    client.prompt(&session_id, "read the notes").await;
    assert_eq!(tool_statuses(&client), vec!["failed"], "Read is denied for this session");
    assert!(client.permission_requests.lock().is_empty(), "a denied tool is not asked for");
    assert_eq!(provider.requests.lock()[0].max_tokens, 1234);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_session_options_are_refused() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("bad-options");
    let client = Client::start(scripted_runtime(&cwd, ScriptedProvider::new(vec![])));
    let response = new_session_with(
        &client,
        &cwd,
        json!({ "claurst": { "options": { "allowedTools": "Bash(unclosed" } } }),
    )
    .await;
    let error: acp::Error = serde_json::from_value(response["error"].clone()).expect("an error");
    assert_eq!(error.code, acp::ErrorCode::InvalidParams);
    assert!(error.data.unwrap().to_string().contains("allowedTools"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_sessions_use_their_own_model() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("session-models");
    let provider = ScriptedProvider::new(vec![Step::Text("from a"), Step::Text("from b")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let init = client.initialize_with_extensions().await;
    assert_eq!(init["result"]["agentCapabilities"]["_meta"]["claurst"]["sessionModel"], true);

    let mut sessions = Vec::new();
    for model in ["scripted/model-a", "scripted/model-b"] {
        let response = new_session_with(
            &client,
            &cwd,
            json!({ "claurst": { "options": { "provider": "scripted", "model": model, "effort": "low" } } }),
        )
        .await;
        sessions.push(
            serde_json::from_value::<acp::SessionId>(response["result"]["sessionId"].clone()).expect("session id"),
        );
    }
    client.prompt(&sessions[0], "hello a").await;
    client.prompt(&sessions[1], "hello b").await;
    let models: Vec<String> = provider.requests.lock().iter().map(|r| r.model.clone()).collect();
    assert_eq!(models, ["model-a", "model-b"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_plan_session_only_offers_read_tools() {
    let _home = isolated_home().await;
    let cwd = scratch_dir("plan-agent");
    let provider = ScriptedProvider::new(vec![Step::Text("here is the plan")]);
    let client = Client::start(scripted_runtime(&cwd, provider.clone()));
    let init = client.initialize_with_extensions().await;
    assert_eq!(init["result"]["agentCapabilities"]["_meta"]["claurst"]["sessionAgents"], true);
    let response =
        new_session_with(&client, &cwd, json!({ "claurst": { "options": { "agent": "plan" } } })).await;
    let session_id: acp::SessionId =
        serde_json::from_value(response["result"]["sessionId"].clone()).expect("session id");

    client.prompt(&session_id, "plan the refactor").await;
    let requests = provider.requests.lock();
    let tools: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tools.contains(&"Read"), "{tools:?}");
    for forbidden in ["Write", "Edit", "Bash", "ExitPlanMode"] {
        assert!(!tools.contains(&forbidden), "{forbidden} offered in plan mode: {tools:?}");
    }
    let system = serde_json::to_string(&requests[0].system_prompt).unwrap();
    assert!(system.contains("You are the plan agent"), "the agent's prompt is used");

    let unknown =
        new_session_with(&client, &cwd, json!({ "claurst": { "options": { "agent": "wizard" } } })).await;
    assert_eq!(unknown["error"]["code"], -32602);
}
