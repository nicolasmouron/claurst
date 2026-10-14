//! Per-session state for the ACP server.

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema as acp;
use claurst_core::history::{CompactionRecord, ConversationSession};
use claurst_core::permissions::PermissionManager;
use claurst_core::session_journal::SessionJournal;
use claurst_core::types::{Message, Role};
use claurst_query::{CommandPriority, CommandQueue, QueuedCommand};
use claurst_tools::PendingPermissionStore;
use dashmap::DashMap;
use tokio_util::sync::CancellationToken;

use crate::runtime::AgentRuntime;

/// One ACP session — a logical conversation with its own cwd, transcript,
/// MCP server roster, and the cancellation token of its running prompt.
pub struct SessionState {
    pub session_id: acp::SessionId,
    pub cwd: PathBuf,
    /// Runtime built for `cwd`: settings, context, tools and MCP servers.
    pub runtime: Arc<AgentRuntime>,
    pub messages: parking_lot::Mutex<Vec<Message>>,
    /// The prompt currently running on this session, `None` while the
    /// session is idle. Each prompt gets a fresh cancel token and input queue,
    /// so a `session/cancel` or a queued message only ever affects the turn
    /// that was in flight.
    active_turn: parking_lot::Mutex<Option<TurnSlot>>,
    pub pending_permissions: Arc<parking_lot::Mutex<PendingPermissionStore>>,
    /// Decides every tool check before anything is sent to the client, and
    /// holds this session's "Allow always" rules.
    pub permission_manager: Arc<std::sync::Mutex<PermissionManager>>,
    pub file_history: Arc<parking_lot::Mutex<claurst_core::file_history::FileHistory>>,
    pub current_turn: Arc<std::sync::atomic::AtomicUsize>,
    /// What is saved in `~/.claurst/sessions/<id>.json`, so a later process
    /// can `session/load` this session. Its `messages` are refreshed at the
    /// end of every turn.
    pub record: parking_lot::Mutex<ConversationSession>,
    /// Crash-safe writer for `record`: the query loop journals every step.
    pub journal: Arc<SessionJournal>,
}

impl SessionState {
    pub fn new(
        session_id: acp::SessionId,
        runtime: Arc<AgentRuntime>,
        permission_manager: PermissionManager,
    ) -> Arc<Self> {
        let mut record = ConversationSession::new(runtime.query_config.model.clone());
        record.id = session_id.0.to_string();
        record.working_dir = Some(runtime.working_dir.display().to_string());
        Self::from_record(session_id, runtime, permission_manager, record)
    }

    /// A session restored from its saved record (`session/load`).
    pub fn from_record(
        session_id: acp::SessionId,
        runtime: Arc<AgentRuntime>,
        permission_manager: PermissionManager,
        record: ConversationSession,
    ) -> Arc<Self> {
        let journal = SessionJournal::for_session(&record);
        Arc::new(Self {
            session_id,
            cwd: runtime.working_dir.clone(),
            runtime,
            messages: parking_lot::Mutex::new(record.messages.clone()),
            active_turn: parking_lot::Mutex::new(None),
            pending_permissions: Arc::new(parking_lot::Mutex::new(
                PendingPermissionStore::default(),
            )),
            permission_manager: Arc::new(std::sync::Mutex::new(permission_manager)),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            record: parking_lot::Mutex::new(record),
            journal,
        })
    }

    /// Whether a prompt is running on this session.
    pub fn is_busy(&self) -> bool {
        self.active_turn.lock().is_some()
    }

    /// Record a finished turn and checkpoint the session to disk. The query
    /// loop already journaled each step; this writes the snapshot.
    pub fn checkpoint(
        &self,
        messages: &[Message],
        compactions: Vec<CompactionRecord>,
        turn_cost_usd: f64,
        turn_tokens: u64,
    ) -> anyhow::Result<()> {
        let mut record = self.record.lock();
        record.messages = messages.to_vec();
        record.compactions.extend(compactions);
        record.updated_at = chrono::Utc::now();
        record.total_cost += turn_cost_usd;
        record.total_tokens += turn_tokens;
        if record.title.is_none() {
            record.title = first_user_line(messages);
        }
        claurst_core::session_journal::save(&claurst_core::history::sessions_dir(), &mut record)
    }

    /// Cost of the session so far, in USD.
    pub fn total_cost(&self) -> f64 {
        self.record.lock().total_cost
    }

    /// Start a prompt turn with a fresh cancellation token. Returns `None`
    /// when another prompt is already running on this session.
    pub fn begin_turn(&self) -> Option<ActiveTurn<'_>> {
        let mut slot = self.active_turn.lock();
        if slot.is_some() {
            return None;
        }
        let token = CancellationToken::new();
        let queue = CommandQueue::new();
        *slot = Some(TurnSlot { token: token.clone(), queue: queue.clone() });
        Some(ActiveTurn {
            slot: &self.active_turn,
            token,
            queue,
        })
    }

    /// Queue `text` for the running prompt: the query loop adds it to the
    /// conversation before its next model request. Returns the id that
    /// [`SessionState::unqueue_message`] accepts, or `None` when no prompt is
    /// running (the client then sends it as a new prompt).
    pub fn queue_message(&self, text: String) -> Option<u64> {
        let slot = self.active_turn.lock();
        let turn = slot.as_ref()?;
        Some(turn.queue.push(QueuedCommand::InjectUserMessage(text), CommandPriority::Normal))
    }

    /// Take back a queued message the query loop has not delivered yet.
    pub fn unqueue_message(&self, id: u64) -> bool {
        let slot = self.active_turn.lock();
        slot.as_ref().is_some_and(|turn| turn.queue.remove(id).is_some())
    }

    /// Cancel the running prompt, if any. Returns whether a turn was running.
    pub fn cancel_turn(&self) -> bool {
        match self.active_turn.lock().as_ref() {
            Some(turn) => {
                turn.token.cancel();
                true
            }
            None => false,
        }
    }
}

struct TurnSlot {
    token: CancellationToken,
    queue: CommandQueue,
}

/// A running prompt turn. Dropping it marks the session idle again.
pub struct ActiveTurn<'a> {
    slot: &'a parking_lot::Mutex<Option<TurnSlot>>,
    pub token: CancellationToken,
    /// Messages the client queued while this turn runs.
    pub queue: CommandQueue,
}

impl ActiveTurn<'_> {
    /// Mark the session idle and return the queued messages the query loop
    /// never delivered, in order. Done under the slot lock, so a message is
    /// either returned here or refused by [`SessionState::queue_message`].
    pub fn finish(self) -> Vec<String> {
        *self.slot.lock() = None;
        let mut left = Vec::new();
        for command in self.queue.drain() {
            if let QueuedCommand::InjectUserMessage(text) = command {
                left.push(text);
            }
        }
        left
    }
}

impl Drop for ActiveTurn<'_> {
    fn drop(&mut self) {
        *self.slot.lock() = None;
    }
}

/// First line of the first user message, as the saved session's title.
fn first_user_line(messages: &[Message]) -> Option<String> {
    let text = messages.iter().find(|m| m.role == Role::User)?.get_all_text();
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(line.chars().take(80).collect())
}

/// Whether `id` can name a saved session. Ids become file names in the
/// sessions directory, so anything that could leave it is refused.
pub fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Map of active sessions keyed by ACP session id.
#[derive(Default)]
pub struct SessionRegistry {
    inner: DashMap<acp::SessionId, Arc<SessionState>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, state: Arc<SessionState>) {
        self.inner.insert(state.session_id.clone(), state);
    }

    pub fn get(&self, id: &acp::SessionId) -> Option<Arc<SessionState>> {
        self.inner.get(id).map(|r| r.value().clone())
    }

    pub fn remove(&self, id: &acp::SessionId) -> Option<Arc<SessionState>> {
        self.inner.remove(id).map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: &str, cwd: &str) -> Arc<SessionState> {
        let manager = PermissionManager::new(
            claurst_core::PermissionMode::Default,
            &claurst_core::config::Settings::default(),
        );
        let provider = crate::e2e_tests::ScriptedProvider::new(Vec::new());
        let runtime = crate::e2e_tests::scripted_runtime(std::path::Path::new(cwd), provider);
        SessionState::new(acp::SessionId::new(id), Arc::new(runtime), manager)
    }

    #[test]
    fn new_session_starts_empty_and_idle() {
        let id = acp::SessionId::new("session-1");
        let cwd = PathBuf::from("/tmp/claurst-test");
        let state = session("session-1", "/tmp/claurst-test");

        assert_eq!(state.session_id, id);
        assert_eq!(state.cwd, cwd);
        assert!(state.messages.lock().is_empty());
        assert_eq!(
            state.current_turn.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert!(!state.cancel_turn(), "a new session has no turn to cancel");
    }

    #[test]
    fn second_concurrent_turn_is_rejected() {
        let state = session("busy", "/tmp");
        let first = state.begin_turn().expect("idle session accepts a turn");
        assert!(state.begin_turn().is_none(), "a running turn must block a second one");
        drop(first);
        assert!(state.begin_turn().is_some(), "the session is idle again once the turn ends");
    }

    #[test]
    fn cancel_only_affects_the_running_turn() {
        let state = session("cancel", "/tmp");
        let first = state.begin_turn().expect("idle session accepts a turn");
        assert!(state.cancel_turn());
        assert!(first.token.is_cancelled());
        drop(first);

        let second = state.begin_turn().expect("idle session accepts a turn");
        assert!(
            !second.token.is_cancelled(),
            "a prompt after a cancel must start with a live token"
        );
    }

    #[test]
    fn registry_insert_get_remove_round_trip() {
        let registry = SessionRegistry::new();
        let id = acp::SessionId::new("session-2");
        let state = session("session-2", "/tmp");

        assert!(registry.get(&id).is_none());

        registry.insert(Arc::clone(&state));
        let fetched = registry.get(&id).expect("session should be present after insert");
        assert!(Arc::ptr_eq(&fetched, &state));

        let removed = registry.remove(&id).expect("session should be present to remove");
        assert!(Arc::ptr_eq(&removed, &state));
        assert!(registry.get(&id).is_none());
    }

    #[test]
    fn session_ids_that_could_leave_the_sessions_directory_are_refused() {
        assert!(valid_session_id("acp-1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"));
        assert!(valid_session_id("abc_DEF-123"));
        for bad in ["", "../secrets", "a/b", "a\\b", "C:x", "a.json", &"x".repeat(129)] {
            assert!(!valid_session_id(bad), "{bad:?} must be refused");
        }
    }

    #[test]
    fn title_is_the_first_non_empty_line_of_the_first_user_message() {
        let messages = vec![Message::user("\n  fix the build\nthen run tests")];
        assert_eq!(first_user_line(&messages).as_deref(), Some("fix the build"));
        assert_eq!(first_user_line(&[]), None);
    }

    #[test]
    fn registry_remove_unknown_id_returns_none() {
        let registry = SessionRegistry::new();
        let id = acp::SessionId::new("missing");
        assert!(registry.remove(&id).is_none());
    }
}
