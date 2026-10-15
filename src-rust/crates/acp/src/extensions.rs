//! Claurst extensions to the ACP stream, sent only to clients that ask.
//!
//! A client opts in at `initialize` through the reserved `_meta` field of its
//! capabilities:
//!
//! ```json
//! { "clientCapabilities": { "_meta": { "claurst": { "sessionEvents": true, "usageUpdates": true } } } }
//! ```
//!
//! - `usageUpdates`: a `usage_update` session update (context used / window
//!   size / session cost) after every model request.
//! - `sessionEvents`: the query-loop events plain ACP has no update for
//!   (retries, compactions, context warnings, turn cap, rate limits, status
//!   lines), sent as the extension notification `_claurst/session_event`
//!   (`{ "sessionId", "event": { "type", ... } }`). Method names starting with
//!   `_` are reserved by ACP for extensions.
//!
//! Clients that do not opt in get the plain ACP stream, unchanged.
//!
//! Two extension requests let a client add input to a running prompt, like a
//! message typed in the terminal UI while the agent works (advertised as
//! `agentCapabilities._meta.claurst.queueMessages`):
//!
//! - `_claurst/queue_message` `{ "sessionId", "text" }` → `{ "queued": true, "id" }`,
//!   or `{ "queued": false }` when no prompt is running. The message joins the
//!   conversation before the next model request (a `input_injected` session
//!   event reports it); one queued during the final step makes the turn go on.
//! - `_claurst/unqueue_message` `{ "sessionId", "id" }` → `{ "removed": bool }`.
//!
//! Messages a cancelled turn never delivered come back in the prompt
//! response, `_meta.claurst.undeliveredMessages`.
//!
//! `session/new` and `session/load` accept `_meta.claurst.appendSystemPrompt`:
//! instructions added to the system prompt of that session, after the ones
//! from the settings. `claurst acp` takes no command-line options, so this is
//! how a client sets them per session.

use agent_client_protocol_schema as acp;
use claurst_query::{QueryEvent, TokenWarningState};
use serde_json::{json, Value};

/// Method of the extension notification carrying session events.
pub const SESSION_EVENT: &str = "_claurst/session_event";

/// Request adding a message to the running prompt.
pub const QUEUE_MESSAGE: &str = "_claurst/queue_message";
/// Request taking back a queued message not delivered yet.
pub const UNQUEUE_MESSAGE: &str = "_claurst/unqueue_message";

/// `_meta.claurst.appendSystemPrompt` of a `session/new` or `session/load`
/// request, when it is a non-blank string.
pub fn session_system_prompt(meta: Option<&acp::Meta>) -> Option<String> {
    let text = meta?.get("claurst")?.get("appendSystemPrompt")?.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// The extensions a client opted into.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClientExtensions {
    pub session_events: bool,
    pub usage_updates: bool,
}

impl ClientExtensions {
    pub fn from_capabilities(caps: &acp::ClientCapabilities) -> Self {
        let ext = caps.meta.as_ref().and_then(|m| m.get("claurst"));
        let flag = |key: &str| {
            ext.and_then(|e| e.get(key))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        Self {
            session_events: flag("sessionEvents"),
            usage_updates: flag("usageUpdates"),
        }
    }
}

/// The `event` payload of a `_claurst/session_event` notification for `event`,
/// or `None` when plain ACP already covers it (text, tools) or it is not
/// worth reporting.
pub fn session_event(event: &QueryEvent) -> Option<Value> {
    Some(match event {
        QueryEvent::Retry {
            attempt,
            max_attempts,
            delay_ms,
            reason,
        } => json!({
            "type": "retry",
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "reason": reason,
        }),
        QueryEvent::Compacted(record) => json!({
            "type": "compacted",
            "trigger": record.trigger,
            "messagesBefore": record.messages_before,
            "messagesAfter": record.messages_after,
            "tokensBefore": record.tokens_before,
            "tokensAfter": record.tokens_after,
            "summarised": record.summarised,
        }),
        QueryEvent::TokenWarning { state, pct_used } => json!({
            "type": "context_warning",
            "level": match state {
                TokenWarningState::Ok => return None,
                TokenWarningState::Warning => "warning",
                TokenWarningState::Critical => "critical",
            },
            "pctUsed": pct_used,
        }),
        QueryEvent::MaxTurnsReached { max_turns } => json!({
            "type": "max_turns",
            "maxTurns": max_turns,
        }),
        QueryEvent::RateLimits {
            primary_used_percent,
            secondary_used_percent,
        } => json!({
            "type": "rate_limits",
            "primaryUsedPercent": primary_used_percent,
            "secondaryUsedPercent": secondary_used_percent,
        }),
        QueryEvent::Status(message) => json!({
            "type": "status",
            "message": message,
        }),
        // Queued messages (`_claurst/queue_message`) that just joined the
        // conversation, in queue order.
        QueryEvent::InputInjected(texts) => json!({
            "type": "input_injected",
            "messages": texts,
        }),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_are_off_unless_the_client_asks() {
        assert_eq!(
            ClientExtensions::from_capabilities(&acp::ClientCapabilities::default()),
            ClientExtensions::default()
        );
        let caps: acp::ClientCapabilities = serde_json::from_value(json!({
            "_meta": { "claurst": { "sessionEvents": true, "usageUpdates": false } }
        }))
        .expect("capabilities");
        assert_eq!(
            ClientExtensions::from_capabilities(&caps),
            ClientExtensions {
                session_events: true,
                usage_updates: false
            }
        );
    }

    #[test]
    fn retry_and_turn_cap_are_reported() {
        let retry = QueryEvent::Retry {
            attempt: 2,
            max_attempts: 5,
            delay_ms: 6000,
            reason: "overloaded".into(),
        };
        assert_eq!(
            session_event(&retry),
            Some(
                json!({ "type": "retry", "attempt": 2, "maxAttempts": 5, "delayMs": 6000, "reason": "overloaded" })
            )
        );
        assert_eq!(
            session_event(&QueryEvent::MaxTurnsReached { max_turns: 10 }),
            Some(json!({ "type": "max_turns", "maxTurns": 10 }))
        );
    }

    #[test]
    fn context_warnings_skip_the_ok_state() {
        let ok = QueryEvent::TokenWarning {
            state: TokenWarningState::Ok,
            pct_used: 0.1,
        };
        assert_eq!(session_event(&ok), None);
        let critical = QueryEvent::TokenWarning {
            state: TokenWarningState::Critical,
            pct_used: 0.97,
        };
        assert_eq!(session_event(&critical).unwrap()["level"], "critical");
    }

    #[test]
    fn streamed_content_is_left_to_plain_acp() {
        assert_eq!(session_event(&QueryEvent::Error("boom".into())), None);
    }
}
