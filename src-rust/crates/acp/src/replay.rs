//! Replay of a loaded session's history, as `session/load` requires: the
//! client rebuilds the conversation from ordinary `session/update`s.

use std::sync::Arc;

use agent_client_protocol_schema as acp;
use claurst_core::types::{ContentBlock, Message, Role, ToolResultContent};

use crate::connection::Connection;
use crate::prompt::{classify_tool_kind, send_session_update, tool_title};

/// The updates that replay `messages`, in order.
pub fn history_updates(messages: &[Message]) -> Vec<acp::SessionUpdate> {
    let mut out = Vec::new();
    for message in messages {
        for block in message.content_blocks() {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(
                        acp::TextContent::new(text),
                    ));
                    out.push(match message.role {
                        Role::User => acp::SessionUpdate::UserMessageChunk(chunk),
                        Role::Assistant => acp::SessionUpdate::AgentMessageChunk(chunk),
                    });
                }
                ContentBlock::Image { source } => {
                    if let (Some(data), Some(mime)) = (source.data, source.media_type) {
                        let chunk = acp::ContentChunk::new(acp::ContentBlock::Image(
                            acp::ImageContent::new(data, mime),
                        ));
                        out.push(acp::SessionUpdate::UserMessageChunk(chunk));
                    }
                }
                ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
                    let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(
                        acp::TextContent::new(thinking),
                    ));
                    out.push(acp::SessionUpdate::AgentThoughtChunk(chunk));
                }
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    let call = acp::ToolCall::new(
                        acp::ToolCallId::new(id.as_str()),
                        tool_title(&name, Some(&input)),
                    )
                    .kind(classify_tool_kind(&name))
                    .status(acp::ToolCallStatus::InProgress)
                    .raw_input(Some(input));
                    out.push(acp::SessionUpdate::ToolCall(call));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let status = if is_error == Some(true) {
                        acp::ToolCallStatus::Failed
                    } else {
                        acp::ToolCallStatus::Completed
                    };
                    let content = vec![acp::ToolCallContent::Content(acp::Content::new(
                        acp::ContentBlock::Text(acp::TextContent::new(result_text(&content))),
                    ))];
                    let fields = acp::ToolCallUpdateFields::new()
                        .status(status)
                        .content(content);
                    out.push(acp::SessionUpdate::ToolCallUpdate(
                        acp::ToolCallUpdate::new(
                            acp::ToolCallId::new(tool_use_id.as_str()),
                            fields,
                        ),
                    ));
                }
                _ => {}
            }
        }
    }
    out
}

fn result_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(text) => text.clone(),
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Send the replay of `messages` to the client.
pub async fn replay(
    connection: &Arc<Connection>,
    session_id: &acp::SessionId,
    messages: &[Message],
) {
    for update in history_updates(messages) {
        send_session_update(connection, session_id, update).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::MessageContent;
    use serde_json::json;

    fn kinds(updates: &[acp::SessionUpdate]) -> Vec<String> {
        updates
            .iter()
            .map(|u| {
                serde_json::to_value(u).unwrap()["sessionUpdate"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn a_turn_with_a_tool_replays_in_order() {
        let assistant = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "look first".into(),
                    signature: String::new(),
                },
                ContentBlock::Text {
                    text: "Reading it.".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "Read".into(),
                    input: json!({ "file_path": "a.txt" }),
                    thought_signature: None,
                },
            ]),
            uuid: None,
            cost: None,
            snapshot_patch: None,
        };
        let result = Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: ToolResultContent::Text("hello".into()),
                is_error: Some(true),
            }]),
            uuid: None,
            cost: None,
            snapshot_patch: None,
        };
        let updates = history_updates(&[Message::user("read a.txt"), assistant, result]);
        assert_eq!(
            kinds(&updates),
            [
                "user_message_chunk",
                "agent_thought_chunk",
                "agent_message_chunk",
                "tool_call",
                "tool_call_update"
            ]
        );
        let call = serde_json::to_value(&updates[3]).unwrap();
        assert_eq!(call["title"], "Read: a.txt");
        assert_eq!(call["kind"], "read");
        let done = serde_json::to_value(&updates[4]).unwrap();
        assert_eq!(done["status"], "failed");
        assert_eq!(done["content"][0]["content"]["text"], "hello");
    }
}
