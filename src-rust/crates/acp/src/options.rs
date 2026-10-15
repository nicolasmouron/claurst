//! Per-session options a client sets in `_meta.claurst.options` of `session/new`
//! or `session/load`, since `claurst acp` takes no command-line options. They
//! mirror the CLI flags of the same name and apply to that session only:
//!
//! ```json
//! { "maxTurns": 100, "maxTokens": 8000, "thinking": 4000, "maxBudgetUsd": 2.5,
//!   "fallbackModel": "provider/model", "autoCompact": false,
//!   "allowedTools": "Read, Bash(git:*)", "disallowedTools": "WebFetch",
//!   "additionalDirs": ["/abs/path"], "noClaudeMd": true, "autoCommits": true }
//! ```

use std::path::PathBuf;

use agent_client_protocol_schema as acp;
use claurst_core::config::Config;
use claurst_core::permissions::{parse_rule_spec, split_rule_specs, PermissionAction};
use claurst_query::QueryConfig;
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct SessionOptions {
    /// Model requests per prompt (`--max-turns`).
    pub max_turns: Option<u32>,
    /// Output tokens per response (`--max-tokens`).
    pub max_tokens: Option<u32>,
    /// Extended-thinking budget in tokens (`--thinking`).
    pub thinking: Option<u32>,
    /// Stop the loop past this spend (`--max-budget-usd`).
    pub max_budget_usd: Option<f64>,
    /// Model used when the main one is overloaded (`--fallback-model`).
    pub fallback_model: Option<String>,
    /// `false` turns automatic compaction off (`--no-auto-compact`).
    pub auto_compact: Option<bool>,
    /// Extra allow rules, `Tool`, `Tool(pattern)` or `*`, comma separated (`--allowed-tools`).
    pub allowed_tools: Option<String>,
    /// Extra deny rules, same syntax (`--disallowed-tools`).
    pub disallowed_tools: Option<String>,
    /// Directories the tools may reach besides the working one (`--add-dir`).
    pub additional_dirs: Vec<PathBuf>,
    /// Ignore AGENTS.md / CLAUDE.md instruction files (`--no-claude-md`).
    pub no_claude_md: bool,
    /// Git shadow snapshots before edits (`--auto-commits`).
    pub auto_commits: Option<bool>,
}

impl SessionOptions {
    /// The options of a `session/new` or `session/load` request: absent means
    /// the defaults; invalid ones are an error for the client to show.
    pub fn from_meta(meta: Option<&acp::Meta>) -> Result<Self, String> {
        let Some(value) = meta.and_then(|m| m.get("claurst")).and_then(|c| c.get("options")) else {
            return Ok(Self::default());
        };
        let options: Self =
            serde_json::from_value(value.clone()).map_err(|e| format!("invalid options: {e}"))?;
        for (name, list) in [
            ("allowedTools", &options.allowed_tools),
            ("disallowedTools", &options.disallowed_tools),
        ] {
            for spec in list.as_deref().map(split_rule_specs).unwrap_or_default() {
                if parse_rule_spec(&spec, PermissionAction::Allow).is_none() {
                    return Err(format!(
                        "invalid {name} entry {spec:?}: expected Tool, Tool(pattern) or *"
                    ));
                }
            }
        }
        if let Some(dir) = options.additional_dirs.iter().find(|d| !d.is_absolute()) {
            return Err(format!("additionalDirs must be absolute: {}", dir.display()));
        }
        if options.max_turns == Some(0) {
            return Err("maxTurns must be at least 1".to_string());
        }
        Ok(options)
    }

    /// The session's view of the runtime settings: tool rules, extra
    /// directories, snapshots. Feeds the permission manager and the tools.
    pub fn apply_config(&self, config: &mut Config) {
        let specs = |list: &Option<String>| list.as_deref().map(split_rule_specs).unwrap_or_default();
        config.allowed_tools.extend(specs(&self.allowed_tools));
        config.disallowed_tools.extend(specs(&self.disallowed_tools));
        config.additional_dirs.extend(self.additional_dirs.iter().cloned());
        if self.auto_commits.is_some() {
            config.auto_commits = self.auto_commits;
        }
        if let Some(auto) = self.auto_compact {
            config.auto_compact = auto;
        }
        if self.no_claude_md {
            config.disable_claude_mds = true;
        }
    }

    /// The prompt's query-loop settings.
    pub fn apply_query(&self, query: &mut QueryConfig) {
        if let Some(turns) = self.max_turns {
            query.max_turns = turns;
        }
        if let Some(tokens) = self.max_tokens {
            query.max_tokens = tokens;
        }
        if let Some(budget) = self.thinking {
            query.thinking_budget = Some(budget);
        }
        if let Some(usd) = self.max_budget_usd {
            query.max_budget_usd = Some(usd);
        }
        if let Some(model) = self.fallback_model.as_ref().filter(|m| !m.trim().is_empty()) {
            query.fallback_model = Some(model.clone());
        }
        if let Some(auto) = self.auto_compact {
            query.auto_compact = auto;
        }
        if self.no_claude_md {
            query.instruction_files.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(options: serde_json::Value) -> acp::Meta {
        json!({ "claurst": { "options": options } }).as_object().cloned().unwrap()
    }

    #[test]
    fn absent_options_are_the_defaults() {
        assert_eq!(SessionOptions::from_meta(None), Ok(SessionOptions::default()));
        let other = json!({ "claurst": { "appendSystemPrompt": "x" } }).as_object().cloned().unwrap();
        assert_eq!(SessionOptions::from_meta(Some(&other)), Ok(SessionOptions::default()));
    }

    #[test]
    fn options_reach_the_config_and_the_query_loop() {
        let o = SessionOptions::from_meta(Some(&meta(json!({
            "maxTurns": 250, "maxTokens": 8000, "thinking": 4000, "maxBudgetUsd": 2.5,
            "fallbackModel": "p/backup", "autoCompact": false,
            "allowedTools": "Read, Bash(git status:*)", "disallowedTools": "WebFetch",
            "additionalDirs": [std::env::temp_dir()], "noClaudeMd": true, "autoCommits": true,
        }))))
        .unwrap();

        let mut config = Config::default();
        o.apply_config(&mut config);
        assert_eq!(config.allowed_tools, ["Read", "Bash(git status:*)"]);
        assert_eq!(config.disallowed_tools, ["WebFetch"]);
        assert_eq!(config.additional_dirs, [std::env::temp_dir()]);
        assert_eq!(config.auto_commits, Some(true));
        assert!(config.disable_claude_mds && !config.auto_compact);

        let mut query = QueryConfig::from_config(&Config::default());
        o.apply_query(&mut query);
        assert_eq!(query.max_turns, 250);
        assert_eq!(query.max_tokens, 8000);
        assert_eq!(query.thinking_budget, Some(4000));
        assert_eq!(query.max_budget_usd, Some(2.5));
        assert_eq!(query.fallback_model.as_deref(), Some("p/backup"));
        assert!(!query.auto_compact);
    }

    #[test]
    fn invalid_options_are_refused() {
        for bad in [
            json!({ "allowedTools": "Bash(unclosed" }),
            json!({ "additionalDirs": ["relative/dir"] }),
            json!({ "maxTurns": 0 }),
            json!({ "maxTurns": "many" }),
            json!({ "unknownOption": true }),
        ] {
            assert!(SessionOptions::from_meta(Some(&meta(bad.clone()))).is_err(), "{bad}");
        }
    }
}
