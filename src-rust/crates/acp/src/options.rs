//! Per-session options a client sets in `_meta.claurst.options` of `session/new`
//! or `session/load`, since `claurst acp` takes no command-line options. They
//! mirror the CLI flags of the same name and apply to that session only:
//!
//! ```json
//! { "provider": "custom-openai2", "model": "custom-openai2/qwen", "apiBase": "http://host:8080/v1",
//!   "effort": "low", "agent": "plan", "maxTurns": 100, "maxTokens": 8000, "thinking": 4000, "maxBudgetUsd": 2.5,
//!   "fallbackModel": "provider/model", "autoCompact": false,
//!   "allowedTools": "Read, Bash(git:*)", "disallowedTools": "WebFetch",
//!   "additionalDirs": ["/abs/path"], "noClaudeMd": true, "autoCommits": true }
//! ```

use std::path::PathBuf;

use agent_client_protocol_schema as acp;
use claurst_core::config::Config;
use claurst_core::AgentDefinition;
use claurst_tools::{PermissionLevel, Tool};
use claurst_core::effort::EffortLevel;
use claurst_core::permissions::{parse_rule_spec, split_rule_specs, PermissionAction};
use claurst_query::QueryConfig;
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct SessionOptions {
    /// Provider of the session's model (`--provider`): several sessions can
    /// use different providers in one process.
    pub provider: Option<String>,
    /// Model, `provider/model` or bare (`--model`).
    pub model: Option<String>,
    /// Base URL of that provider for this session (`--api-base`).
    pub api_base: Option<String>,
    /// Reasoning effort (`--effort`): low, medium, high, max or none.
    pub effort: Option<String>,
    /// Named agent (`--agent`): a built-in one (build, plan, explore) or one
    /// from the settings' `agents`. Its prompt, model and turn cap apply, and
    /// its access level restricts the tools offered to the model.
    pub agent: Option<String>,
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
    /// Refuse an agent name that neither the settings nor the built-ins define.
    pub fn check_agent(&self, config: &Config) -> Result<(), String> {
        match self.agent.as_deref().filter(|a| !a.is_empty()) {
            Some(name) if agent_definition(config, name).is_none() => {
                let mut known: Vec<String> = claurst_core::default_agents().into_keys().collect();
                known.extend(config.agents.keys().cloned());
                known.sort();
                known.dedup();
                Err(format!("unknown agent {name:?} (known: {})", known.join(", ")))
            }
            _ => Ok(()),
        }
    }

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
        if let Some(effort) = options.effort.as_deref().filter(|e| !e.is_empty()) {
            if EffortLevel::from_str(effort).is_none() {
                return Err(format!("invalid effort {effort:?}"));
            }
        }
        for (name, value) in [("provider", &options.provider), ("model", &options.model)] {
            if value.as_deref().is_some_and(|v| v.trim().is_empty()) {
                return Err(format!("{name} must not be empty"));
            }
        }
        if options.max_turns == Some(0) {
            return Err("maxTurns must be at least 1".to_string());
        }
        Ok(options)
    }

    /// The session's view of the runtime settings: tool rules, extra
    /// directories, snapshots. Feeds the permission manager and the tools.
    pub fn apply_config(&self, config: &mut Config) {
        if let Some(provider) = &self.provider {
            config.provider = Some(provider.clone());
        }
        if let Some(model) = &self.model {
            config.model = Some(model.clone());
        }
        if let Some(base) = self.api_base.as_ref().filter(|b| !b.trim().is_empty()) {
            let id = config.selected_provider_id().to_string();
            config.provider_configs.entry(id).or_default().api_base = Some(base.clone());
        }
        if let Some(effort) = self.effort.as_ref().filter(|e| !e.is_empty()) {
            config.effort = Some(effort.clone());
        }
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

    /// The prompt's query-loop settings. The named agent, if any, comes from
    /// `config` (the session's view of the settings).
    pub fn apply_query_with_agent(&self, query: &mut QueryConfig, config: &Config) {
        self.apply_query(query);
        let Some(name) = self.agent.as_deref().filter(|a| !a.is_empty()) else { return };
        let Some(definition) = agent_definition(config, name) else { return };
        // Like `--agent`: the agent's turn cap applies, unless the session sets its own.
        if let (Some(turns), None) = (definition.max_turns, self.max_turns) {
            query.max_turns = turns;
        }
        query.agent_name = Some(name.to_string());
        query.agent_definition = Some(definition);
    }

    /// The prompt's query-loop settings.
    pub fn apply_query(&self, query: &mut QueryConfig) {
        if let Some(model) = &self.model {
            query.model = model.clone();
        }
        if let Some(effort) = self.effort.as_deref().and_then(EffortLevel::from_str) {
            query.effort_level = Some(effort);
        }
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

/// Definition of a named agent: the settings' `agents` over the built-ins.
pub fn agent_definition(config: &Config, name: &str) -> Option<AgentDefinition> {
    config.agents.get(name).cloned().or_else(|| claurst_core::default_agents().remove(name))
}

/// The tools an agent's access level allows, rebuilt from the built-in tools
/// like the CLI's `--agent` does (MCP tools and sub-agents are left out of the
/// restricted levels), or `None` for full access. The plan-mode tools only
/// switch the terminal UI's mode, so they are left out too: in ACP they would
/// tell the model it may now write.
pub fn tools_for_access(tools: &[Box<dyn Tool>], access: &str) -> Option<Vec<Box<dyn Tool>>> {
    const PLAN_MODE_TOOLS: &[&str] = &["EnterPlanMode", "ExitPlanMode"];
    const SEARCH_TOOLS: &[&str] = &["Grep", "Glob", "Read", "WebSearch", "WebFetch"];
    let allowed: Vec<String> = match access {
        "read-only" => tools
            .iter()
            .filter(|t| {
                matches!(t.permission_level(), PermissionLevel::ReadOnly | PermissionLevel::None)
                    || t.name() == "AskUserQuestion"
            })
            .map(|t| t.name().to_string())
            .collect(),
        "search-only" => SEARCH_TOOLS.iter().map(|n| n.to_string()).collect(),
        _ => return None,
    };
    Some(
        claurst_tools::all_tools()
            .into_iter()
            .filter(|t| allowed.iter().any(|n| n == t.name()) && !PLAN_MODE_TOOLS.contains(&t.name()))
            .collect(),
    )
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
    fn a_session_picks_its_own_provider_model_and_effort() {
        let o = SessionOptions::from_meta(Some(&meta(json!({
            "provider": "custom-openai2", "model": "custom-openai2/qwen",
            "apiBase": "http://pixelia03:3002/v1", "effort": "high",
        }))))
        .unwrap();
        let mut config = Config::default();
        config.provider = Some("custom-openai".into());
        o.apply_config(&mut config);
        assert_eq!(config.selected_provider_id(), "custom-openai2");
        assert_eq!(
            config.provider_configs["custom-openai2"].api_base.as_deref(),
            Some("http://pixelia03:3002/v1")
        );
        let mut query = QueryConfig::from_config(&Config::default());
        o.apply_query(&mut query);
        assert_eq!(query.model, "custom-openai2/qwen");
        assert_eq!(query.effort_level, EffortLevel::from_str("high"));
    }

    #[test]
    fn the_plan_agent_reads_and_never_writes() {
        let tools = claurst_tools::all_tools();
        let names = |list: &[Box<dyn Tool>]| list.iter().map(|t| t.name().to_string()).collect::<Vec<_>>();
        let plan = names(&tools_for_access(&tools, "read-only").unwrap());
        assert!(plan.iter().any(|n| n == "Read") && plan.iter().any(|n| n == "Grep"));
        for forbidden in ["Write", "Edit", "Bash", "ApplyPatch", "ExitPlanMode", "EnterPlanMode"] {
            assert!(!plan.iter().any(|n| n == forbidden), "{forbidden} offered to the plan agent");
        }
        let explore = names(&tools_for_access(&tools, "search-only").unwrap());
        assert!(explore.iter().all(|n| ["Grep", "Glob", "Read", "WebSearch", "WebFetch"].contains(&n.as_str())));
        assert!(tools_for_access(&tools, "full").is_none());
    }

    #[test]
    fn a_named_agent_brings_its_definition_and_turn_cap() {
        let config = Config::default();
        let o = SessionOptions::from_meta(Some(&meta(json!({ "agent": "plan" })))).unwrap();
        assert!(o.check_agent(&config).is_ok());
        let mut query = QueryConfig::from_config(&config);
        o.apply_query_with_agent(&mut query, &config);
        assert_eq!(query.agent_name.as_deref(), Some("plan"));
        assert_eq!(query.max_turns, 20, "the plan agent's own cap");

        let o = SessionOptions::from_meta(Some(&meta(json!({ "agent": "plan", "maxTurns": 60 })))).unwrap();
        let mut query = QueryConfig::from_config(&config);
        o.apply_query_with_agent(&mut query, &config);
        assert_eq!(query.max_turns, 60, "the session's own cap wins");

        let unknown = SessionOptions::from_meta(Some(&meta(json!({ "agent": "wizard" })))).unwrap();
        assert!(unknown.check_agent(&config).unwrap_err().contains("plan"));
    }

    #[test]
    fn invalid_options_are_refused() {
        for bad in [
            json!({ "allowedTools": "Bash(unclosed" }),
            json!({ "additionalDirs": ["relative/dir"] }),
            json!({ "maxTurns": 0 }),
            json!({ "maxTurns": "many" }),
            json!({ "unknownOption": true }),
            json!({ "effort": "extreme" }),
            json!({ "provider": " " }),
        ] {
            assert!(SessionOptions::from_meta(Some(&meta(bad.clone()))).is_err(), "{bad}");
        }
    }
}
