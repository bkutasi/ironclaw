//! Done tool for explicit completion signaling from container agents.

use async_trait::async_trait;

use crate::context::JobContext;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

/// Tool that signals the agent is done.
///
/// When the LLM calls this tool, the agentic loop breaks immediately
/// and the job is marked as completed with the provided summary.
/// This replaces fragile text-based completion detection.
pub struct DoneTool;

impl DoneTool {
    /// Create a new done tool.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DoneTool {
    fn name(&self) -> &str {
        "done"
    }

    fn description(&self) -> &str {
        "Signal that you are done. Call this tool when you have finished all work and are ready to report your final results."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The final summary/result of the job"
                }
            },
            "required": ["summary"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();
        let summary = require_str(&params, "summary")?;
        Ok(ToolOutput::text(summary, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal tool, no external data
    }
}
