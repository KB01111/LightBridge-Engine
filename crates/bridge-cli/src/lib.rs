//! Deterministic reports for checked model inspection.

mod report;
mod text;

pub use report::{
    build_report, Aggregate, ExpertProjectionReport, ExpertStorageReport, FileReport, GeneralReport,
    GgufReport, Hy3Report, InspectionReport, ReportError, TensorSummary, TokenizerReport,
};
pub use text::render_text;

pub fn render_json(report: &InspectionReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report).map(|mut json| {
        json.push('\n');
        json
    })
}
