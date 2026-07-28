use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use bridge_cli::{build_report, render_json, render_text};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "bridge",
    version,
    about = "Inspect selected Hy3 GGUF metadata and tensor directories"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect and validate one selected-profile Hy3 GGUF model set.
    InspectGguf {
        /// GGUF file or numbered shard in the model set.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        /// Emit one deterministic pretty-printed JSON report.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::InspectGguf { model, json } => inspect_gguf(&model, json),
    }
}

fn inspect_gguf(model: &Path, json: bool) -> Result<()> {
    let set = bridge_gguf_split::open_set(model)
        .with_context(|| format!("failed to open GGUF set {}", model.display()))?;
    let report =
        build_report(&set).with_context(|| format!("failed to validate Hy3 model {}", model.display()))?;
    let rendered = if json {
        render_json(&report).context("failed to serialize inspection report")?
    } else {
        render_text(&report)
    };

    io::stdout()
        .lock()
        .write_all(rendered.as_bytes())
        .context("failed to write inspection report to stdout")
}
