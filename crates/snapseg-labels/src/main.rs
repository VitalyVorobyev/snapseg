//! `snapseg-labels` — companion CLI for the snapseg label format.
//!
//! Subcommands:
//!   - `coco`: roll a `labels/` directory into a COCO-shaped JSON
//!     dataset for downstream training. See
//!     [`snapseg_labels::coco`] for the on-disk shape.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Top-level CLI definition.
#[derive(Parser, Debug)]
#[command(name = "snapseg-labels", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Subcommands exposed by the binary.
#[derive(Subcommand, Debug)]
enum Cmd {
    /// Convert a labels/ directory to a COCO-shaped JSON file.
    Coco {
        /// Path to the labels/ root (the directory that contains
        /// per-label subdirectories named by SHA-8).
        #[arg(short, long, value_name = "DIR")]
        r#in: PathBuf,
        /// Output JSON file path.
        #[arg(short, long, value_name = "FILE")]
        out: PathBuf,
        /// Skip labels whose meta.toml has `operator.quality = reject`.
        /// Defaults to true. Pass `--skip-rejected false` (or
        /// `--no-skip-rejected` via clap's negation, if enabled) to
        /// include rejected labels.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        skip_rejected: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Coco {
            r#in,
            out,
            skip_rejected,
        } => {
            let n = snapseg_labels::coco::convert_dir_to_file(
                &r#in,
                &out,
                snapseg_labels::coco::ConvertOptions { skip_rejected },
            )?;
            eprintln!("wrote {} labels to {}", n, out.display());
        }
    }
    Ok(())
}
