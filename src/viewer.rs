//! Open output files in VS Code, and set it up to view the 3D plot files.
//!
//! The game installs the PLY viewer extension on first use, so the player only
//! runs `show <puzzle> --plot`. If the `code` command is not on PATH, the
//! functions report that and the game prints the file paths instead.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const VIEWER_EXTENSION_ID: &str = "kleinicke.ply-visualizer";

pub enum ViewerStatus {
    /// Already installed.
    Ready,
    /// Installed now.
    Installed,
    /// The `code` command is not on PATH.
    NoVsCode,
    Failed,
}

/// Find a command on PATH, like Python's `shutil.which` (PATHEXT on Windows).
fn which(name: &str) -> Option<PathBuf> {
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT").unwrap_or(".COM;.EXE;.BAT;.CMD".into()).split(';').map(str::to_lowercase).collect()
    } else {
        vec![String::new()]
    };
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|dir| {
        extensions.iter().map(|ext| dir.join(format!("{name}{ext}"))).find(|p| is_executable(p))
    })
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Run a command, capturing its output, and give up after `timeout`.
fn run(program: &Path, args: &[&std::ffi::OsStr], timeout: Duration) -> Option<Output> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Install the PLY viewer into VS Code if it is missing.
pub fn ensure_viewer_extension() -> ViewerStatus {
    let Some(code) = which("code") else { return ViewerStatus::NoVsCode };
    let Some(listed) = run(&code, &["--list-extensions".as_ref()], Duration::from_secs(120)) else {
        return ViewerStatus::Failed;
    };
    let installed = String::from_utf8_lossy(&listed.stdout).lines().any(|l| l.trim().to_lowercase() == VIEWER_EXTENSION_ID);
    if installed {
        return ViewerStatus::Ready;
    }
    let args = ["--install-extension".as_ref(), VIEWER_EXTENSION_ID.as_ref(), "--force".as_ref()];
    match run(&code, &args, Duration::from_secs(300)) {
        Some(out) if out.status.success() => ViewerStatus::Installed,
        _ => ViewerStatus::Failed,
    }
}

/// Open the given files in VS Code, reusing the current window.
pub fn open_in_vscode(paths: &[PathBuf]) -> bool {
    let Some(code) = which("code") else { return false };
    if paths.is_empty() {
        return false;
    }
    let mut args: Vec<&std::ffi::OsStr> = vec!["--reuse-window".as_ref()];
    args.extend(paths.iter().map(|p| p.as_os_str()));
    run(&code, &args, Duration::from_secs(120)).is_some_and(|out| out.status.success())
}
