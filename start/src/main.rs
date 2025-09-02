use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Search upwards from the current folder for PyAI
fn find_pyai_upwards() -> Option<PathBuf> {
    let mut dir = env::current_exe().ok()?.parent()?.to_path_buf();
    for _ in 0..10 {
        let candidate = dir.join("PyAI");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

/// Recursively search downwards from a folder for PyAI
fn find_pyai_downwards(start: &Path) -> Option<PathBuf> {
    for entry in fs::read_dir(start).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() && path.file_name()? == "PyAI" {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_pyai_downwards(&path) {
                return Some(found);
            }
        }
    }
    None
}

/// Locate PyAI folder either upwards or downwards
fn find_pyai() -> PathBuf {
    if let Some(up) = find_pyai_upwards() {
        return up;
    }
    let exe_dir = env::current_exe().unwrap().parent().unwrap().to_path_buf();
    if let Some(down) = find_pyai_downwards(&exe_dir) {
        return down;
    }
    panic!("Could not locate PyAI folder!");
}

/// Try to find a valid Python interpreter on the system
fn find_system_python() -> &'static str {
    if Command::new("python").arg("--version").output().is_ok() {
        "python"
    } else if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        panic!("No Python interpreter found on system (tried 'python' and 'python3')");
    }
}

fn main() {
    let project_dir = find_pyai();
    println!("Using PyAI folder: {}", project_dir.display());

    // Set venv path
    let venv_path = project_dir.join("venv");

    #[cfg(windows)]
    let python_path = venv_path.join("Scripts").join("python.exe");
    #[cfg(not(windows))]
    let python_path = venv_path.join("bin").join("python3");

    // Create venv if missing
    if !venv_path.exists() {
        println!("No venv found, creating one...");
        let system_python = find_system_python();
        let status = Command::new(system_python)
            .args(["-m", "venv", "venv"])
            .current_dir(&project_dir)
            .status()
            .expect("Failed to create venv");

        if !status.success() {
            panic!("Failed to create virtual environment");
        }
    }

    // Install requirements if file exists
    let req_file = project_dir.join("requirements.txt");
    if req_file.exists() {
        println!("Installing requirements...");
        let status = Command::new(&python_path)
            .args(["-m", "pip", "install", "-r", "requirements.txt"])
            .current_dir(&project_dir)
            .status()
            .expect("Failed to install requirements");

        if !status.success() {
            panic!("pip install failed");
        }
    }

    // Run Vinny.py
    println!("Running Vinny.py...");
    let status = Command::new(&python_path)
        .arg("Vinny.py")
        .current_dir(&project_dir)
        .status()
        .expect("Failed to run Vinny.py");

    println!("Vinny.py exited with status: {}", status);
}
