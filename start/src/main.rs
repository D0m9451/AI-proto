use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let exe_path = env::current_exe().unwrap();
    let mut project_dir = exe_path.parent().unwrap().to_path_buf();
    project_dir.pop();
    project_dir.pop();
    project_dir.pop();
    project_dir.push("PyAI");
    
    let venv_path = project_dir.join("venv");
    let mut python_path = venv_path.join("Scripts").join("python.exe");

    
    if !venv_path.exists() {
        println!("No venv found, creating one...");

        let system_python = "python";
        let venv_status = Command::new(system_python)
            .args(["-m", "venv", "venv"])
            .current_dir(&project_dir)
            .status()
            .expect("failed to create venv");

        if !venv_status.success() {
            panic!("Failed to create virtual environment");
        }
    }

    
    let pip_status = Command::new(&python_path)
        .args(["-m", "pip", "install", "-r", "requirements.txt"])
        .current_dir(&project_dir)
        .status()
        .expect("failed to run pip install");
    println!("pip exited with status: {}", pip_status);

    
    let script_status = Command::new(&python_path)
        .arg("Vinny.py")
        .current_dir(&project_dir)
        .status()
        .expect("failed to run python");
    println!("Python exited with status: {}", script_status);
}
