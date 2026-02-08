use std::sync::{Arc, Mutex, Condvar};
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;
struct ShortMem {
    model: Option<String>,
    user: Option<String>,
    updated: bool,
}

struct Shared {
    mem: Mutex<ShortMem>,
    condvar: Condvar,
}

fn construct(mem: &ShortMem) -> String {
    let user = mem.user.as_deref().unwrap_or("");
    let model = mem.model.as_deref().unwrap_or("");

    format!("Context:\n Vinny: {}\n User: {}", model, user)
}

fn core_loop(shared: Arc<Shared>) {
    let mut guard = shared.mem.lock().unwrap();
    loop {
        while !guard.updated {
                guard = shared.condvar.wait(guard).unwrap();
            }

            println!("{}", construct(&guard));

            guard.updated = false; // consume the event
    }
    
}





fn guilisten(shared: Arc<Shared>) {

    let listener = TcpListener::bind("127.0.0.1:9090").unwrap();

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buf = [0; 4096];
        let n = stream.read(&mut buf).unwrap();

        let msg = String::from_utf8_lossy(&buf[..n]).trim().to_string();

        let mut mem = shared.mem.lock().unwrap();
        mem.user = Some(msg);
        mem.updated = true;

        shared.condvar.notify_one();
    }
}

fn modellisten(shared: Arc<Shared>) {
    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buf = [0; 4096];
        let n = stream.read(&mut buf).unwrap();

        let msg = String::from_utf8_lossy(&buf[..n]).trim().to_string();

        let mut mem = shared.mem.lock().unwrap();
        mem.model = Some(msg);
        mem.updated = true;

        shared.condvar.notify_one(); 
    }
}






fn sendgui() {
    let mut stream = TcpStream::connect("127.0.0.1:8081").unwrap();
    stream.write_all(b"RAHGOOO").unwrap();
}

fn sendmodel(prompt: &str) {
    let prompt = prompt.to_string();
        thread::spawn(move || {
            if let Ok(mut stream) = std::net::TcpStream::connect("127.0.0.1:9091") {
                use std::io::Write;
                let _ = stream.write_all(prompt.as_bytes());
            }
        });
}




fn short() {
    let shared = Arc::new(Shared {
        mem: Mutex::new(ShortMem {
            model: None,
            user: None,
            updated: false,
        }),
        condvar: Condvar::new(),
    });

    let gui_shared = Arc::clone(&shared);
    let model_shared = Arc::clone(&shared);

    thread::spawn(move || guilisten(gui_shared));
    thread::spawn(move || modellisten(model_shared));

    core_loop(shared);

    

}

/*------------------------------------------------------------------------------------------------------------ */

fn long() {
    // Placeholder for a more complex implementation
    println!("Long memory system is not implemented yet.");
}


fn main() {
    short();
    //long();
}