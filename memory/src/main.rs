use std::sync::{Arc, Mutex, Condvar};
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;

struct ShortMem {
    memory: Option<String>,
    model: Option<String>,
    user: Option<String>,
    updated: bool,
    last_source: Option<Source>,
}

struct Shared {
    mem: Mutex<ShortMem>,
    condvar: Condvar,
}

enum Source {
    Model,
    User,
}

fn construct(mem: &ShortMem, conversation: &mut String) -> String {
    let memory = mem.memory.as_deref().unwrap_or("");
    let user = mem.user.as_deref().unwrap_or("");
    let model = mem.model.as_deref().unwrap_or("");

    conversation.push_str(
        &format!("Vinny: {}\nUser: {}\n", model, user)
    );

    format!("Persistent User data:\n {}\n 
Rules:
    - The persistant User data contains long-term information to help you respond better.\n
    - Use it when relevant\n
    - Do not mention it to the user unless asked\n
Conversation:\n {}\n", memory, conversation)
    
}

fn core_loop(shared: Arc<Shared>, conversation: &mut String) {
    let mut guard = shared.mem.lock().unwrap();

    loop {

        while !guard.updated {
            guard = shared.condvar.wait(guard).unwrap();
        }

        match guard.last_source {
            Some(Source::User) => {
                let prompt = construct(&guard, conversation);

                // consume the event
                guard.updated = false;

                // unlock BEFORE network I/O
                drop(guard);
                
                println!("\n{}", prompt);
                sendmodel(&prompt);

                // re-lock and continue loop
                guard = shared.mem.lock().unwrap();
            }

            Some(Source::Model) => {
                /*if let (Some(user), Some(model)) = (&guard.user, &guard.model) {
                    conversation.push_str(
                        &format!("User: {}\nVinny: {}\n", user, model)
                    );*/

                guard.updated = false;

            }

            None => {
                guard.updated = false;
            }
        }
    }
}



fn longlisten(shared: Arc<Shared>) {

    let listener = TcpListener::bind("127.0.0.1:8087").unwrap();

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buf = [0; 4096];
        let n = stream.read(&mut buf).unwrap();

        let longmem = String::from_utf8_lossy(&buf[..n]).trim().to_string();

            let mut mem = shared.mem.lock().unwrap();
            mem.memory = Some(longmem);
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
        mem.last_source = Some(Source::User);
        mem.updated = true;

        shared.condvar.notify_one();
    }
}

fn modellisten(shared: Arc<Shared>) {
    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    let mut response = String::new();

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        let mut buf = [0; 1024];

        loop {
            let n = stream.read(&mut buf).unwrap();
            if n == 0 { break; }

            let chunk = String::from_utf8_lossy(&buf[..n]);
            response.push_str(&chunk);

            if response.contains("<<END>>") {
                let clean = response.replace("<<END>>", "").trim().to_string();

                let mut mem = shared.mem.lock().unwrap();
                mem.model = Some(clean);
                mem.last_source = Some(Source::Model);
                mem.updated = true;
                shared.condvar.notify_one();

                response.clear();
                break;
            }
        }
    }
}



fn sendmodel(prompt: &str) {
    let prompt = prompt.to_string();
    thread::spawn(move || {
        if let Ok(mut stream) = std::net::TcpStream::connect("127.0.0.1:9091") {
            let _ = stream.write_all(format!("{}\n",prompt).as_bytes());
        }
            
    });
}




fn short() {
    let mut conversation = String::new();
    let shared = Arc::new(Shared {
        mem: Mutex::new(ShortMem {
            memory: None,
            model: None,
            user: None,
            updated: false,
            last_source: None,
        }),
        condvar: Condvar::new(),
    });

    let gui_shared = Arc::clone(&shared);
    let model_shared = Arc::clone(&shared);
    let longmem_shared = Arc::clone(&shared);
    
    thread::spawn(move || guilisten(gui_shared));
    thread::spawn(move || modellisten(model_shared));
    thread::spawn(move || longlisten(longmem_shared));

    core_loop(shared, &mut conversation);

    

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