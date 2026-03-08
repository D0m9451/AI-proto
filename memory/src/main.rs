use std::sync::{Arc, Mutex, Condvar};
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;

struct ShortMem {
    memory: Option<String>, // persistent memory from long-term module
    model: Option<String>, // latest model response
    user: Option<String>, // latest user input
    updated: bool, // flag to indicate if there's an update
    last_source: Option<Source>, // track whether the last update was from the model or user
} 

struct Shared {
    mem: Mutex<ShortMem>, // shared state protected by a mutex
    condvar: Condvar, // condition variable to signal updates
}

enum Source {
    Model, // update came from the model
    User, // update came from the user
}

fn construct(mem: &ShortMem, conversation: &mut String) -> String {
    let memory = mem.memory.as_deref().unwrap_or(""); // get the persistent memory, defaulting to empty string if not set
    let user = mem.user.as_deref().unwrap_or(""); // get the latest user input, defaulting to empty string if not set
    let model = mem.model.as_deref().unwrap_or(""); // get the latest model response, defaulting to empty string if not set

    conversation.push_str(
        &format!("Vinny: {}\nUser: {}\n", model, user)
    ); // append latest exchange to conversation history

    format!("Persistent User data:\n {}\n 
Rules:
    - The persistant User data contains long-term information to help you respond better.\n
    - Use it when relevant\n
    - Do not mention it to the user unless asked\n
Conversation:\n {}\n", memory, conversation) // construct the prompt for the model, including the persistent memory and conversation history
    
}

fn core_loop(shared: Arc<Shared>, conversation: &mut String) {
    let mut guard = shared.mem.lock().unwrap(); // lock the shared state to start the loop

    loop {

        while !guard.updated {
            guard = shared.condvar.wait(guard).unwrap();
        } // wait until there's an update from either the model or the user

        match guard.last_source {
            Some(Source::User) => {
                let prompt = construct(&guard, conversation); // construct the prompt for the model based on the latest user input and the conversation history

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

                guard.updated = false; // consume the event by resetting the flag, the model response has already been incorporated into the conversation history in the construct function, so we just need to reset the flag and continue waiting for the next user input or model response

            }

            None => {
                guard.updated = false; // should never happen, but just reset the flag and continue waiting if it does
            }
        }
    }
}



fn longlisten(shared: Arc<Shared>) {

    let listener = TcpListener::bind("127.0.0.1:8087").unwrap(); // listen for incoming connections on port 8087 for the long-term memory module

    for stream in listener.incoming() {
        let mut stream = stream.unwrap(); // accept the incoming connection and unwrap it to get the stream
        let mut buf = [0; 4096]; // buffer to read the incoming data, adjust size as needed
        let n = stream.read(&mut buf).unwrap(); // read the data into the buffer and get the number of bytes read

        let longmem = String::from_utf8_lossy(&buf[..n]).trim().to_string(); // convert the bytes read into a string, trimming any whitespace

        let mut mem = shared.mem.lock().unwrap(); // lock the shared state to update the persistent memory with the new long-term memory received from the long-term memory module
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
        mem.user = Some(msg); // update the latest user input in the shared state
        mem.last_source = Some(Source::User); // set the last source to User since this update came from the user input
        mem.updated = true; // set the updated flag to true to signal that there's a new user input that needs to be processed by the core loop

        shared.condvar.notify_one(); // notify the core loop that there's an update, so it can wake up and process the new user input
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
                let clean = response.replace("<<END>>", "").trim().to_string(); // remove the end marker

                let mut mem = shared.mem.lock().unwrap(); // lock the shared state to update the latest model response with the clean response that has the end marker removed
                mem.model = Some(clean); // update the latest model response in the shared state with the clean response that has the end marker removed
                mem.last_source = Some(Source::Model); // set the last source to Model since this update came from the model response
                mem.updated = true; // set the updated flag to true to signal that there's a new model response that needs to be processed by the core loop, which will incorporate it into the conversation history and then wait for the next user input
                shared.condvar.notify_one(); // notify the core loop that there's an update, so it can wake up and process the new model response

                response.clear(); // clear the response buffer for the next response
                break;
            }
        }
    }
}



fn sendmodel(prompt: &str) {
    let prompt = prompt.to_string();
    thread::spawn(move || {
        if let Ok(mut stream) = std::net::TcpStream::connect("127.0.0.1:9091") {
            let _ = stream.write_all(format!("{}\n",prompt).as_bytes()); // send the prompt to the model on port 9091, adding a newline at the end to signal the end of the prompt
        }
            
    });
}




fn short() {
    let mut conversation = String::new();
    let shared = Arc::new(Shared { // create the shared state
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
    // create clones of the shared state reference for each listener thread, so they can all access and update the same shared state

    thread::spawn(move || guilisten(gui_shared));
    thread::spawn(move || modellisten(model_shared));
    thread::spawn(move || longlisten(longmem_shared));
    // spawn the listener threads for the GUI, model, and long-term memory modules, passing them their respective references to the shared state

    core_loop(shared, &mut conversation); // start the core loop, which will process updates from the listeners and manage the conversation history and prompt construction for the model

    

}

fn main() {
    short();
}