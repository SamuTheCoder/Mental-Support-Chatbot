import logo from './logo.svg';
import './App.css';
import React, { useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");

    const sendMessage = async () => {
        const response = await fetch("/chatbot/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: input }),
        });
        const data = await response.json();
        setMessages([...messages, { user: input, bot: data.response }]);
        setInput("");
    };

    return (
        <div>
            <h1>Chatbot</h1>
            <div>
                {messages.map((msg, index) => (
                    <div key={index}>
                        <p><strong>User:</strong> {msg.user}</p>
                        <p><strong>Bot:</strong> {msg.bot}</p>
                    </div>
                ))}
            </div>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message"
            />
            <button onClick={sendMessage}>Send</button>
        </div>
    );
}

export default App;
