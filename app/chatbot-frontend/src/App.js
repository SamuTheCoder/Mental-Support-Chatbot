import logo from './logo.svg';
import './App.css';
import './CSS/styles.css';
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

    const handleKeyDown = (e) => {
        if(e.key === 'Enter') {
            sendMessage();
        }
    }

    return (
        <div className='body'>
            <div className='centered'>
                <div className='chatbot-title'>Mental Support Chatbot</div>
                <div className='chat-container'>
                    <div className='chat-messages-container'>
                        {messages.map((msg, index) => (
                            <div key={index} className='in-out-container'>
                                <p className='message user'><strong>User:</strong> {msg.user}</p>
                                <p className='message bot'><strong>Bot:</strong> {msg.bot}</p>
                            </div>
                        ))}
                    </div>
                    <div className='input-container'>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Type your message"
                        />
                        <button className='button' onClick={sendMessage}>Send</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
