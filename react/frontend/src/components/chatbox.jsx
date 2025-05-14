import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function ChatBox() {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  useEffect(() => {
    // Optional: Load old chat history from backend
    axios.get('http://localhost:5000/api/history')
      .then(res => {
        const msgs = res.data.flatMap(item => [
          { sender: 'user', text: item.user },
          { sender: 'bot', text: item.bot }
        ]);
        setChatHistory(msgs);
      });
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      const res = await axios.post('http://localhost:5000/api/chat', { message: input });
      const botMessage = { sender: 'bot', text: res.data.answer || "No response." };
      setChatHistory(prev => [...prev, botMessage]);
    } catch {
      setChatHistory(prev => [...prev, { sender: 'bot', text: "Error: Unable to get response." }]);
    }

    setInput('');
  };

  return (
    <div style={{ padding: 20, maxWidth: 600, margin: "auto" }}>
      <h3>Chat with AI</h3>

      <div style={{
        border: '1px solid #ccc',
        borderRadius: 8,
        height: 300,
        overflowY: 'auto',
        padding: 10,
        background: '#f9f9f9',
        marginBottom: 10
      }}>
        {chatHistory.map((msg, index) => (
          <div
            key={index}
            style={{
              textAlign: msg.sender === 'user' ? 'right' : 'left',
              margin: '10px 0',
            }}
          >
            <span
              style={{
                display: 'inline-block',
                backgroundColor: msg.sender === 'user' ? '#dcf8c6' : '#fff',
                padding: '10px 14px',
                borderRadius: 12,
                maxWidth: '80%',
                wordWrap: 'break-word'
              }}
            >
              {msg.text}
            </span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type your message..."
          style={{ flex: 1, padding: '10px', borderRadius: 6, border: '1px solid #ccc' }}
        />
        <button onClick={handleSend} style={{ marginLeft: 8, padding: '10px 15px' }}>
          ➤
        </button>
      </div>
    </div>
  );
}
