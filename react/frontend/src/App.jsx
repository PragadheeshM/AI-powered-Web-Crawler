import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleScrape = async () => {
    setLoading(true);
    try {
      await axios.post('http://localhost:5000/api/scrape', { url });
      alert("Website scraped and indexed successfully!");
    } catch (error) {
      console.error(error);
      alert("Failed to scrape website.");
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/chat', { message });
      
      // ✅ FIXED: access 'answer' not 'response'
      if (res.data.answer) {
        setResponse(res.data.answer);
      } else {
        setResponse("No answer returned.");
      }

    } catch (error) {
      console.error(error);
      setResponse("Error in chat response.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h2>RAG App - Website QA</h2>

      <input
        type="text"
        placeholder="Enter website URL"
        value={url}
        onChange={e => setUrl(e.target.value)}
        style={{ width: '400px', padding: '10px' }}
      />
      <button onClick={handleScrape} disabled={loading} style={{ marginLeft: '10px' }}>
        Scrape Website
      </button>

      <div style={{ marginTop: '30px' }}>
        <textarea
          rows={4}
          cols={50}
          placeholder="Ask your question..."
          value={message}
          onChange={e => setMessage(e.target.value)}
          style={{ padding: '10px' }}
        />
        <br />
        <button onClick={handleChat} disabled={loading}>Send</button>

        <div style={{ marginTop: '20px' }} id="responseBox">
          <strong>Answer:</strong>
          <p>{response}</p>
        </div>
      </div>
    </div>
  );
}

export default App;