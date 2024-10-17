
import React, { useState } from 'react';

const LLNInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState('en');
  const [mode, setMode] = useState('search'); // Default mode is 'search'

  const fetchData = async (input, lang, mode) => {
    const response = await fetch('http://127.0.0.1:8000/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: input,
        language: lang,
        mode: mode,
      }),
    });
    const data = await response.json();
    console.log(data);
    return data;
  };

  const handleSendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { sender: 'User', text: input }]);
      setInput('');
      fetchData(input, language, mode).then((responseData) => {
        setTimeout(() => {
          setMessages((prevMessages) => [
            ...prevMessages,
            { sender: 'LLM', text: responseData.response },
          ]);
        }, 1000);
      });
    }
  };

  return (
    <div className="flex flex-col max-h-[80vh] bg-white p-4 shadow-lg rounded-lg">
      <div className="sticky top-0 bg-gradient-to-r from-pink-500 to-purple-500 p-2 text-center font-bold text-xl text-white rounded-md">
        Genpatch AI Assistant
      </div>
      <div className="flex-1 max-h-[80%] overflow-y-scroll bg-gray-100 p-4 shadow-md rounded-lg mb-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`my-2 p-2 rounded-md ${
              msg.sender === 'User' ? 'bg-blue-200 self-end' : 'bg-gray-300 self-start'
            }`}
          >
            <strong>{msg.sender}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <div className="flex">
        <input
          type="text"
          className="flex-1 p-2 bg-white border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-purple-500"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleSendMessage();
            }
          }}
        />
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="p-2 bg-white border border-gray-300 rounded-md mx-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="fr">French</option>
          <option value="es">Spanish</option>
        </select>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="p-2 bg-white border border-gray-300 rounded-md mx-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option value="search">Search</option>
          <option value="summarize">Summarize</option>
          <option value="translate">Translate</option> {/* New translate option */}
        </select>
        <button
          className="p-2 bg-purple-500 text-white rounded-r-md hover:bg-purple-600"
          onClick={handleSendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default LLNInterface;
