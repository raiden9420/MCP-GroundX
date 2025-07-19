import os
import logging
from dotenv import load_dotenv
from groundx import GroundX, Document
from mcp.server.fastmcp import FastMCP
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import json
import google.generativeai as genai
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MCP Server setup
mcp = FastMCP("eyelevel-rag")

# Load API keys from environment variables
groundx_api_key = os.getenv("GROUNDX_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
bucket_id = int(os.getenv("GROUNDX_BUCKET_ID", "19837"))

if groundx_api_key is None:
    raise ValueError("GROUNDX_API_KEY environment variable is not set.")

if gemini_api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Initialize GroundX client with error handling
try:
    groundx_client = GroundX(api_key=groundx_api_key)
    logger.info("GroundX client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GroundX client: {e}")
    raise

# Initialize Gemini with error handling
try:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise

# Flask app for HTTP endpoints
app = Flask(__name__)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080"])

# HTML template (embedded in server)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCP RAG Chatbot</title>
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    flex: 1;
    display: flex;
    flex-direction: column;
}

.header {
    text-align: center;
    color: white;
    margin-bottom: 30px;
    padding: 20px 0;
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    overflow: hidden;
    min-height: 600px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: #f8fafc;
    max-height: calc(100vh - 300px);
}

.message {
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    animation: fadeIn 0.3s ease-in;
}

.message.user {
    justify-content: flex-end;
}

.message.assistant {
    justify-content: flex-start;
}

.message-content {
    max-width: 70%;
    padding: 15px 20px;
    border-radius: 18px;
    word-wrap: break-word;
    position: relative;
    white-space: pre-wrap;
    line-height: 1.5;
}

.message.user .message-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.message.assistant .message-content {
    background: white;
    color: #333;
    border: 2px solid #e2e8f0;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin: 0 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    font-size: 14px;
    flex-shrink: 0;
}

.message.user .message-avatar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.message.assistant .message-avatar {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
}

.input-section {
    padding: 20px;
    background: white;
    border-top: 1px solid #e2e8f0;
}

.input-container {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

.input-field {
    flex: 1;
    padding: 15px 20px;
    border: 2px solid #e2e8f0;
    border-radius: 25px;
    font-size: 16px;
    resize: none;
    min-height: 50px;
    max-height: 150px;
    font-family: inherit;
    transition: border-color 0.3s ease;
}

.input-field:focus {
    outline: none;
    border-color: #667eea;
}

.send-button {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 50%;
    color: white;
    font-size: 18px;
    cursor: pointer;
    transition: transform 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.send-button:hover:not(:disabled) {
    transform: scale(1.05);
}

.send-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.loading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #666;
}

.loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #e2e8f0;
    border-top: 2px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.welcome-message {
    text-align: center;
    color: #666;
    padding: 40px 20px;
    font-size: 1.1rem;
    line-height: 1.6;
}

.welcome-message h3 {
    color: #333;
    margin-bottom: 15px;
}

.context-indicator {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
    font-style: italic;
    padding: 4px 8px;
    background: #f1f5f9;
    border-radius: 8px;
    display: inline-block;
}

.context-indicator.with-context {
    background: #dcfce7;
    color: #166534;
}

.context-indicator.no-context {
    background: #fef3cd;
    color: #92400e;
}

.error-message {
    color: #dc3545;
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    animation: fadeIn 0.3s ease-in;
}

.retry-button {
    background: #dc3545;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    margin-top: 8px;
    transition: background-color 0.2s ease;
}

.retry-button:hover {
    background: #c82333;
}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>🤖 MCP RAG Chatbot</h1>
    <p>Intelligent document search and question answering</p>
</div>

<div class="chat-container">
    <div class="chat-messages" id="chatMessages">
        <div class="welcome-message">
            <h3>Welcome to MCP RAG Chatbot!</h3>
            <p>I can search through your document knowledge base to answer questions.</p>
            <p>Ask me anything about the documents that have been ingested into the system.</p>
            <p><strong>Server Status:</strong> ✅ Connected and Ready</p>
        </div>
    </div>
    <div class="input-section">
        <div class="input-container">
            <textarea id="messageInput" class="input-field" placeholder="Ask a question about your documents..." rows="1"></textarea>
            <button id="sendButton" class="send-button" onclick="sendMessage()">
                <span>▶</span>
            </button>
        </div>
    </div>
</div>
</div>

<script>
const mcpServerUrl = window.location.origin;
let lastFailedMessage = null;

function addMessage(sender, content, isLoading = false, hasContext = undefined) {
    const messagesContainer = document.getElementById('chatMessages');
    
    const welcomeMessage = messagesContainer.querySelector('.welcome-message');
    if (welcomeMessage && sender === 'user') {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = sender === 'user' ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <span>Searching documents and generating response...</span>
            </div>
        `;
    } else {
        contentDiv.textContent = content;
        
        if (sender === 'assistant' && hasContext !== undefined) {
            const contextDiv = document.createElement('div');
            contextDiv.className = `context-indicator ${hasContext ? 'with-context' : 'no-context'}`;
            contextDiv.textContent = hasContext ? 
                '📚 Answer based on document search' : 
                '💭 No relevant documents found - general response';
            contentDiv.appendChild(contextDiv);
        }
    }
    
    if (sender === 'user') {
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(avatarDiv);
    } else {
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageDiv;
}

function addErrorMessage(message, canRetry = false, retryData = null) {
    const messagesContainer = document.getElementById('chatMessages');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    
    errorDiv.innerHTML = `
        <div>${message}</div>
        ${canRetry ? '<button class="retry-button" onclick="retryLastMessage()">Retry</button>' : ''}
    `;
    
    if (canRetry && retryData) {
        lastFailedMessage = retryData;
    }
    
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function retryLastMessage() {
    if (lastFailedMessage) {
        const input = document.getElementById('messageInput');
        input.value = lastFailedMessage;
        await sendMessage();
        lastFailedMessage = null;
    }
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) {
        input.focus();
        return;
    }
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    input.disabled = true;
    
    addMessage('user', message);
    input.value = '';
    
    const loadingMessage = addMessage('assistant', '', true);
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);
        
        const response = await fetch(`${mcpServerUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        loadingMessage.remove();
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            addErrorMessage(`❌ Server Error: ${result.error}`, true, message);
        } else {
            addMessage('assistant', result.answer, false, result.context_used);
        }
        
    } catch (error) {
        loadingMessage.remove();
        
        let errorMessage = '❌ Error: ';
        if (error.name === 'AbortError') {
            errorMessage += 'Request timed out. Please try again.';
        } else {
            errorMessage += error.message;
        }
        
        addErrorMessage(errorMessage, true, message);
        console.error('Error:', error);
    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

function adjustTextareaHeight() {
    const textarea = document.getElementById('messageInput');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

document.getElementById('messageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

document.getElementById('messageInput').addEventListener('input', adjustTextareaHeight);

window.addEventListener('load', () => {
    document.getElementById('messageInput').focus();
});
</script>
</body>
</html>'''

@mcp.tool()
def search_doc_for_rag_context(query: str) -> str:
    """
    Searches and retrieves relevant context from a knowledge base,
    based on the user's query.
    Args:
        query: The search query supplied by the user.
    Returns:
        str: Relevant text content that can be used by the LLM to answer the query.
    """
    try:
        logger.info(f"Searching documents with query: {query}")
        response = groundx_client.search.content(
            id=bucket_id,
            query=query,
            n=10,
        )
        
        search_text = response.search.text if hasattr(response.search, 'text') else ""
        logger.info(f"Search returned {len(search_text)} characters")
        return search_text
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return ""

@mcp.tool()
def ingest_documents(local_file_path: str, file_type: str = "pdf") -> str:
    """
    Ingest documents from a local file into the knowledge base.
    Args:
        local_file_path: The path to the local file containing the documents to ingest.
        file_type: The type of file (pdf, txt, docx, etc.)
    Returns:
        str: A message indicating the documents have been ingested.
    """
    try:
        if not os.path.exists(local_file_path):
            return f"Error: File not found at {local_file_path}"
        
        file_name = os.path.basename(local_file_path)
        logger.info(f"Ingesting file: {file_name}")
        
        document = Document(
            bucket_id=bucket_id,
            file_name=file_name,
            file_path=local_file_path,
            file_type=file_type.lower(),
            search_data={"uploaded_at": str(int(time.time()))}
        )
        
        result = groundx_client.ingest(documents=[document])
        
        logger.info(f"Successfully ingested {file_name}")
        return f"Successfully ingested {file_name} into the knowledge base. It should be available for search in a few minutes."
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        return f"Error ingesting document: {str(e)}"

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint that processes user queries using RAG
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Processing chat query: {query}")
        
        # Search for relevant context
        context = search_doc_for_rag_context(query)
        
        # Generate response using Gemini with context
        if context.strip():
            prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

User question: {query}

Please provide a helpful and accurate answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can based on what's available. Be specific and cite relevant information from the context when possible."""
            
            context_used = True
            logger.info("Using context from document search")
        else:
            # No context found, provide a general response
            prompt = f"""You are a helpful assistant. The user asked: {query}

No relevant documents were found in the knowledge base for this question. Please provide a helpful general response and suggest that the user might want to ask about topics that are covered in the document knowledge base, or they might need to ingest relevant documents first."""
            
            context_used = False
            logger.info("No context found, providing general response")
        
        # Generate response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                    )
                )
                
                if response.text:
                    answer = response.text
                    break
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Brief delay before retry
        
        logger.info("Successfully generated response")
        
        return jsonify({
            "answer": answer,
            "context_used": context_used,
            "query": query
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/tools/search_doc_for_rag_context', methods=['POST'])
def http_search_doc_for_rag_context():
    """HTTP endpoint for document search"""
    try:
        data = request.get_json()
        if not data or 'arguments' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        query = data['arguments'].get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        result = search_doc_for_rag_context(query)
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/tools/ingest_documents', methods=['POST'])
def http_ingest_documents():
    """HTTP endpoint for document ingestion"""
    try:
        data = request.get_json()
        if not data or 'arguments' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        local_file_path = data['arguments'].get('local_file_path')
        file_type = data['arguments'].get('file_type', 'pdf')
        
        if not local_file_path:
            return jsonify({"error": "No file path provided"}), 400
            
        result = ingest_documents(local_file_path, file_type)
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test GroundX connection
        test_query = "test"
        groundx_client.search.content(id=bucket_id, query=test_query, n=1)
        
        # Test Gemini connection
        model.generate_content("Hello", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
        
        return jsonify({
            "status": "healthy",
            "groundx": "connected",
            "gemini": "connected",
            "bucket_id": bucket_id
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

def run_flask():
    """Run Flask server with proper error handling"""
    try:
        app.run(
            host='0.0.0.0', 
            port=8080, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting MCP RAG Server...")
        
        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True  # Make thread daemon so it dies when main thread dies
        flask_thread.start()
        
        # Give Flask a moment to start
        time.sleep(2)
        
        logger.info("Web Interface available at: http://localhost:8080")
        logger.info("API endpoints:")
        logger.info("- GET / (main chat interface)")
        logger.info("- POST /chat (main chat endpoint for RAG)")
        logger.info("- POST /tools/search_doc_for_rag_context")
        logger.info("- POST /tools/ingest_documents")
        logger.info("- GET /health")
        logger.info(f"Using GroundX bucket ID: {bucket_id}")
        logger.info("Starting MCP server on stdio...")
        
        # Run MCP server on main thread
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise