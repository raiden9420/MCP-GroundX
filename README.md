# MCP RAG Chatbot 🤖

A powerful **Model Context Protocol (MCP)** based **Retrieval-Augmented Generation (RAG)** chatbot that intelligently searches through your document knowledge base to provide accurate, context-aware responses. Built with GroundX for document storage and retrieval, and Google's Gemini AI for natural language generation.

## ✨ Features

- **🔍 Intelligent Document Search**: Leverages GroundX's powerful semantic search capabilities
- **🤖 AI-Powered Responses**: Uses Google Gemini 2.0 Flash for generating contextual answers
- **📚 Document Ingestion**: Easy document upload and processing (PDF, TXT, DOCX, etc.)
- **🌐 Web Interface**: Beautiful, responsive chat interface with real-time messaging
- **🔌 MCP Integration**: Standard Model Context Protocol implementation for tool interoperability
- **🚀 RESTful API**: Complete HTTP API for programmatic access
- **📊 Health Monitoring**: Built-in health checks and comprehensive logging
- **⚡ Real-time Processing**: Fast document search with context-aware responses

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Web Client    │◄───┤ Flask Server ├───►│   MCP Server    │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────┐    ┌─────────────────┐
                       │   Gemini AI  │    │    GroundX      │
                       │ (Text Gen.)  │    │ (Doc Storage)   │
                       └──────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GroundX API account
- Google AI Studio API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mcp-rag-chatbot.git
   cd mcp-rag-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure your `.env` file:**
   ```env
   GROUNDX_API_KEY=your_groundx_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Server

```bash
python newserver.py
```

The server will start with:
- **Web Interface**: http://localhost:8080
- **MCP Server**: Running on stdio
- **API Endpoints**: Available at base URL

## 📖 Usage

### Web Interface

1. Open http://localhost:8080 in your browser
2. Start asking questions about your documents
3. The system will automatically search for relevant context and provide informed answers


### MCP Integration

The server implements standard MCP tools that can be used by any MCP-compatible client:

- `search_doc_for_rag_context(query: str)` - Search documents for relevant context
- `ingest_documents(local_file_path: str, file_type: str)` - Add documents to knowledge base

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROUNDX_API_KEY` | GroundX API key (required) |
| `GEMINI_API_KEY` | Google AI Studio API key (required) |


## 📚 API Reference

### Endpoints

#### `GET /`
Serves the main chat web interface.

#### `POST /chat`
Main RAG endpoint for conversational queries.

**Request:**
```json
{
  "query": "Your question here"
}
```

**Response:**
```json
{
  "answer": "AI-generated response",
  "context_used": true,
  "query": "Your original question"
}
```

### Dependencies

- **fastmcp**: MCP server implementation
- **groundx**: Document storage and retrieval
- **google-generativeai**: Gemini AI integration
- **flask**: Web server framework
- **flask-cors**: CORS support
- **python-dotenv**: Environment variable management


## 🔍 How It Works

1. **Document Ingestion**: Upload documents to GroundX buckets for semantic indexing
2. **Query Processing**: User questions are processed and relevant context is retrieved
3. **Context Augmentation**: Retrieved context is combined with the user query
4. **AI Generation**: Gemini AI generates informed responses based on the context
5. **Response Delivery**: Contextual answers are delivered via web interface or API

## 🚨 Troubleshooting

### Common Issues

**"GROUNDX_API_KEY environment variable is not set"**
- Ensure your `.env` file contains the correct API key
- Verify the `.env` file is in the same directory as `newserver.py`

**"No relevant documents found"**
- Make sure documents have been ingested into your GroundX bucket
- Wait a few minutes after ingestion for documents to be indexed
- Try different query phrasings

**Connection errors**
- Check your internet connection
- Verify API keys are valid and have sufficient quotas

### Logs

The server provides comprehensive logging. Check the console output for detailed information about:
- Document search operations
- AI response generation
- API connectivity status
- Error details
