# Alfred's Email Processing System 🤵📧

A sophisticated AI-powered email processing system featuring Alfred Pennyworth as Batman's loyal butler, designed to
analyze, categorize, and respond to emails with structured intelligence.

## 🌟 Features

- **Intelligent Email Analysis**: Spam detection, threat assessment, and priority classification
- **Structured Pydantic Outputs**: Type-safe data models for reliable processing
- **LangGraph Workflow**: Advanced state-based email processing pipeline
- **Batman Integration**: Special threat detection for Gotham City security
- **Comprehensive Logging**: Detailed processing statistics and error tracking
- **Fallback Systems**: Robust error handling with rule-based backups

## 🏗️ Architecture

### Core Components

- **EmailAgent**: Main orchestrator class managing the entire workflow
- **Pydantic Models**: Structured data validation for all outputs
- **LangGraph StateGraph**: Visual workflow management
- **Ollama LLM Integration**: Local language model processing
- **LangSmith Tracing**: Optional monitoring and debugging

### Data Models

```python
class EmailAnalysis(BaseModel):
    is_spam: bool
    spam_confidence: float
    spam_reason: Optional[str]
    category: EmailCategory
    priority: PriorityLevel
    requires_batman: bool
    tone_needed: ResponseTone
    key_points: List[str]


class ResponseDraft(BaseModel):
    draft_content: str
    estimated_reading_time: int
    key_actions_required: List[str]
    follow_up_needed: bool
    urgency_note: Optional[str]


class ThreatAssessment(BaseModel):
    threat_level: Literal["low", "medium", "high", "critical"]
    threat_type: List[str]
    immediate_actions: List[str]
    security_protocols: List[str]
    batman_involvement: bool
```

## 🚀 Installation

### Prerequisites

- Python 3.13
- Ollama running locally on port 11434
- Required Python packages

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:

```bash
pip install langgraph langchain-core langchain-ollama pydantic langsmith
```

3. **Start Ollama service**:

```bash
# Install Ollama if not already installed
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a compatible model (e.g., llama2, mistral, etc.)
ollama pull llama3.1
```

4. **Configure the Ollama utility**:
   Ensure your `utils/ollama.py` file is properly configured to connect to your Ollama instance.

## 📖 Usage

### Basic Usage

```python
from email_agent import EmailAgent

# Initialize the agent
agent = EmailAgent()

# Process an email
email = {
    "sender": "john.doe@example.com",
    "subject": "Meeting Request",
    "body": "Hi Bruce, can we schedule a meeting for tomorrow?"
}

result = agent.process_email(email)
```

### Email Processing Workflow

1. **Analysis Phase**: Email is analyzed for spam, threats, and categorization
2. **Routing Phase**: Based on analysis, email is routed to appropriate handler
3. **Processing Phase**:
    - **Spam**: Logged and filtered
    - **Threat**: Security assessment and Batman alert
    - **Legitimate**: Response drafting and action planning
4. **Summary Phase**: Comprehensive processing summary generated
5. **Notification Phase**: Structured briefing for Mr. Wayne

### Example Output

```
======================================================================
📧 ALFRED'S STRUCTURED EMAIL BRIEFING
======================================================================
From: Joker
Subject: Found you Batman!
Received: 2024-08-04T22:05:07.238000

📊 ANALYSIS RESULTS:
   Category: THREAT
   Priority: URGENT
   Spam Confidence: 0.10
   Key Points: Identity revelation threat, Revenge threat

🦇 BATMAN ALERT: Immediate attention required!

🛡️ THREAT ASSESSMENT:
   Level: CRITICAL
   Types: Identity compromise, Physical threat
   Actions: Activate security protocols, Alert Batman immediately

📋 PROCESSING SUMMARY:
   Steps Completed: 5
   Confidence: 0.90
   Recommendations: Review this email promptly, Consider Batman's involvement
   Next Actions: Secure identity, Increase security measures
======================================================================
```

## 🔧 Configuration

### Email Categories

- `THREAT`: Security threats requiring immediate attention
- `WORK`: Business-related correspondence
- `PERSONAL`: Personal messages
- `MARKETING`: Promotional content
- `SCAM`: Fraudulent emails
- `URGENT`: Time-sensitive matters
- `UNKNOWN`: Unclassified emails

### Priority Levels

- `LOW`: Standard processing
- `MEDIUM`: Regular attention
- `HIGH`: Elevated priority
- `URGENT`: Immediate action required

### Response Tones

- `FORMAL`: Professional business tone
- `FRIENDLY`: Warm personal tone
- `URGENT`: Direct emergency tone
- `DIPLOMATIC`: Careful political tone

## 📊 Monitoring & Statistics

The system tracks comprehensive statistics:

```python
stats = agent.get_stats_with_errors()
print(stats)
# Output:
# {
#     'total_processed': 10,
#     'spam_detected': 2,
#     'high_priority': 3,
#     'batman_alerts': 1,
#     'parsing_errors': 0,
#     'parsing_success_rate': 1.0
# }
```

## 🛡️ Error Handling

### Fallback Systems

- **Rule-based Analysis**: Keyword-based spam/threat detection
- **Template Responses**: Standard reply templates
- **Error Logging**: Comprehensive error tracking
- **Graceful Degradation**: System continues operating with reduced functionality

### Common Issues & Solutions

1. **Ollama Connection Issues**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart Ollama service
   ollama serve
   ```

2. **Model Loading Issues**:
   ```bash
   # Ensure model is available
   ollama list
   
   # Pull missing model
   ollama pull llama2
   ```

3. **Pydantic Validation Errors**:
    - Check that your LLM model supports structured output
    - Verify enum values match expected categories
    - Review field requirements in Pydantic models

## 🧪 Testing

### Test Emails Included

```python
test_emails = [
    {
        "sender": "Joker",
        "subject": "Found you Batman!",
        "body": "Identity threat message..."
    },
    {
        "sender": "spam@crypto.com",
        "subject": "🚀 Make $10,000 TODAY! 🚀",
        "body": "Investment scam message..."
    },
    {
        "sender": "Lucius Fox",
        "subject": "Wayne Enterprises Board Meeting",
        "body": "Legitimate business message..."
    }
]
```

### Running Tests

```bash
python email_agent.py
```

## 🔄 Workflow Visualization

```
START → Analyze Email → Route Decision
                          ├── SPAM → Handle Spam → END
                          ├── THREAT → Assess Threat → Create Summary → Notify Wayne → END  
                          └── LEGITIMATE → Draft Response → Create Summary → Notify Wayne → END
```

## 📚 Dependencies

- **langgraph**: Workflow orchestration
- **langchain-core**: LLM abstraction layer
- **pydantic**: Data validation and serialization
- **langsmith**: Optional tracing and monitoring
- **logging**: Built-in Python logging

## 🦇 About Alfred

*"A good butler always anticipates his master's needs. In Mr. Wayne's case, that includes managing correspondence that
might threaten Gotham City's safety."* - Alfred Pennyworth

---

## 🚨 Security Notice

This system is designed for educational and demonstration purposes. For production use:

- Implement proper authentication
- Add encryption for sensitive data
- Review and audit all LLM outputs
- Follow your organization's security policies

---
*Made with ❤️ for the Dark Knight*