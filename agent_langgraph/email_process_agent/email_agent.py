from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from utils.ollama import OllamaLLM
from langsmith import traceable
import logging
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enums for better type safety
class EmailCategory(str, Enum):
    THREAT = "threat"
    WORK = "work"
    PERSONAL = "personal"
    MARKETING = "marketing"
    SCAM = "scam"
    URGENT = "urgent"
    UNKNOWN = "unknown"

class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ResponseTone(str, Enum):
    FORMAL = "formal"
    FRIENDLY = "friendly"
    URGENT = "urgent"
    DIPLOMATIC = "diplomatic"

# Pydantic models for structured outputs
class EmailAnalysis(BaseModel):
    """Structured output for email analysis."""
    is_spam: bool = Field(description="Whether the email is spam")
    spam_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for spam detection")
    spam_reason: Optional[str] = Field(description="Reason if classified as spam")
    category: EmailCategory = Field(description="Email category classification")
    priority: PriorityLevel = Field(description="Priority level for Mr. Wayne's attention")
    requires_batman: bool = Field(description="Whether this requires Batman's immediate attention")
    tone_needed: ResponseTone = Field(description="Appropriate response tone")
    key_points: List[str] = Field(description="Key points from the email content")

    @field_validator('spam_reason')
    @classmethod
    def spam_reason_required_if_spam(cls, v, info):
        # Get the values from the validation context
        if hasattr(info, 'data') and info.data.get('is_spam') and not v:
            raise ValueError('spam_reason is required when is_spam is True')
        return v

class ResponseDraft(BaseModel):
    """Structured output for response drafting."""
    draft_content: str = Field(description="The drafted response content")
    estimated_reading_time: int = Field(description="Estimated reading time in seconds")
    key_actions_required: List[str] = Field(description="Actions Mr. Wayne might need to take")
    follow_up_needed: bool = Field(description="Whether follow-up is likely needed")
    urgency_note: Optional[str] = Field(description="Special urgency notes if any")

class ThreatAssessment(BaseModel):
    """Structured output for threat assessment."""
    threat_level: Literal["low", "medium", "high", "critical"] = Field(description="Assessed threat level")
    threat_type: List[str] = Field(description="Types of threats identified")
    immediate_actions: List[str] = Field(description="Immediate actions recommended")
    security_protocols: List[str] = Field(description="Security protocols to activate")
    batman_involvement: bool = Field(description="Whether Batman should be directly involved")

class ProcessingSummary(BaseModel):
    """Structured output for processing summary."""
    total_steps: int = Field(description="Total processing steps completed")
    processing_time_estimate: float = Field(description="Estimated processing time in seconds")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in processing")
    recommendations: List[str] = Field(description="Recommendations for Mr. Wayne")
    next_actions: List[str] = Field(description="Suggested next actions")

# Enhanced state with Pydantic integration
class EmailState(TypedDict):
    # Email content
    sender: str
    subject: str
    body: str
    received_at: str

    # Structured analysis results
    analysis: Optional[EmailAnalysis]
    draft_info: Optional[ResponseDraft]
    threat_assessment: Optional[ThreatAssessment]
    processing_summary: Optional[ProcessingSummary]

    # Legacy fields for compatibility
    is_spam: Optional[bool]
    email_draft: Optional[str]

    # Processing tracking
    messages: List[Dict[str, Any]]
    processing_steps: List[str]
    errors: List[str]

class EmailAgent:
    def __init__(self):
        self.model = OllamaLLM()
        self.llm = self.model.get_llm()
        self.processing_stats = {
            "total_processed": 0,
            "spam_detected": 0,
            "high_priority": 0,
            "batman_alerts": 0,
            "parsing_errors": 0
        }

        # Initialize structured output chains
        self.analysis_chain = self.llm.with_structured_output(EmailAnalysis)
        self.draft_chain = self.llm.with_structured_output(ResponseDraft)
        self.threat_chain = self.llm.with_structured_output(ThreatAssessment)
        self.summary_chain = self.llm.with_structured_output(ProcessingSummary)

        # Initialize tools after class is created
        self._setup_tools()

    def _setup_tools(self):
        """Setup tools with access to class instance."""

        @tool
        def analyze_email_structured(sender: str, subject: str, body: str) -> Dict[str, Any]:
            """Analyze email using structured Pydantic output."""
            analysis_prompt = f"""
As Alfred, Batman's loyal butler, analyze this email comprehensively:

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Consider:
1. Is this spam/scam? (provide confidence level 0.0-1.0)
2. What category does this fall into?
3. Priority level for Mr. Wayne's attention
4. Does this require Batman's immediate attention? (threats, emergencies)
5. What tone should the response take?
6. Extract key points from the email content
"""

            try:
                analysis = self.analysis_chain.invoke([
                    SystemMessage(content="You are Alfred Pennyworth, Batman's butler. Analyze emails professionally and thoroughly."),
                    HumanMessage(content=analysis_prompt)
                ])

                logger.info(f"✅ Successfully parsed email analysis: {analysis.category.value} - {analysis.priority.value} priority")

                return {
                    "analysis": analysis,
                    "success": True
                }

            except Exception as e:
                logger.error(f"❌ Failed to parse structured analysis: {e}")
                self.processing_stats["parsing_errors"] += 1
                return {
                    "analysis": self._fallback_analysis(sender, subject, body),
                    "success": False,
                    "error": str(e)
                }

        @tool
        def draft_response_structured(analysis_dict: Dict[str, Any], sender: str, subject: str, body: str) -> Dict[str, Any]:
            """Draft response using structured Pydantic output."""
            # Convert dict back to EmailAnalysis if needed
            if isinstance(analysis_dict, dict):
                try:
                    analysis = EmailAnalysis(**analysis_dict)
                except:
                    # If conversion fails, use the dict directly
                    analysis = analysis_dict
            else:
                analysis = analysis_dict

            # Get values safely
            category = getattr(analysis, 'category', analysis_dict.get('category', 'unknown'))
            priority = getattr(analysis, 'priority', analysis_dict.get('priority', 'medium'))
            tone_needed = getattr(analysis, 'tone_needed', analysis_dict.get('tone_needed', 'formal'))
            key_points = getattr(analysis, 'key_points', analysis_dict.get('key_points', []))

            draft_prompt = f"""
As Alfred the butler, draft a response to this email based on the analysis:

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Analysis Results:
- Category: {category}
- Priority: {priority}
- Tone needed: {tone_needed}
- Key points: {', '.join(key_points) if key_points else 'None'}

Draft a response and provide metadata about it including:
- The drafted response content
- Estimated reading time in seconds
- Key actions Mr. Wayne might need to take
- Whether follow-up is likely needed
- Any special urgency notes
"""

            try:
                draft_info = self.draft_chain.invoke([
                    SystemMessage(content="You are Alfred Pennyworth. Create a professional response and provide structured metadata."),
                    HumanMessage(content=draft_prompt)
                ])

                logger.info(f"✅ Successfully drafted {tone_needed} response ({draft_info.estimated_reading_time}s read time)")

                return {
                    "draft_info": draft_info,
                    "success": True
                }

            except Exception as e:
                logger.error(f"❌ Failed to parse structured draft: {e}")
                self.processing_stats["parsing_errors"] += 1
                return {
                    "draft_info": self._fallback_draft(analysis, sender),
                    "success": False,
                    "error": str(e)
                }

        @tool
        def assess_threat_structured(analysis_dict: Dict[str, Any], sender: str, body: str) -> Dict[str, Any]:
            """Assess threats using structured Pydantic output."""
            # Convert dict back to EmailAnalysis if needed
            if isinstance(analysis_dict, dict):
                try:
                    analysis = EmailAnalysis(**analysis_dict)
                except:
                    analysis = analysis_dict
            else:
                analysis = analysis_dict

            # Get values safely
            category = getattr(analysis, 'category', analysis_dict.get('category', 'unknown'))
            requires_batman = getattr(analysis, 'requires_batman', analysis_dict.get('requires_batman', False))

            threat_prompt = f"""
As Alfred, assess the threat level of this email:

From: {sender}
Body: {body}
Category: {category}
Batman Required: {requires_batman}

Provide a detailed threat assessment including:
- Threat level (low, medium, high, critical)
- Types of threats identified
- Immediate actions recommended
- Security protocols to activate
- Whether Batman should be directly involved
"""

            try:
                threat_assessment = self.threat_chain.invoke([
                    SystemMessage(content="You are Alfred Pennyworth, providing security threat assessment."),
                    HumanMessage(content=threat_prompt)
                ])

                logger.info(f"🛡️ Threat assessment: {threat_assessment.threat_level} level")

                return {
                    "threat_assessment": threat_assessment,
                    "success": True
                }

            except Exception as e:
                logger.error(f"❌ Failed to parse threat assessment: {e}")
                return {
                    "threat_assessment": self._fallback_threat_assessment(),
                    "success": False,
                    "error": str(e)
                }

        # Store tools as instance attributes
        self.analyze_email_tool = analyze_email_structured
        self.draft_response_tool = draft_response_structured
        self.assess_threat_tool = assess_threat_structured

    def _fallback_analysis(self, sender: str, subject: str, body: str) -> EmailAnalysis:
        """Fallback analysis using rule-based logic."""
        body_lower = body.lower()
        subject_lower = subject.lower()

        spam_keywords = ["investment", "crypto", "bitcoin", "make money", "click here", "winner", "urgent money"]
        threat_keywords = ["kill", "revenge", "destroy", "bomb", "threat", "hurt", "found you", "secret"]

        is_spam = any(keyword in body_lower + subject_lower for keyword in spam_keywords)
        is_threat = any(keyword in body_lower + subject_lower for keyword in threat_keywords)

        if is_threat:
            category = EmailCategory.THREAT
            priority = PriorityLevel.URGENT
            requires_batman = True
            tone = ResponseTone.URGENT
        elif is_spam:
            category = EmailCategory.SCAM
            priority = PriorityLevel.LOW
            requires_batman = False
            tone = ResponseTone.FORMAL
        else:
            category = EmailCategory.PERSONAL
            priority = PriorityLevel.MEDIUM
            requires_batman = False
            tone = ResponseTone.FRIENDLY

        return EmailAnalysis(
            is_spam=is_spam,
            spam_confidence=0.8 if is_spam else 0.2,
            spam_reason="Contains suspicious keywords" if is_spam else None,
            category=category,
            priority=priority,
            requires_batman=requires_batman,
            tone_needed=tone,
            key_points=["Fallback analysis used", f"Sender: {sender}"]
        )

    def _fallback_draft(self, analysis: Any, sender: str) -> ResponseDraft:
        """Fallback draft creation."""
        content = f"Dear {sender},\n\nThank you for your message. Mr. Wayne will review it accordingly.\n\nBest regards,\nAlfred Pennyworth"

        return ResponseDraft(
            draft_content=content,
            estimated_reading_time=10,
            key_actions_required=["Review message", "Decide on response"],
            follow_up_needed=False,
            urgency_note=None
        )

    def _fallback_threat_assessment(self) -> ThreatAssessment:
        """Fallback threat assessment."""
        return ThreatAssessment(
            threat_level="medium",
            threat_type=["Unknown"],
            immediate_actions=["Review manually"],
            security_protocols=["Standard monitoring"],
            batman_involvement=False
        )

    def analyze_email_node(self, state: EmailState) -> EmailState:
        """Node for structured email analysis."""
        # Use the tool properly with .invoke()
        result = self.analyze_email_tool.invoke({
            "sender": state["sender"],
            "subject": state["subject"],
            "body": state["body"]
        })

        processing_steps = state.get("processing_steps", [])
        processing_steps.append(f"Analyzed email using {'structured' if result['success'] else 'fallback'} parser")

        errors = state.get("errors", [])
        if not result["success"]:
            errors.append(f"Analysis parsing failed: {result.get('error', 'Unknown error')}")

        analysis = result["analysis"]

        return {
            **state,
            "analysis": analysis,
            "is_spam": analysis.is_spam,  # For legacy compatibility
            "processing_steps": processing_steps,
            "errors": errors
        }

    def draft_response_node(self, state: EmailState) -> EmailState:
        """Node for structured response drafting."""
        analysis = state["analysis"]

        # Convert Pydantic model to dict for tool input
        analysis_dict = analysis.model_dump() if hasattr(analysis, 'model_dump') else analysis.dict()

        result = self.draft_response_tool.invoke({
            "analysis_dict": analysis_dict,
            "sender": state["sender"],
            "subject": state["subject"],
            "body": state["body"]
        })

        processing_steps = state.get("processing_steps", [])
        processing_steps.append(f"Drafted response using {'structured' if result['success'] else 'fallback'} parser")

        errors = state.get("errors", [])
        if not result["success"]:
            errors.append(f"Draft parsing failed: {result.get('error', 'Unknown error')}")

        draft_info = result["draft_info"]

        return {
            **state,
            "draft_info": draft_info,
            "email_draft": draft_info.draft_content,  # For legacy compatibility
            "processing_steps": processing_steps,
            "errors": errors
        }

    def threat_assessment_node(self, state: EmailState) -> EmailState:
        """Node for structured threat assessment."""
        analysis = state["analysis"]

        # Convert Pydantic model to dict for tool input
        analysis_dict = analysis.model_dump() if hasattr(analysis, 'model_dump') else analysis.dict()

        result = self.assess_threat_tool.invoke({
            "analysis_dict": analysis_dict,
            "sender": state["sender"],
            "body": state["body"]
        })

        processing_steps = state.get("processing_steps", [])
        processing_steps.append(f"Assessed threats using {'structured' if result['success'] else 'fallback'} parser")

        errors = state.get("errors", [])
        if not result["success"]:
            errors.append(f"Threat assessment parsing failed: {result.get('error', 'Unknown error')}")

        return {
            **state,
            "threat_assessment": result["threat_assessment"],
            "processing_steps": processing_steps,
            "errors": errors
        }

    def handle_spam_node(self, state: EmailState) -> EmailState:
        """Handle spam with structured logging."""
        analysis = state["analysis"]

        logger.info(f"🚫 SPAM DETECTED")
        logger.info(f"   Sender: {state['sender']}")
        logger.info(f"   Confidence: {analysis.spam_confidence:.2f}")
        logger.info(f"   Reason: {analysis.spam_reason}")
        logger.info(f"   Category: {analysis.category.value}")

        self.processing_stats["spam_detected"] += 1

        processing_steps = state.get("processing_steps", [])
        processing_steps.append(f"Handled as spam (confidence: {analysis.spam_confidence:.2f})")

        return {
            **state,
            "processing_steps": processing_steps
        }

    def create_processing_summary_node(self, state: EmailState) -> EmailState:
        """Create final processing summary."""
        analysis = state["analysis"]
        draft_info = state.get("draft_info")
        threat_assessment = state.get("threat_assessment")

        # Calculate processing metrics
        total_steps = len(state.get("processing_steps", []))
        has_errors = len(state.get("errors", [])) > 0
        confidence = 0.9 if not has_errors else 0.6

        # Generate recommendations
        recommendations = []
        if analysis.priority in [PriorityLevel.HIGH, PriorityLevel.URGENT]:
            recommendations.append("Review this email promptly")
        if analysis.requires_batman:
            recommendations.append("Consider Batman's involvement")
        if draft_info and draft_info.follow_up_needed:
            recommendations.append("Schedule follow-up")

        # Generate next actions
        next_actions = []
        if draft_info:
            next_actions.extend(draft_info.key_actions_required)
        if threat_assessment and threat_assessment.batman_involvement:
            next_actions.extend(threat_assessment.immediate_actions)

        summary = ProcessingSummary(
            total_steps=total_steps,
            processing_time_estimate=2.5,  # Estimated
            confidence_score=confidence,
            recommendations=recommendations or ["Standard email handling"],
            next_actions=next_actions or ["Review and respond as appropriate"]
        )

        processing_steps = state.get("processing_steps", [])
        processing_steps.append("Generated processing summary")

        return {
            **state,
            "processing_summary": summary,
            "processing_steps": processing_steps
        }

    def notify_mr_wayne_structured(self, state: EmailState) -> EmailState:
        """Enhanced notification with structured data."""
        analysis = state["analysis"]
        draft_info = state.get("draft_info")
        threat_assessment = state.get("threat_assessment")
        summary = state.get("processing_summary")

        # Update stats
        self.processing_stats["total_processed"] += 1
        if analysis.priority in [PriorityLevel.HIGH, PriorityLevel.URGENT]:
            self.processing_stats["high_priority"] += 1
        if analysis.requires_batman:
            self.processing_stats["batman_alerts"] += 1

        # Display structured notification
        logger.info("\n" + "=" * 70)
        logger.info("📧 ALFRED'S STRUCTURED EMAIL BRIEFING")
        logger.info("=" * 70)

        # Basic info
        logger.info(f"From: {state['sender']}")
        logger.info(f"Subject: {state['subject']}")
        logger.info(f"Received: {state['received_at']}")

        # Analysis results
        logger.info(f"\n📊 ANALYSIS RESULTS:")
        logger.info(f"   Category: {analysis.category.value.upper()}")
        logger.info(f"   Priority: {analysis.priority.value.upper()}")
        logger.info(f"   Spam Confidence: {analysis.spam_confidence:.2f}")
        logger.info(f"   Key Points: {', '.join(analysis.key_points)}")

        if analysis.requires_batman:
            logger.info(f"🦇 BATMAN ALERT: Immediate attention required!")

        # Threat assessment
        if threat_assessment:
            logger.info(f"\n🛡️ THREAT ASSESSMENT:")
            logger.info(f"   Level: {threat_assessment.threat_level.upper()}")
            logger.info(f"   Types: {', '.join(threat_assessment.threat_type)}")
            logger.info(f"   Actions: {', '.join(threat_assessment.immediate_actions)}")

        # Draft info
        if draft_info:
            logger.info(f"\n✍️ RESPONSE DRAFT:")
            logger.info(f"   Reading Time: {draft_info.estimated_reading_time}s")
            logger.info(f"   Follow-up Needed: {'Yes' if draft_info.follow_up_needed else 'No'}")
            logger.info("-" * 50)
            logger.info(draft_info.draft_content)

        # Processing summary
        if summary:
            logger.info(f"\n📋 PROCESSING SUMMARY:")
            logger.info(f"   Steps Completed: {summary.total_steps}")
            logger.info(f"   Confidence: {summary.confidence_score:.2f}")
            logger.info(f"   Recommendations: {', '.join(summary.recommendations)}")
            logger.info(f"   Next Actions: {', '.join(summary.next_actions)}")

        # Errors if any
        errors = state.get("errors", [])
        if errors:
            logger.info(f"\n⚠️ PROCESSING ERRORS:")
            for error in errors:
                logger.info(f"   • {error}")

        logger.info("=" * 70 + "\n")

        return state

    def route_email_enhanced(self, state: EmailState) -> str:
        """Enhanced routing based on structured analysis."""
        analysis = state["analysis"]

        if analysis.is_spam:
            return "spam"
        elif analysis.requires_batman:
            return "threat"
        else:
            return "legitimate"

    def create_graph(self):
        """Create the structured workflow graph."""
        self.email_graph = StateGraph(EmailState)

        # Add nodes
        self.email_graph.add_node("analyze", self.analyze_email_node)
        self.email_graph.add_node("handle_spam", self.handle_spam_node)
        self.email_graph.add_node("draft_response", self.draft_response_node)
        self.email_graph.add_node("assess_threat", self.threat_assessment_node)
        self.email_graph.add_node("create_summary", self.create_processing_summary_node)
        self.email_graph.add_node("notify_wayne", self.notify_mr_wayne_structured)

        # Define workflow
        self.email_graph.add_edge(START, "analyze")
        self.email_graph.add_conditional_edges(
            "analyze",
            self.route_email_enhanced,
            {
                "spam": "handle_spam",
                "threat": "assess_threat",
                "legitimate": "draft_response"
            }
        )

        # Spam path
        self.email_graph.add_edge("handle_spam", END)

        # Threat path
        self.email_graph.add_edge("assess_threat", "create_summary")
        self.email_graph.add_edge("create_summary", "notify_wayne")

        # Legitimate path
        self.email_graph.add_edge("draft_response", "create_summary")

        # Final notification
        self.email_graph.add_edge("notify_wayne", END)

    @traceable
    def process_email(self, email: Dict[str, str]):
        """Process email with structured validation."""
        self.create_graph()
        compiled_graph = self.email_graph.compile()

        logger.info("🤵 Alfred initializing structured email processing...")

        # Initialize state
        state = EmailState(
            sender=email["sender"],
            subject=email["subject"],
            body=email["body"],
            received_at=datetime.now().isoformat(),
            analysis=None,
            draft_info=None,
            threat_assessment=None,
            processing_summary=None,
            is_spam=None,
            email_draft=None,
            messages=[],
            processing_steps=[f"Email received from {email['sender']}"],
            errors=[]
        )

        try:
            result = compiled_graph.invoke(state)
            logger.info("✅ Structured email processing completed successfully.")
            return result
        except Exception as e:
            logger.error(f"❌ Email processing failed: {e}")
            return state

    def get_stats_with_errors(self) -> Dict[str, Any]:
        """Get processing statistics including parsing errors."""
        stats = self.processing_stats.copy()
        stats["parsing_success_rate"] = 1.0 - (stats["parsing_errors"] / max(stats["total_processed"], 1))
        return stats

if __name__ == "__main__":
    agent = EmailAgent()

    # Test emails
    test_emails = [
        {
            "sender": "Joker",
            "subject": "Found you Batman!",
            "body": "Mr. Wayne, I found your secret identity! I know you're Batman! I'm coming for revenge. You can't hide from me!"
        },
        {
            "sender": "spam@crypto.com",
            "subject": "🚀 Make $10,000 TODAY! 🚀",
            "body": "Mr Wayne, invest in my new crypto coin! Guaranteed returns! Click here to become rich!"
        },
        {
            "sender": "Lucius Fox",
            "subject": "Wayne Enterprises Board Meeting",
            "body": "Bruce, the board meeting is scheduled for tomorrow at 2 PM. Please review the quarterly reports I sent earlier."
        }
    ]

    logger.info('🤵 Alfred Structured Email Processing System')
    logger.info('=' * 60)

    for i, email in enumerate(test_emails, 1):
        logger.info(f'\n📧 Processing Email {i}/{len(test_emails)}')
        agent.process_email(email)

    # Show final statistics
    stats = agent.get_stats_with_errors()
    logger.info('\n📊 FINAL PROCESSING STATISTICS:')
    for key, value in stats.items():
        logger.info(f"   {key.replace('_', ' ').title()}: {value}")