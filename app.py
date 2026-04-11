from datetime import datetime
from typing import Literal

from openai import AsyncOpenAI
from agents.tracing import set_tracing_disabled
from agents import OpenAIChatCompletionsModel
from agents import Agent, Runner, ModelSettings, function_tool
from pydantic import BaseModel, Field


set_tracing_disabled(True)

def get_ollama_model():
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    local_model = OpenAIChatCompletionsModel(
        model="minimax-m2.7:cloud",
        openai_client=client  
    )
    return local_model

MODEL = get_ollama_model()

class TicketClassification(BaseModel):
    """Structured output for ticket classification."""
    category: Literal["billing", "technical", "account", "general"] = Field(
        description="Which department handles this ticket"
    )
    priority: Literal["P1-critical", "P2-high", "P3-medium", "P4-low"] = Field(
        description="Ticket priority level"
    )
    sentiment: Literal["angry", "frustrated", "neutral", "positive"] = Field(
        description="Customer's emotional state"
    )
    summary: str = Field(
        description="One-line summary for the support dashboard"
    )

# SPECIAL AGENT
classifier_agent = Agent(
    name="Ticket Classifier",
    instructions="""
    You classify customer support messages.
    Analyze the message and return structured classification data.
    Be accurate — wrong classification wastes everyone's time.
    
    Priority guide:
    - P1-critical: Service down, data loss, security issue
    - P2-high: Feature broken, payment failure, angry customer
    - P3-medium: Bug report, how-to question, feature request
    - P4-low: General feedback, suggestions
    """,
    model=MODEL,
    output_type=TicketClassification,
    model_settings=ModelSettings(temperature=0.1) # concise result 
)

def support_instructions(context, agent):
    """Dynamic instructions that change based on current state."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    You are an AI customer support agent for "CloudSync" — a SaaS platform.
    Current time: {current_time}

    YOUR WORKFLOW (follow this order):
    1. Greet the customer professionally.
    2. Look up their account using lookup_customer if they provide an email.
    3. Understand their issue — search_knowledge_base for known solutions.
    4. If it's a service issue, check_service_status to see if it's a known outage.
    5. If you can solve it, provide the solution clearly.
    6. If you can't solve it, create_ticket to escalate.
    7. Always end with: "Is there anything else I can help with?"

    YOUR RULES:
    - Be empathetic, professional, and concise.
    - If the customer is angry, acknowledge their frustration first.
    - ALWAYS use tools to get real data — never make up information.
    - For billing issues on Enterprise plans, always escalate (create a ticket).
    - Include ticket ID when creating tickets.
    """



# MAIN AGENT 
support_agent = Agent(
    name="CloudSync Support",
    instructions=support_instructions,
    model=MODEL,
    model_setting=ModelSettings(
        temperature=0.3,
        max_tokens=1000,
    ), 
    tools=[
        classifier_agent.as_tool(
            tool_name="classify_ticket",
            tool_description="Classify a customer message into category, priority, and sentiment"
        ),
    ]
)

query = "HI"
result = Runner.run_sync(support_agent, query)
print(result.final_output)

