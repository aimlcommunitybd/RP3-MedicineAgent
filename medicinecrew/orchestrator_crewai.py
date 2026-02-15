from crewai import Crew, Task, Process
from medicinecrew.agents.multi_agent import (
    create_triage_agent,
    create_ner_agent,
    create_drug_researcher_agent,
    create_expert_agent,
    create_safety_verifier_agent,
)
from medicinecrew import prompts as template


def run_medicine_agent(query: str) -> str:
    """Run the multi-agent medicine consultation system

    Flow:
    1. First, Triage Agent evaluates query (standalone)
       - Simple/greeting/off-topic → Answer directly
       - Complex medical → Continue to research pipeline
    2. If complex: Researcher → Expert → Safety
    3. Return final response
    """

    # Step 1: Run triage agent first to decide the flow
    triage_agent = create_triage_agent()

    triage_task = Task(
        description=f"""Analyze this user query and decide how to handle it:

User Query: {query}

Decision rules (MUST follow these exactly):
1. If query is ONLY a greeting (hi, hello, hey, good morning) → Answer directly
2. If query is off-topic (not about medicine/health/drugs) → Answer directly with polite redirect
3. If query asks about specific drug interactions, side effects, or combines multiple medications → DELEGATE to research
4. If query is a general medical question you can answer with your knowledge → Answer directly

IMPORTANT: For drug interaction questions (e.g., "what is interaction between X and Y"), you MUST delegate to the research agent. Do NOT try to answer drug interactions from memory.

Your response must start with either:
- "DELEGATE:" if delegating to research
- "ANSWER:" if answering directly""",
        agent=triage_agent,
        expected_output="DELEGATE: or ANSWER: followed by response",
    )

    # Run triage alone first
    triage_crew = Crew(
        agents=[triage_agent],
        tasks=[triage_task],
        verbose=False,
    )
    triage_crew.kickoff()

    triage_response = str(triage_task.output) if triage_task.output else ""
    triage_upper = triage_response.upper()

    print(f"DEBUG: Triage response: {triage_response[:200]}...")

    # Check if triage delegated or answered directly
    # If it starts with "DELEGATE", it's complex
    needs_delegation = triage_upper.startswith("DELEGATE")

    if not needs_delegation:
        # Simple query - triage answered directly
        # Remove "ANSWER:" prefix if present
        if triage_upper.startswith("ANSWER:"):
            return triage_response[7:].strip()
        return triage_response

    # Complex query - run full pipeline
    print("DEBUG: Delegating to research pipeline...")

    ner_agent = create_ner_agent()
    researcher_agent = create_drug_researcher_agent()
    expert_agent = create_expert_agent()
    safety_agent = create_safety_verifier_agent()

    ner_task = Task(
        description=template.named_entity.format(query=query),
        agent=ner_agent,
        expected_output="JSON list of medicine names found",
    )

    research_task = Task(
        description=template.drug_research,
        agent=researcher_agent,
        expected_output="Detailed drug information with citations [Source: ...]",
        context=[ner_task],
    )

    advice_task = Task(
        description=template.medical_advice.format(query=query),
        agent=expert_agent,
        expected_output="Medical advice with citations",
        context=[research_task],
    )

    safety_task = Task(
        description=template.safety_verification.format(
            advice=str(advice_task.output)
            if advice_task.output
            else "No advice generated"
        ),
        agent=safety_agent,
        expected_output="Safety verification with concerns if any",
    )

    # Run remaining pipeline
    pipeline_crew = Crew(
        agents=[ner_agent, researcher_agent, expert_agent, safety_agent],
        tasks=[ner_task, research_task, advice_task, safety_task],
        process=Process.sequential,
        verbose=True,
    )
    pipeline_crew.kickoff()

    return str(advice_task.output) if advice_task.output else "No response generated"



