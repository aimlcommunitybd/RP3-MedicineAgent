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
    """Run the multi-agent medicine consultation system"""

    # Create agents
    triage_agent = create_triage_agent()
    ner_agent = create_ner_agent()
    researcher_agent = create_drug_researcher_agent()
    expert_agent = create_expert_agent()
    safety_agent = create_safety_verifier_agent()

    # Create tasks
    triage_task = Task(
        description=template.triage_query.format(query=query),
        agent=triage_agent,
        expected_output="Medical category classification with reasoning",
    )

    ner_task = Task(
        description=template.named_entity.format(query=query),
        agent=ner_agent,
        expected_output="JSON list of medicine names found",
    )

    research_task = Task(
        description=template.drug_research,
        agent=researcher_agent,
        expected_output="Detailed drug information including interactions",
        context=[ner_task],
    )

    advice_task = Task(
        description=template.medical_advice.format(query=query),
        agent=expert_agent,
        expected_output="Medical advice response",
        context=[research_task],
    )

    safety_task = Task(
        description=template.safety_verification.format(
            advice=(
                advice_task.output if advice_task.output else "No advice generated yet"
            )
        ),
        agent=safety_agent,
        expected_output="Safety verification with concerns if any",
    )

    # Create crew with sequential process
    crew = Crew(
        agents=[triage_agent, ner_agent, researcher_agent, expert_agent, safety_agent],
        tasks=[triage_task, ner_task, research_task, advice_task, safety_task],
        process=Process.sequential,
        verbose=True,
    )

    # Run the crew
    crew.kickoff()

    # Return the expert's advice (not the safety verification)
    return str(advice_task.output) if advice_task.output else "No response generated"


if __name__ == "__main__":
    # Test
    result = run_medicine_agent(
        "What is the interaction between Napa500 and Fymoxil500?"
    )
    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)
    print(result)
