from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from medicinecrew.agents.openrouter import create_llm


def create_triage_agent() -> Agent:
    """Agent that determines if query is medical-relevant and routes appropriately"""
    llm = create_llm(temperature=0.2)

    return Agent(
        role="Medical Triage Specialist",
        goal="Determine if the user's query is related to medicine/healthcare and needs specialized handling",
        backstory="""You are a medical triage specialist. Your job is to:
        1. Identify if the query is medical/health-related
        2. Route non-medical queries to general handling
        3. Identify what type of medical assistance is needed
        
        Categories:
        - DRUG_INTERACTION: Questions about interactions between medications
        - MEDICATION_INFO: Questions about specific medications
        - GENERAL_HEALTH: General health questions
        - NON_MEDICAL: Not medical-related""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )


def create_ner_agent() -> Agent:
    """Agent that extracts medicine names from user queries"""
    llm = create_llm(temperature=0.2)

    return Agent(
        role="Medicine Name Extractor",
        goal="Extract all medicine/drug names from the user query including brand names and generics",
        backstory="""You specialize in identifying medication names in user queries.
        Look for:
        - Brand names (Napa500, Fymoxil, Panadol)
        - Generic names (Paracetamol, Amoxicillin)
        - Patterns with numbers (Napa500 = Paracetamol 500mg)
        - Common suffixes (-cilin, -ox, -ine, -ol)
        
        Return a JSON list of all medicine names found.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )


def create_drug_researcher_agent() -> Agent:
    """Agent that researches drug information"""
    llm = create_llm(temperature=0.2)

    # Web search tool
    search_tool = SerperDevTool()

    return Agent(
        role="Drug Information Researcher",
        goal="Find detailed information about medications including generic names, interactions, and safety",
        backstory="""You are a pharmaceutical researcher. Your job is to:
        1. Find generic names for brand medications
        2. Research drug interactions
        3. Gather safety information and contraindications
        4. Find reliable medical sources
        
        Use web search to find up-to-date drug information.""",
        verbose=True,
        llm=llm,
        tools=[search_tool],
        allow_delegation=True,
    )


def create_expert_agent() -> Agent:
    """Agent that provides expert medical advice"""
    llm = create_llm(temperature=0.2)

    return Agent(
        role="Medical Expert Doctor",
        goal="Provide accurate, evidence-based medical advice based on available information",
        backstory="""You are an expert physician. Your job is to:
        1. Analyze the user's medical question
        2. Use research findings to provide advice
        3. Include warnings about drug interactions
        4. Always recommend consulting healthcare professionals
        5. Be honest about limitations in knowledge
        
        Safety first - if unsure, say so.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )


def create_safety_verifier_agent() -> Agent:
    """Agent that verifies safety of medical advice"""
    llm = create_llm(temperature=0.1)  # Lower temperature for safety

    return Agent(
        role="Medical Safety Verifier",
        goal="Verify that medical advice is safe, accurate, and appropriate",
        backstory="""You are a medical safety specialist. Your job is to:
        1. Review expert advice for safety concerns
        2. Check for dangerous drug interactions
        3. Verify medical accuracy
        4. Flag any potentially harmful suggestions
        
        Be strict - patient safety is paramount.
        If advice has any safety issues, mark it for revision.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )
