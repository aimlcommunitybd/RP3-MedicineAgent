from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool
from medicinecrew.engines.openrouter import create_llm
from medicinecrew.rag import local_drug_tool


def create_triage_agent() -> Agent:
    """Agent that handles simple queries or routes complex ones to other agents"""
    llm = create_llm(temperature=0.2)

    # Web search for quick lookups
    search_tool = SerperDevTool()

    return Agent(
        role="Medical Triage Specialist",
        goal="""Handle simple greetings/off-topic queries directly. 
        Route complex medical queries to specialist agents.
        
        Decision logic:
        - GREETING: "hi", "hello", "hey", "good morning" → Reply warmly and ask how to help
        - OFF_TOPIC: Questions about non-medical topics → Politely redirect to medical focus
        - SIMPLE_MEDICAL: Basic medication questions answerable from knowledge → Answer directly
        - COMPLEX: Drug interactions, multiple medications, detailed questions → Delegate to research""",
        backstory="""You are a medical triage specialist. Your job is to:
        1. Identify if the query is simple (greeting/off-topic) → respond directly
        2. Identify if it's a complex medical query → delegate to research agent
        3. For complex queries, extract medicine names first
        
        You have access to search tools to answer simple medication questions directly.""",
        verbose=True,
        llm=llm,
        tools=[search_tool],
        allow_delegation=True,  # Can delegate to other agents
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
        tools=[local_drug_tool],  # Can check local DB
        allow_delegation=False,
    )


def create_drug_researcher_agent() -> Agent:
    """Agent that researches drug information using both local RAG and web search"""
    llm = create_llm(temperature=0.2)

    # Both web search AND local RAG
    web_search = SerperDevTool()

    return Agent(
        role="Drug Information Researcher",
        goal="Find detailed drug information with EVIDENCE and CITATIONS. Use BOTH local database AND web search.",
        backstory="""You are a pharmaceutical researcher. Your job is to:
        1. Search local drug interaction database first
        2. Search web for additional/updated information
        3. ALWAYS cite sources in your response
        4. Prioritize reliable medical sources
        
        For EVERY piece of information, indicate the source:
        - [Source: Local DB] for local database matches
        - [Source: Web - URL] for web search results
        
        This ensures evidence-based responses.""",
        verbose=True,
        llm=llm,
        tools=[local_drug_tool, web_search],  # BOTH sources!
        allow_delegation=True,
    )


def create_expert_agent() -> Agent:
    """Agent that provides expert medical advice with access to all tools"""
    llm = create_llm(temperature=0.2)

    # Both web and local for expert too
    web_search = SerperDevTool()

    return Agent(
        role="Medical Expert Doctor",
        goal="Provide accurate, evidence-based medical advice with CITATIONS. Use all available tools.",
        backstory="""You are an expert physician. Your job is to:
        1. Analyze the user's medical question
        2. Check local drug database for known interactions
        3. Use web search for latest information
        4. Provide advice WITH CITATIONS to sources
        5. Include warnings about drug interactions
        6. Always recommend consulting healthcare professionals
        7. Be honest about limitations in knowledge
        
        ALWAYS cite your sources:
        - [Source: Local DB] for local database
        - [Source: Web - URL] for web sources
        
        Safety first - if unsure, say so.""",
        verbose=True,
        llm=llm,
        tools=[local_drug_tool, web_search],  # Expert also has both!
        allow_delegation=False,
    )


def create_safety_verifier_agent() -> Agent:
    """Agent that verifies safety of medical advice"""
    llm = create_llm(temperature=0.1)  # Lower temperature for safety

    return Agent(
        role="Medical Safety Verifier",
        goal="Verify that medical advice is safe, accurate, appropriate, and properly cited",
        backstory="""You are a medical safety specialist. Your job is to:
        1. Review expert advice for safety concerns
        2. Check for dangerous drug interactions
        3. Verify medical accuracy
        4. Check that sources are properly cited
        5. Flag any potentially harmful suggestions
        
        Be strict - patient safety is paramount.
        
        Verify citations are present - every medical claim should have a source.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )
