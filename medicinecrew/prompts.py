triage_query = """Analyze this query and determine if it's medical-related:
        
{query}

Return:
- Category: DRUG_INTERACTION, MEDICATION_INFO, GENERAL_HEALTH, or NON_MEDICAL
- Reasoning: Why you chose this category"""

named_entity = """Extract all medicine/drug names from this query:
        
{query}

Look for brand names, generic names, and any medication references."""


drug_research = """Research information about these medications:
        
1. Find generic names for any brand medications
2. Find information about drug interactions
3. Find safety and contraindications info

Use web search to find reliable medical information."""


medical_advice = """Based on the user's query and research findings, provide medical advice:

User Query: {query}

Provide a comprehensive response that:
1. Answers the user's question
2. Includes relevant drug interaction warnings
3. Recommends consulting a healthcare professional
4. Notes any limitations in the information available"""


safety_verification = """Review this medical advice for safety concerns:

{advice}

Check for:
1. Dangerous drug interactions
2. Incorrect medical information
3. Missing safety warnings
4. Potentially harmful suggestions

Return: SAFE or UNSAFE with specific concerns."""
