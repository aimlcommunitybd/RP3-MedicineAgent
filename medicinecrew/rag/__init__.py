import os
from crewai.tools import BaseTool
from pydantic import Field
import pandas as pd


class LocalDrugInteractionTool(BaseTool):
    name: str = "Local Drug Interaction Database"
    description: str = "Search a local database of drug-drug interactions. Returns interaction descriptions between medication pairs."

    def _run(self, query: str) -> str:
        """Search local drug interaction database"""
        try:
            # Load the dataset
            df = pd.read_csv("dataset/db_drug_interactions.csv")

            # Normalize query for matching
            query_upper = query.upper()
            query_lower = query.lower()

            # Search in both columns
            mask = (
                df["Drug 1"].str.lower().str.contains(query_lower, na=False)
                | df["Drug 2"].str.lower().str.contains(query_lower, na=False)
                | df["Drug 1"].str.upper().str.contains(query_upper, na=False)
                | df["Drug 2"].str.upper().str.contains(query_upper, na=False)
            )

            results = df[mask]

            if results.empty:
                return f"No interactions found in local database for: {query}"

            # Return top 5 results
            output = f"Found {len(results)} interactions in local database:\n\n"
            for _, row in results.head(5).iterrows():
                output += f"- {row['Drug 1']} + {row['Drug 2']}: {row['Interaction Description']}\n"

            return output

        except Exception as e:
            return f"Error searching local database: {str(e)}"


# Create the tool instance
local_drug_tool = LocalDrugInteractionTool()
