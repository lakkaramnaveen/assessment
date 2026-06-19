import os
import json
from openai import OpenAI
from src.utils import get_schema

class TextToSQLAgent:
    def __init__(self, conn):
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY")
        )
        self.model = "accounts/fireworks/models/llama-v3p1-70b-instruct"
        self.conn = conn
        self.schema = get_schema(conn)
        self.chat_history = []
        
    def _get_system_prompt(self) -> str:
        schema_str = json.dumps(self.schema, indent=2)
        return (
            "You are an expert SQLite developer. Your job is to convert natural language questions into valid SQLite queries.\n"
            f"Here is the database schema:\n{schema_str}\n\n"
            "Rules:\n"
            "1. Output ONLY the raw SQL query inside a ```sql block. No explanations.\n"
            "2. Ensure you use the exact table and column names provided in the schema.\n"
            "3. If joining tables, use appropriate aliases.\n"
            "4. Limit your results to a reasonable number if not specified (e.g., LIMIT 10)."
        )

    def generate_sql(self, user_input: str, error_feedback: str = None) -> str:
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        
        # Include conversation history for context (follow-up questions)
        messages.extend(self.chat_history)
        
        if error_feedback:
            messages.append({"role": "user", "content": f"The previous query failed with error: {error_feedback}. Fix the SQL and return ONLY the corrected SQL block."})
        else:
            messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0 # Deterministic for SQL
        )
        
        raw_output = response.choices[0].message.content
        return self._extract_sql(raw_output)

    def _extract_sql(self, text: str) -> str:
        """Parses the SQL out of the markdown code block."""
        if "```sql" in text:
            return text.split("```sql")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def generate_summary(self, question: str, sql: str, raw_results: list) -> str:
        """Takes the raw data and makes it human-readable."""
        prompt = (
            f"Question: {question}\n"
            f"Executed SQL: {sql}\n"
            f"Raw Results: {raw_results}\n\n"
            "Provide a brief, human-readable summary of these results. Do not output the raw JSON."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = response.choices[0].message.content
        
        # Save to history for multi-turn conversations
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": summary})
        
        return summary