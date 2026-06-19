import sqlite3
import sys
from src.utils import load_db, query_db
from src.agent import TextToSQLAgent

def main() -> None:
    print("Welcome to the Agentic BI CLI. Type 'exit' or 'quit' to end.")
    try:
        conn = load_db()
    except Exception as e:
        print(f"Failed to load database: {e}")
        sys.exit(1)

    agent = TextToSQLAgent(conn)

    while True:
        try:
            user_input = input("\nAsk a question: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue

            max_retries = 3
            sql_query = ""
            results = None
            error_msg = None

            for attempt in range(max_retries):
                print(f"Generating SQL (Attempt {attempt + 1})...")
                sql_query = agent.generate_sql(user_input, error_msg)
                print(f"Generated SQL:\n{sql_query}\n")

                try:
                    # return_as_df=False so we get a list of dicts for the LLM
                    results = query_db(conn, sql_query, return_as_df=False) 
                    break # Success! Exit the retry loop
                except sqlite3.Error as e:
                    error_msg = str(e)
                    print(f"Execution Error: {error_msg}. Retrying...")

            if results is None:
                print("Failed to generate valid SQL after maximum retries.")
                continue

            print("Summarizing results...")
            summary = agent.generate_summary(user_input, sql_query, results)
            print(f"\nResult:\n{summary}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    print("\nGoodbye!")

if __name__ == "__main__":
    main()