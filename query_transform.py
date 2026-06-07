from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def rewrite_query(query):
    prompt = f"""
    Rewrite the following user query to make it more clear, specific and suitable for document retrieval.

    Rules:
     - Keep the original intent
     - Expand vague queries
     - Add context if missing
     - Do Not answer the question
     - Only rewrite the query

    User Query: {query}

    Rewritten Query:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()