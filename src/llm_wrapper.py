import os
from typing import Dict, List, Union

from dotenv import load_dotenv
from openai import OpenAI


class LLM:
    """Simple wrapper for LLM API calls"""

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def query(
        self, prompt: Union[str, List[Dict[str, str]]], temperature: float = 0.0
    ) -> str:
        """
        Make a basic API call to the LLM

        Args:
            prompt: Either a string prompt or a list of message dictionaries
            temperature: Controls randomness in the response

        Returns:
            The LLM's response text
        """
        try:
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": msg["role"], "content": msg["content"]} for msg in messages
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM API call: {str(e)}")
            raise
