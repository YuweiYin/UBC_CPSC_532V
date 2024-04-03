from api_setup import *
from openai import OpenAI
import google.generativeai as genai
import anthropic
import json


class LLMAgent:

    def __init__(self, path, prompt_template):
        self.content = json.load(open(path))
        self.template = prompt_template

    def preprocess_content(self):
        raise NotImplementedError("The preprocessing method must be implemented in subclasses.")

    def query_builder(self):
        return self.template.format(query=self.preprocess_content())

    @staticmethod
    def apply_openai_agent(query: str, verbose: bool = False):
        result = []
        try:
            client = OpenAI(
                api_key=OPENAI_API_KEY,
            )
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
            )
            result.append(chat_completion.choices[0].message.content)
        except Exception as e:
            if verbose:
                print(e)

        return result

    @staticmethod
    def apply_gemini_agent(query: str, verbose: bool = False):
        # %%
        result = []
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(query)
            result.append(response.text)

        except Exception as e:
            if verbose:
                print(e)

        return result

    @staticmethod
    def apply_anthropic_agent(query: str, verbose: bool = False):
        result = []
        try:
            client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=ANTHROPIC_API_KEY,
            )

            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.0,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": query}
                ]
            )

            result.append(message.content[0].text)
        except Exception as e:
            if verbose:
                print(e)

        return result
