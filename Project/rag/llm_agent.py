from api_setup import *
from openai import OpenAI
import google.generativeai as genai
import anthropic
import json


def apply_openai_agent(template, query, verbose: bool = True):
    result = []
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": template.format(query)}
            ],
        )
        result.append(chat_completion.choices[0].message.content)
    except Exception as e:
        if verbose:
            print(e)

    return result


def apply_gemini_agent(template: str, query: str, verbose: bool = True):
    # %%
    result = []
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(template.format(query))
        result.append(response.text)

    except Exception as e:
        if verbose:
            print(e)

    return result


def apply_anthropic_agent(template: str, query: str, verbose: bool = True):
    result = []
    try:
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
        )

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.0,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": template.format(query)}
            ]
        )

        result.append(message.content[0].text)
    except Exception as e:
        if verbose:
            print(e)

    return result


class Agent:
    def __init__(self, template: str, model: str = "google", verbose: bool = True):
        self.template = template
        self.model = model
        self.verbose = verbose

    def apply_agent(self, query: str):
        match self.model:
            case "google":
                return apply_gemini_agent(self.template, query, self.verbose)
            case "openai":
                return apply_openai_agent(self.template, query, self.verbose)
            case "anthropic":
                return apply_anthropic_agent(self.template, query, self.verbose)
            case _:
                raise NameError(f"Unknown model: {self.model}")


# class ExtractiveAgent(Agent):
#     # Extractive agent is used for supplemental keyword extraction. Input query is a (question) string.
#     def __init__(self, template: str, model: str = "google", verbose: bool = True):
#         super().__init__(template, model, verbose)
#
#
# class SummarizeAgent(Agent):
#     # Summarize agent directly summarizes the retrieved information.
#     def __init__(self, path, template):
#         super().__init__(template)
#         self.retrieved_text = json.load(open(path))
#
#     def apply_agent(self, query):
#         return
#
#
# class RerankAgent(Agent):
#     # Rerank agent can take retrieved text and performs reranking.
#     # To be added
#
#     def __init__(self, path, template: str, model: str = "google", verbose: bool = True):
#         super().__init__(template)
#         raise NotImplementedError
