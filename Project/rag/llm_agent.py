from .api_setup import GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
from openai import OpenAI
import google.generativeai as genai
import anthropic
from typing import List


def apply_gemini_agent(
        prompt: str,
        sys_info: str = "You are a helpful assistant.",
        verbose: bool = False) -> List[str]:
    result = []
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        result.append(response.text)
    except Exception as e:
        if verbose:
            print(e)

    return result


def apply_openai_agent(
        prompt: str,
        sys_info: str = "You are a helpful assistant.",
        verbose: bool = False) -> List[str]:
    result = []
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_info},
                {"role": "user", "content": prompt}
            ],
        )
        result.append(chat_completion.choices[0].message.content)
    except Exception as e:
        if verbose:
            print(e)

    return result


def apply_anthropic_agent(
        prompt: str,
        sys_info: str = "You are a helpful assistant.",
        verbose: bool = False) -> List[str]:
    result = []
    try:
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
        )
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.0,
            system=sys_info,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        result.append(message.content[0].text)
    except Exception as e:
        if verbose:
            print(e)

    return result


class LLMAgent:

    def __init__(self, model: str = "google", verbose: bool = False):
        self.model = model
        self.verbose = verbose

    def apply_agent(self, prompt: str, sys_info: str = "You are a helpful assistant.") -> List[str]:
        match self.model:
            case "google":
                responses = apply_gemini_agent(prompt=prompt, sys_info=sys_info, verbose=self.verbose)
            case "openai":
                responses = apply_openai_agent(prompt=prompt, sys_info=sys_info, verbose=self.verbose)
            case "anthropic":
                responses = apply_anthropic_agent(prompt=prompt, sys_info=sys_info, verbose=self.verbose)
            case _:
                raise NameError(f"Unknown model: {self.model}")

        return responses
