from typing import List


class DirectQAPrompts:

    def __init__(self):
        pass

    @staticmethod
    def question_answering(query: str) -> str:
        # Answer the question
        prompt_template = """
Question: "{query}"\nAnswer:
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt


class PreProcessingPrompts:

    def __init__(self):
        pass

    @staticmethod
    def keyword_extraction(query: str) -> str:
        # Keyword Extraction and Simplification
        prompt_template = """
Given the query: "{query}"\n\
Extract the main keywords from the above query. \
Simplify the query to its most essential components or keywords \
to aid in efficient information retrieval.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt

    @staticmethod
    def contextual_clarification(query: str) -> str:
        # Contextual Clarification
        prompt_template = """
Given the query: "{query}"\n\
Clarify the above query by rephrasing it into a more specific question or statement. \
Ensure the revised context is concise and directly related to the core topic \
for effective external information retrieval.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt

    @staticmethod
    def relevance_filtering(query: str) -> str:
        # Relevance Filtering
        prompt_template = """
Given the query: "{query}"\n\
Identify and remove any irrelevant details from the above query \
that may hinder the retrieval of focused information. \
Summarize the refined query to emphasize the most relevant aspects.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt

    @staticmethod
    def query_expansion(query: str) -> str:
        # Query Expansion
        prompt_template = """
Given the query: "{query}"\n\
Expand the above query by adding related terms or questions that \
might help in retrieving more comprehensive and relevant information from external sources.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt

    @staticmethod
    def information_structuring(query: str) -> str:
        # Information Structuring
        prompt_template = """
Given the query: "{query}"\n\
Structure the information within the above query into a clear and organized format. \
Categorize the details into themes or topics to facilitate targeted information retrieval.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt

    @staticmethod
    def intent_clarification(query: str) -> str:
        # Intent Clarification
        prompt_template = """
Given the query: "{query}"\n\
Clarify the intent behind the above query by rephrasing it into a more direct query. \
Highlight the main goal or the type of information sought to guide the retrieval process effectively.
        """.strip()
        prompt = prompt_template.format(query=query)
        return prompt


class PostProcessingPrompts:

    def __init__(self):
        pass

    @staticmethod
    def ranking_documents(query: str, docs: List[str]) -> str:
        # Ranking Documents Based on Relevance
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Rank these documents in order of their relevance to the original query. Provide the top 5 documents.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def summarizing_documents(query: str, docs: List[str]) -> str:
        # Summarizing Individual Documents
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Summarize the above documents by extracting its core message or information. \
Ensure the summary is concise and captures the essence of the document related to the original query.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

#     @staticmethod
#     def generating_summary(query: str, docs: List[str]) -> str:
#         # Generating a Consolidated Summary from Multiple Documents
#         prompt_template = """
# Given the original query: "{query}"\n\
# Give a list of retrieved documents as follows:\n{docs}\n\
# From the above documents, generate a comprehensive summary that integrates the most relevant information \
# from each document related to the original query. The summary should be coherent and concise.
#         """.strip()
#         docs_str = ""
#         for idx, doc in enumerate(docs, start=1):
#             docs_str += f"{idx}. {doc}\n"
#         prompt = prompt_template.format(query=query, docs=docs_str)
#         return prompt

    @staticmethod
    def extracting_key_info(query: str, docs: List[str]) -> str:
        # Extracting Key Information from Documents
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
From the above documents, extract the most critical pieces of information related to the original query. \
Organize the information by relevance and clarity.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def refining_documents(query: str, docs: List[str]) -> str:
        # Refining and Clarifying Documents
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Refine and clarify the content of the above documents to make it more directly related to the original query. \
Remove any irrelevant details and enhance the clarity of the document's main points.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def evaluating_documents(query: str, docs: List[str]) -> str:
        # Evaluating Document Quality and Relevance
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Evaluate the relevance and quality of the above documents in relation to the original query. \
Provide a brief assessment highlighting its relevance, accuracy, and any biases or inaccuracies detected.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def identifying_conflict(query: str, docs: List[str]) -> str:
        # Identifying and Highlighting Contradictions or Agreements
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Identify and highlight any agreements or contradictions among the above documents with respect to the original query. \
Summarize the points of agreement or conflict.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def filter_duplication(query: str, docs: List[str]) -> str:
        # Filtering Out Duplicate Information
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Identify and remove duplicate information found across the above documents. \
Provide a cleaned-up version of the content that retains unique information relevant to the original query.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def structured_format(query: str, docs: List[str]) -> str:
        # Transforming Document Content into a Structured Format
        prompt_template = """
Given the original query: "{query}"\n\
Give a list of retrieved documents as follows:\n{docs}\n\
Transform the key information found in the above documents into a structured format (e.g., bullet points, tables) \
to make the information more accessible and understandable in relation to the original query.
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            docs_str += f"{idx}. {doc}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt


class AugmentationPrompts:

    def __init__(self):
        # NOTE: Put the {query} to the end of the prompt (to fit the evaluation method)
        pass

    @staticmethod
    def augmentation_short(query: str, docs: List[str]) -> str:
        # Transforming Document Content into a Structured Format
        prompt_template = """
Give a list of retrieved documents as follows:\n{docs}\n\
Based on the above documents, generate an accurate, concise, and reasonable answer to the following query:\n\n{query}
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            if isinstance(doc, str) and len(doc.strip()) > 0:
                docs_str += f"{idx}. {doc.strip()}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def augmentation_medium(query: str, docs: List[str]) -> str:
        # Transforming Document Content into a Structured Format
        prompt_template = """
Give the relevant information extracted from external documents as follows:\n{docs}\n\
Using the key information from the above documents to create an accurate, concise, and reasonable response. \
Aim for coherence and insight, addressing the query with depth and clarity. \
Highlight any significant agreements or contradictions from the external information, ensuring a balanced view. \
Answer the following query:\n\n{query}
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            if isinstance(doc, str) and len(doc.strip()) > 0:
                docs_str += f"{idx}. {doc.strip()}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt

    @staticmethod
    def augmentation_long(query: str, docs: List[str]) -> str:
        prompt_template = """
Give the relevant information extracted from external documents as follows:\n{docs}\n\
Generate a comprehensive response that incorporates this information to provide \
an accurate, concise, and reasonable answer.\n\
The response should reflect an understanding of the query's intent and the knowledge contained \
within the processed documents. Ensure the generated content is coherent, logically structured, \
and seamlessly integrates the external information to enhance the quality and depth of the answer. \
If the processed information supports or contradicts the query, \
highlight these aspects appropriately, providing a balanced and informed perspective. \
Answer the following query:\n\n{query}
        """.strip()
        docs_str = ""
        for idx, doc in enumerate(docs, start=1):
            if isinstance(doc, str) and len(doc.strip()) > 0:
                docs_str += f"{idx}. {doc.strip()}\n"
        prompt = prompt_template.format(query=query, docs=docs_str)
        return prompt


class BackupPrompts:

    def __init__(self):
        self.PROMPT = """
Tell me some commonsense knowledge you can come up with related to the following context: {query}
        """.strip()

        self.fastRAG_PROMPT = """
Answer the question using the provided context. \
Your answer should be in your own words and be no longer than 50 words.\n\
Context: {join(documents)}\n\nQuestion: {query}\n\nAnswer:
        """.strip()

        self.general_PROMPT = """
DOCUMENT:
(document text)

QUESTION:
(users question)

INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return {NONE}
        """.strip()

        self.chatGPT_PROMPT_1 = """
Given the background information: {merge(information sources)}. \
What conclusions can we draw about {specific aspect}? \
Provide a concise response not exceeding 40 words.
        """.strip()

        self.chatGPT_PROMPT_2 = """
Using the details from: {concatenate(descriptions)}. \
Can you infer the potential outcomes for {particular scenario}? \
Keep your answer brief and under 60 words.
        """.strip()

        self.chatGPT_PROMPT_3 = """
Context provided: {aggregate(text snippets)}. \
What are the implicit assumptions regarding {specific element}? \
Summarize your insights in no more than 50 words.
        """.strip()

        self.chatGPT_PROMPT_4 = """
Considering the information: {combine(contents)}. \
How does {certain detail} impact the overall situation? \
Offer a succinct explanation within 45 words.
        """.strip()

        self.chatGPT_PROMPT_5 = """
Based on the narratives: {link(stories)}. \
What can be understood about {character or event}'s motivations? \
Craft a brief reply, limiting it to 50 words.
        """.strip()
