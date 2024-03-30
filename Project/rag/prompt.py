PROMPT = """\
Tell me some commonsense knowledge you can come up with related to the following context: {query}
"""

fastRAG_PROMPT = """\
\"Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \\n\\n Context: {join(documents)} \\n\\n Question: {query} \\n\\n Answer:\",\n"
"""

general_PROMPT = """\
DOCUMENT:
(document text)

QUESTION:
(users question)

INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return {NONE}
"""

chatGPT_PROMPT_1 = """\
Given the background information: {merge(information sources)}. What conclusions can we draw about {specific aspect}? Provide a concise response not exceeding 40 words.
"""

chatGPT_PROMPT_2 = """\
Using the details from: {concatenate(descriptions)}. Can you infer the potential outcomes for {particular scenario}? Keep your answer brief and under 60 words.
"""

chatGPT_PROMPT_3 = """\
Context provided: {aggregate(text snippets)}. What are the implicit assumptions regarding {specific element}? Summarize your insights in no more than 50 words.
"""

chatGPT_PROMPT_4 = """\
Considering the information: {combine(contents)}. How does {certain detail} impact the overall situation? Offer a succinct explanation within 45 words.
"""

chatGPT_PROMPT_5 = """\
Based on the narratives: {link(stories)}. What can be understood about {character or event}'s motivations? Craft a brief reply, limiting it to 50 words.
"""
