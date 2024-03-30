import os
import json
from typing import List, Any


class JsonWriter:

    def __init__(self, retriever_list: List[Any], path: str = "retrieved_text.json"):

        self.retrievers = retriever_list
        self.path = path

    def write(self, query: str):

        retrieved_dict = {"query": query}
        for r in self.retrievers:
            retrieved_dict[r.name] = r.retrieve(query)

        if os.path.exists(self.path):
            with open(self.path, "r+", encoding="utf-8") as file:
                file.seek(0, os.SEEK_END)
                pos = file.tell() - 1
                while pos > 0 and file.read(1) != "\n":
                    pos -= 1
                    file.seek(pos, os.SEEK_SET)
                if pos > 0:
                    file.seek(pos, os.SEEK_SET)
                    file.truncate()
                    file.write(',')
                else:
                    file.seek(0, os.SEEK_SET)
                json.dump([retrieved_dict], file, indent=4)
                file.write('\n]')
        else:
            with open(self.path, "w", encoding="utf-8") as file:
                json.dump([retrieved_dict], file, indent=4)
