import asyncio
from functools import lru_cache
from typing import IO, Any, overload
import numpy as np
import torch
from transformers import pipeline  # type: ignore
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore
import nest_asyncio  # type: ignore


SOURCES: list[str] = []


nest_asyncio.apply()  # type: ignore


@lru_cache(None)
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> tuple[str, ...]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return tuple(chunks)


def search(question: str, faiss_index: Any, chunk_pipe: Any) -> list[str]:
    encoded_question = np.array(chunk_pipe([question])).astype("float32")
    index = faiss_index.search(encoded_question, 5)[1][0]
    return [chunk_text("\n\n".join(SOURCES))[i] for i in index]


class AI:
    instances: list["AI"] = []

    def __init__(self) -> None:
        # This one does not support the latest version of transformers (see README.md)
        # self.pipe: Any = pipeline(
        #     "text-generation",
        #     model="deepseek-ai/DeepSeek-R1",
        #     trust_remote_code=True,
        #     device_map="auto"
        # )

        # self.pipe: Any = pipeline(
        #     "document-question-answering",
        #     model="impira/layoutlm-document-qa",
        #     trust_remote_code=True,
        #     device_map="auto"
        # )

        # self.pipe: Any = pipeline(
        #     "text2text-generation",
        #     model="google/flan-t5-large",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )

        self.question_pipe: Any = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        self.chunk_pipe: Any = SentenceTransformer("all-MiniLM-L6-v2").encode  # type: ignore

        AI.instances.append(self)

    async def run(self, question: str) -> str:
        # return self.pipe({
        #     "role": "user",
        #     "content": prompt
        # })[0]["generated_text"]

        sources_chunk = chunk_text("\n\n".join(SOURCES))

        encoded_sources = np.array(self.chunk_pipe(sources_chunk)).astype("float32")

        faiss_index = faiss.IndexFlatIP(encoded_sources.shape[1])
        faiss_index.add(encoded_sources)  # type: ignore

        sources = search(question, faiss_index, self.chunk_pipe)

        prompt = (
            "## PLEASE ANSWER THE QUESTION IN FRENCH USING THE SOURCES ##\n\n" +
            "NEW SOURCE:" + "\n\nNEW SOURCE:".join(sources) + "\n\nQUESTION:" + question
        )

        return self.question_pipe(prompt)[0]["generated_text"]

    async def __call__(self, question: str) -> str:
        return await self.run(question)

    @staticmethod
    async def get_response(question: str) -> str:
        threads: list[asyncio.Task[str]] = []
        for i in AI.instances:
            threads.append(asyncio.create_task(i(question)))
        done, pending = await asyncio.wait(threads, return_when=asyncio.FIRST_COMPLETED)
        res = "\n\n".join([e.result() for e in done])
        for t in pending:
            t.cancel()
        return res


def load(key: str = "") -> None:
    """Load the AI model
    @param key: str - The API key if required
    """
    AI()


def ask(question: str, print: IO[str] | None = None) -> str:
    """Ask the AI a question
    @param question: str - The question to ask
    @return str - The response
    """
    result = asyncio.run(AI.get_response(question))
    if print:
        print.write(result)
    return result


@overload
def add(source: list[str]) -> None:
    """Add a source to the AI informations
    @param source: list[str] - A list of sources to add
    """
    ...


@overload
def add(source: str) -> None:
    """Add a source to the AI informations
    @param source: str - The source to add
    """
    ...


@overload
def add(source: IO[str]) -> None:
    """Add a source to the AI informations
    @param source: IO[str] - The file to read the source to add from
    """
    ...


def add(source: str | list[str] | IO[str]) -> None:
    """Add a source to the AI informations
    @param source: str | list[str] - The source or list of sources to add
    """
    try:
        source = str(source.read())  # type: ignore
    except AttributeError:
        pass
    if isinstance(source, IO):  # is never true, present for linters
        source = source.read()
    if isinstance(source, list):
        SOURCES.extend(source)
        return
    SOURCES.append(source)


def unload() -> str:
    AI.instances.clear()
    return "AI model unloaded successfully"


__all__ = ["load", "ask", "add", "unload"]
