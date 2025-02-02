import asyncio
from functools import lru_cache
from os import PathLike
import openai.pagination
import pytesseract  # type: ignore
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path  # type: ignore
from typing import IO, Any, overload
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore
import nest_asyncio  # type: ignore
import openai


SOURCES: list[str] = []


nest_asyncio.apply()  # type: ignore


@lru_cache(None)
def chunk_text(text: str) -> tuple[str, ...]:
    words = text.split(".\n\n")
    return tuple(words)


def search(question: str, faiss_index: Any, chunk_pipe: Any) -> list[str]:
    encoded_question = np.array(chunk_pipe([question])).astype("float32")
    index = faiss_index.search(encoded_question, 5)[1][0]
    return [chunk_text("\n\n".join(SOURCES))[i] for i in index]


def parse(text: str) -> str:
    text = text.replace("\\t", "\t")
    text = text.replace("\\n", "\n")
    text = text.replace("\\r", "\r")
    text = text.replace("\\'", "'")
    text = text.replace('\\"', '"')
    text = text.replace("\\\\", "\\")
    return text


def copilot_generate(prompt: str, client: openai.Client, assistant_id: str) -> str:
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(  # type: ignore
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    while run.status != "completed":
        pass
    response: openai.pagination.SyncCursorPage[Any] = client.beta.threads.messages.list(
        thread_id=thread.id,
    )
    return response.data[0].content[0]


class AI:
    instances: list["AI"] = []

    def __init__(self, key_file: PathLike[str] | str | None = None) -> None:
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

        # self.question_pipe: Any = pipeline(
        #     "text2text-generation",
        #     model="google/flan-t5-large",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )omni-research/Tarsier-7b

        # self.question_pipe: Any = pipeline(
        #     "text2text-generation",
        #     model="google/flan-t5-base",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )

        # self.question_pipe: Any = pipeline(
        #     "text2text-generation",
        #     model="teapotai/teapotllm",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )

        if key_file is None:
            key_file = Path("openai.key")

        with open(key_file) as f:
            client = openai.OpenAI(
                api_key=f.read().strip()
            )

        assistant = client.beta.assistants.create(  # type: ignore
            model="gpt-4-turbo",
            instructions="You are an AI assistant answering questions in french. You do not have internet access",
            tools=[{'type': 'file_search'}],
            name="AI Assistant",
        )

        def generate(prompt: str) -> list[dict[str, str]]:
            return [{"generated_text": copilot_generate(prompt, client, assistant.id)}]

        self.question_pipe: Any = generate

        self.chunk_pipe: Any = SentenceTransformer("all-MiniLM-L6-v2").encode  # type: ignore

        AI.instances.append(self)

    async def run(self, question: str, ignore_filtering: bool = False) -> str:
        # return self.pipe({
        #     "role": "user",
        #     "content": prompt
        # })[0]["generated_text"]

        if not ignore_filtering:
            sources_chunk = chunk_text("\n\n".join(SOURCES))

            encoded_sources = np.array(self.chunk_pipe(sources_chunk)).astype("float32")

            faiss_index = faiss.IndexFlatIP(encoded_sources.shape[1])
            faiss_index.add(encoded_sources)  # type: ignore

            sources = search(question, faiss_index, self.chunk_pipe)
        else:
            sources = SOURCES

        prompt = (
            # "## please answer the question in french using the sources ##\n" +
            "CONTEXT:\nNEW SOURCE:" + "\nNEW SOURCE:".join(sources) + "\n\nQUESTION:" + question
        )

        return parse(self.question_pipe(prompt)[0]["generated_text"].text.value)

    async def __call__(self, question: str, ignore_filtering: bool = False) -> str:
        return await self.run(question, ignore_filtering)

    @staticmethod
    async def get_response(question: str, ignore_filtering: bool = False) -> str:
        threads: list[asyncio.Task[str]] = []
        for i in AI.instances:
            threads.append(asyncio.create_task(i(question, ignore_filtering)))
        done, pending = await asyncio.wait(threads, return_when=asyncio.FIRST_COMPLETED)
        res = "\n\n".join([e.result() for e in done])
        for t in pending:
            t.cancel()
        return res


def load(key: str = "") -> None:
    """Load the AI model
    @param key: str - The API key if required
    """
    AI(key)


def ask(question: str, print: IO[str] | None = None, ignore_filtering: bool = False) -> str:
    """Ask the AI a question
    @param question: str - The question to ask
    @return str - The response
    """
    result = asyncio.run(AI.get_response(question, ignore_filtering))
    if print:
        print.write(result)
        print.write("\n")
        print.flush()
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


def add(source: str | list[str] | IO[str] | PathLike[str]) -> None:
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

    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".pdf":
            try:
                images = convert_from_path(path)
                source = "file" + str(path) + "\n".join([
                    f"page {id}:" + pytesseract.image_to_string(image) for id, image in enumerate(images)  # type: ignore
                ])
            except Exception as e:
                print(f"Error reading PDF {path}: {e}")
        elif path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            try:
                source = "file" + str(path) + str(pytesseract.image_to_string(Image.open(path)))  # type: ignore
            except Exception as e:
                print(f"Error reading image {path}: {e}")
        else:
            try:
                source = "file" + str(path) + path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Error reading file {path}: {e}")

    SOURCES.append(source)  # type: ignore


def unload() -> str:
    AI.instances.clear()
    return "AI model unloaded successfully"


__all__ = ["load", "ask", "add", "unload"]
