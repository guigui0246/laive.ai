from threading import Thread
from typing import IO, Any, overload
from transformers import pipeline, AutoModelForCausalLM  # type: ignore


SOURCES: list[str] = []


class AIWrapper:
    def __init__(self, ai: 'AI', question: str) -> None:
        self.result: Any = None
        self.thread: Thread = Thread(target=ai.__call__, args=[question, self], daemon=True)
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def __bool__(self) -> bool:
        return self.result is not None


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

        self.pipe: Any = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            trust_remote_code=True
        )

        AI.instances.append(self)

    async def run(self, question: str) -> Any:
        return self.pipe({
            "role": "user",
            "content": question
        })

    async def __call__(self, question: str, save_obj: AIWrapper) -> Any:
        save_obj.result = await self.run(question)

    @staticmethod
    def get_response(question: str, sources: list[str]) -> str:
        t: list[AIWrapper] = []
        prompt = (
            "## PLEASE ANSWER THE QUESTION USING THE SOURCES ##" +
            "NEW SOURCE:" + "NEW SOURCE:".join(sources) + "QUESTION:" + question
        )
        for i in AI.instances:
            t.append(AIWrapper(i, prompt))
        test = any(a for a in t if a.is_alive())
        while test:
            for i in t:
                if not i.is_alive():
                    return i.result
        raise ValueError("No response")


def load(key: str = "") -> None:
    """Load the AI model
    @param key: str - The API key if required
    """
    AI()


def ask(question: str) -> str:
    """Ask the AI a question
    @param question: str - The question to ask
    @return str - The response
    """
    return AI.get_response(question, SOURCES)


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
    if isinstance(source, IO):
        source = source.read()
    if isinstance(source, list):
        SOURCES.extend(source)
        return
    SOURCES.append(source)


def unload() -> str:
    return "AI model unloaded successfully"


__all__ = ["load", "ask", "add", "unload"]
