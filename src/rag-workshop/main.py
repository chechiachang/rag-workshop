import asyncio
import base64
import os

import logfire
import nest_asyncio
from agents import Agent
from agents import ModelSettings
from agents import OpenAIChatCompletionsModel
from agents import Runner
from dotenv import find_dotenv
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))

# option 1: use openai api.
# You need to have an paid OpenAI account and set the OPENAI_API_KEY environment variable.
# openai_client = AsyncOpenAI()

# option 2: use azure openai api
openai_client = AsyncAzureOpenAI()

# configure the model
model = OpenAIChatCompletionsModel(
    os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    openai_client=openai_client,
)
model_settings = ModelSettings(temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.0)))

# langfuse
secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
host = os.environ.get("LANGFUSE_HOST")
langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = host + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

nest_asyncio.apply()

logfire.configure(
    service_name="workshop",
    send_to_logfire=False,
)
logfire.instrument_openai_agents()


class Summary(BaseModel):
    reasoning: str
    summary: str
    insights: list[str]
    hashtags: list[str]

    def __str__(self) -> str:
        insights = "\n".join([f"  • {insight.strip()}" for insight in self.insights])
        hashtags = " ".join(self.hashtags)
        return "\n\n".join(
            [
                "📝 Summary",
                self.summary.strip(),
                "💡 Insights",
                insights,
                f"🏷️ Hashtags: {hashtags}",
            ]
        )


INSTRUCTIONS = """
You are a helpful assistant
"""
input = "Write a haiku about recursion in programming."


async def main() -> None:
    result = await Runner.run(
        starting_agent=Agent(
            name="Assistant",
            instructions=INSTRUCTIONS.format(language="English", length=200),
            model=model,
            model_settings=model_settings,
            output_type=Summary,
        ),
        input=input,
    )
    print(result.final_output)


asyncio.run(main())

# const embedding = await openai.embeddings.create({
#  model: "text-embedding-3-small",
#  input: "Your text string goes here",
#  encoding_format: "float",
# });

# response = client.responses.create(
#    model="gpt-4.1-mini",
#    input="Write a one-sentence bedtime story about a unicorn."
# )
# print(response.output_text)

# Call the deployed model (deployment name is required)
# response = client.chat.completions.create(
#    model="gpt-41-mini",  # This is your **deployment name**, not model name like in OpenAI
#    messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Explain what a Kubernetes Pod is."},
#    ],
# )
#
# print(response.choices[0].message.content)
