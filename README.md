# rag-workshop

- [2025 DevOpsDay 工作坊](https://devopsdays.tw/2025/workshop-page/3788)教材
- Workshop lecture on [2025 DevOpsDay](https://devopsdays.tw/2025/workshop-page/3788)
- 需要搭配投影片[https://chechia.net/zh-hant/slides/2025-06-05-devops-rag-internal-ai/](https://chechia.net/zh-hant/slides/2025-06-05-devops-rag-internal-ai/)

# Getting Started

```bash
git clone https://github.com/chechiachang/rag-workshop.git

cd rag-workshop

docker compose up -d

docker exec -it notebook pip install pandas openai qdrant_client tqdm tenacity wget tenacity unstructured markdown ragas sacrebleu langchain_qdrant langchain-openai langchain_openai langchain_community tiktoken

token="workshop1234!"
```

---

# (optional) Tunnel to Jupyter Notebook

- [ngrok](https://dashboard.ngrok.com/login)
  - 登入 Login -> 左手邊 Identity & Access -> Authtokens -> Add Tunnel authtoken to docker-compose.yaml
  - Use https://da38-36-229-233-44.ngrok-free.app to access Jupyter Notebook
- [Pinggy](https://pinggy.io/)，

```
docker compose up -d ngrok
docker logs ngrok

t=2025-05-23T14:48:43+0000 lvl=info msg="started tunnel" obj=tunnels name=command_line addr=http://notebook:8888 url=https://da38-36-229-233-44.ngrok-free.app
```

# Reference

- https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/qdrant/Getting_started_with_Qdrant_and_OpenAI.ipynb
- https://platform.openai.com/docs/guides/embeddings
- https://www.trulens.org/getting_started/quickstarts/quickstart/
- https://github.com/Azure-Samples/ai-rag-chat-evaluator
- https://github.com/kdhaw6/LLM-RAG-based-system
- https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb
