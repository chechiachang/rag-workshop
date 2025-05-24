# rag-workshop

[2025 DevOpsDay 工作坊](https://devopsdays.tw/2025/workshop-page/3788)教材
Workshop lecture on [2025 DevOpsDay](https://devopsdays.tw/2025/workshop-page/3788)
- 需要搭配投影片

# 1. (optional) Install Prerequisites on your machine

git clone this repository

- [ChatGPT](https://chatgpt.com/) 有問題先問他
  - [OpenAI API](https://platform.openai.com/settings/organization/api-keys)，產生一隻 api key，放到 .env 裡面
- [docker](https://docs.docker.com/engine/install/)
- [ngrok](https://dashboard.ngrok.com/login)
  - 登入 Login -> 左手邊 Identity & Access -> Authtokens -> Add Tunnel authtoken -> 寫入 `.env` 檔案

.env
```
NGROK_AUTHTOKEN=
```

# 2. Start the Environment 啟動環境

Start the environment
- [jupyter/notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/)
- [qdrant](https://qdrant.tech/documentation/)

```
make up

t=2025-05-23T14:48:43+0000 lvl=info msg="started tunnel" obj=tunnels name=command_line addr=http://notebook:8888 url=https://da38-36-229-233-44.ngrok-free.app
```

使用browser 透過連結進入 notebook
https://da38-36-229-233-44.ngrok-free.app 

password or token:
```
token
```

# 4. Stop the Environment 停止環境並移除資料卷

Stop the environment and remove the volumes

```
make down
```

# Reference

- https://platform.openai.com/docs/guides/embeddings
- https://www.trulens.org/getting_started/quickstarts/quickstart/
- https://github.com/Azure-Samples/ai-rag-chat-evaluator
