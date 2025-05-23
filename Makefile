up:
	docker-compose up -d
	docker logs ngrok

down:
	docker-compose down -v
	rm -rf notebook
