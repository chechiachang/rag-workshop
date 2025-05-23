up:
	docker-compose up -d
	ping 127.0.0.1 -c 2 > /dev/null # hack to wait 2 sec for ngrok to start
	docker logs ngrok

down:
	docker-compose down -v
	rm -rf notebook
