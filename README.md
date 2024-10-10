# pytorch_ai
To run locally run the following:
```
docker build -t pytorch_ai:latest .
docker run -it --publish 8080:8080 pytorch_ai:latest
```

## Multi-architecture support
```
docker buildx create --use --name mybuilder
docker buildx inspect mybuilder --bootstrap
docker buildx build --platform linux/amd64,linux/arm64 -t pytorch_ai:latest .
```

#build for only amd64
