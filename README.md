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
docker buildx build --platform linux/amd64 -t pytorch_ai:latest .
```

#build API for only amd64
docker buildx build --platform linux/amd64 -t site.com/siteapi:1 --push .

#build UI for only amd64
docker buildx build --platform linux/amd64 -t site.com/siteui:1 --push .


#create secret for huggingface token

kubectl create secret generic huggingface-secret --from-literal=HUGGINGFACE_TOKEN="[TOKEN_HERE]" -n namespace