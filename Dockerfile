FROM python:3.12.5-alpine3.20

# create app directory
WORKDIR /app

# Update package list and upgrade installed packages
RUN apk update
RUN apk upgrade
# Install the necessary packages
RUN apk add python3 python3-dev git wget build-base

# Install app dependencies
COPY src/requirements.txt ./

RUN pip install -r requirements.txt

# copy app source
COPY src /app

EXPOSE 8080

# Run the API
CMD ["uvicorn", "ai:app", "--host", "0.0.0.0", "--port", "8080"]
