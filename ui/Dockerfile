# Step 1: Build the static website using a Node.js image
FROM node:18-alpine AS build

# Set the working directory in the container
WORKDIR /app

# Copy the website files (index.html, CSS, and JS files) into the container
COPY . .

# Step 2: Set up a basic web server using nginx to serve the static files
FROM nginx:alpine

# Set the working directory for nginx
WORKDIR /usr/share/nginx/html

# Remove the default nginx index page
RUN rm -rf ./*

# Copy the static files from the build stage to nginx's web root
COPY --from=build /app .

# Expose port 8080 to the outside world
EXPOSE 8080

# Modify NGINX to use port 8080
RUN sed -i 's/listen       80;/listen       8080;/g' /etc/nginx/conf.d/default.conf

# Start nginx automatically when the container is run
CMD ["nginx", "-g", "daemon off;"]
