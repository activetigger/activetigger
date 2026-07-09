ARG NODE_VERSION="22.13.1"
FROM node:${NODE_VERSION} AS build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./

ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

FROM nginx:1-alpine

COPY --from=build /frontend/dist /var/www/frontend
COPY docker/frontend/nginx.prototype.conf.template /etc/nginx/templates/default.conf.template

RUN mkdir -p /api/static

EXPOSE 80

CMD ["/bin/sh", "-c", ": ${ACTIVETIGGER_API_UPSTREAM:=http://activetigger-api:4000}; envsubst '$ACTIVETIGGER_API_UPSTREAM' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf; exec nginx -g 'daemon off;'"]
