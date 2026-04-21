#!/usr/bin/env bash
envsubst "\$DOMAIN" < /etc/nginx/nginx.prod.template > /etc/nginx/user_conf.d/default.conf
/scripts/start_nginx_certbot.sh