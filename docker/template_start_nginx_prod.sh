#!/usr/bin/env bash
set -euo pipefail

# Fail fast on misconfiguration. The jonasal/nginx-certbot image keeps the
# container "Up" even when certbot can't issue a cert, so a bad DOMAIN or
# CERTBOT_EMAIL silently leaves :443 unconfigured. Validate inputs here so
# the operator sees the problem in `docker logs` instead of staring at a
# healthy-looking container that serves nothing.
if [ -z "${DOMAIN:-}" ]; then
  echo "ERROR: DOMAIN is not set. Edit docker/.env (see DEPLOY.md)." >&2
  exit 1
fi
if [ -z "${CERTBOT_EMAIL:-}" ]; then
  echo "ERROR: CERTBOT_EMAIL is not set. Edit docker/.env (see DEPLOY.md)." >&2
  exit 1
fi
# Loose RFC 5322 check — catches the common cases (missing @, missing TLD,
# leftover placeholder like 'your-email@example'). Certbot's ACME server
# rejects malformed addresses with an opaque traceback, so screen here.
if ! [[ "$CERTBOT_EMAIL" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
  echo "ERROR: CERTBOT_EMAIL='${CERTBOT_EMAIL}' is not a valid email address." >&2
  echo "       Let's Encrypt's ACME server will reject registration." >&2
  echo "       Edit docker/.env and set a real address (see DEPLOY.md)." >&2
  exit 1
fi

envsubst "\$DOMAIN" < /etc/nginx/nginx.prod.template > /etc/nginx/user_conf.d/default.conf
/scripts/start_nginx_certbot.sh