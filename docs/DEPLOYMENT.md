# EvalOm deployment hardening guide

This guide covers the minimum operational safety checks for self-hosted deployment of the authenticated EvalOm API.

## Required fail-closed rule

EvalOm refuses to start with a bootstrap admin key on a non-loopback interface unless all of the following are true:

- `EVALOM_ALLOW_REMOTE_ACCESS=1`
- `EVALOM_TLS_CERTFILE` is set
- `EVALOM_TLS_KEYFILE` is set

This is intentional. The service keeps a local-development default, but it fails closed instead of starting an authenticated API without TLS.

## Recommended environment variables

```bash
export EVALOM_HOST=0.0.0.0
export EVALOM_PORT=8000
export EVALOM_ALLOW_REMOTE_ACCESS=1
export EVALOM_TLS_CERTFILE=/etc/ssl/evalom/fullchain.pem
export EVALOM_TLS_KEYFILE=/etc/ssl/evalom/privkey.pem
export EVALOM_ADMIN_API_KEY="$(openssl rand -hex 32)"
export EVALOM_DEFAULT_WORKSPACE="default"
export EVALOM_DB_PATH="postgresql://<user>:<password>@127.0.0.1:5432/evalom"
```

Keep secrets in a real secret manager or a mounted file system. Do not hard-code API keys or database passwords into shell history or source control.

## Reverse proxy pattern

Prefer placing EvalOm behind a TLS-terminating reverse proxy such as Nginx or Caddy. The API can listen on loopback only and the proxy can terminate TLS, enforce rate limits, and add WAF or request logging.

Example Nginx config:

```nginx
server {
    listen 443 ssl http2;
    server_name evalom.example.com;

    ssl_certificate     /etc/ssl/evalom/fullchain.pem;
    ssl_certificate_key /etc/ssl/evalom/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name evalom.example.com;
    return 301 https://$host$request_uri;
}
```

The API itself should still be run on loopback (`EVALOM_HOST=127.0.0.1`) when behind a proxy unless there is a deliberate, security-reviewed deployment reason to expose it directly.

## TLS and certificate rotation

- Use a trusted certificate authority or a managed certificate from your platform.
- Store certificates outside the repository.
- Rotate certificates before they expire.
- Reload the reverse proxy and/or app after rotation without downtime whenever possible.
- Keep private keys protected with file-system permissions and secret storage.

## Secret rotation

Rotate these values on a schedule or after personnel changes:

- `EVALOM_ADMIN_API_KEY`
- workspace admin keys
- database credentials
- any external secret material used by the backend

After rotating a bootstrap key, invalidate any older admin or member keys that are no longer needed, and review the audit log for unexpected access.

## Backup and restore

For SQLite:

- back up the database file regularly
- use a consistent snapshot or volume backup strategy
- test restore procedures in staging before production

For PostgreSQL:

- use a managed Postgres service or a dedicated backup workflow
- keep a regular point-in-time backup schedule
- test a restore path before relying on it in incident response

## Operational checklist before remote access

Before exposing EvalOm to a shared network:

- [ ] a bootstrap key is set and strong
- [ ] remote access is explicitly enabled
- [ ] TLS certificate and key are present and valid
- [ ] reverse proxy or ingress is configured
- [ ] logs and audit events are enabled
- [ ] backups are configured and tested
- [ ] secret rotation and credential handling are documented
- [ ] admin keys and workspace keys are scoped to the intended workspace

## Production posture

EvalOm remains intentionally local-first and single-instance oriented. This deployment guide is for a self-hosted, secure operating mode—not a multi-node SaaS platform. Use it for a trusted internal or controlled external deployment, not for broad public exposure without a hardened ingress layer.
