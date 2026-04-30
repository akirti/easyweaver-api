# EasyWeaver Aggregator API - Apigee Proxy

Apigee API Proxy configuration for the EasyWeaver Data Aggregation Platform. Provides rate limiting, JWT authentication, CORS, security headers, and request validation in front of the FastAPI backend.

## Structure

```
apigee/
├── apiproxy/
│   ├── easyweaver-aggregator-api.xml    # Proxy bundle descriptor
│   ├── proxies/
│   │   └── default.xml                  # Proxy endpoint (PreFlow, Flows, FaultRules)
│   ├── targets/
│   │   ├── default.xml                  # Target endpoint (backend /api/v1 load balancer)
│   │   └── infrastructure.xml           # Target for /health, /info (no /api/v1 prefix)
│   ├── policies/
│   │   ├── KVM-Get-Credentials.xml      # KVM lookup for hostname + JWT secret
│   │   ├── JWT-VerifyAccessToken.xml    # JWT token verification (HS256)
│   │   ├── SA-RateLimit.xml             # Spike Arrest - rate limiting by client IP
│   │   ├── Q-EnforceQuota.xml           # Quota enforcement by JWT user_id
│   │   ├── AM-AddCORSHeaders.xml        # CORS response headers
│   │   ├── AM-AddSecurityHeaders.xml    # Security headers (HSTS, CSP, etc.)
│   │   ├── AM-SetTargetHeaders.xml      # Target request headers (X-Forwarded-*, X-Request-Id)
│   │   ├── RF-CORSPreflight.xml         # CORS OPTIONS preflight handling
│   │   ├── RF-CORSInvalidOrigin.xml     # Reject invalid CORS origins
│   │   ├── JS-ValidateCORSOrigin.xml    # JavaScript CORS origin validation
│   │   ├── JS-ValidateQueryRequest.xml  # Validate POST /queries/execute payload
│   │   ├── JS-ExtractCookieToken.xml    # Extract JWT from httpOnly cookie
│   │   ├── JS-LogResponse.xml           # Response logging / analytics
│   │   ├── AM-InvalidJWTResponse.xml    # 401 error for bad JWT
│   │   ├── AM-RateLimitExceededResponse.xml  # 429 spike arrest error
│   │   ├── AM-QuotaExceededResponse.xml # 429 quota exceeded error
│   │   ├── AM-TargetUnavailableResponse.xml  # 503 backend down error
│   │   ├── AM-DefaultErrorResponse.xml  # 500 catch-all proxy error
│   │   ├── AM-DefaultTargetErrorResponse.xml # 502 target communication error
│   │   └── RF-ResourceNotFound.xml      # 404 not found
│   └── resources/
│       └── jsc/
│           ├── validateCORSOrigin.js     # CORS origin allowlist logic
│           ├── validateQueryRequest.js   # Query execute payload validation
│           ├── extractCookieToken.js     # Cookie-to-header JWT extraction
│           └── logResponse.js            # Response analytics logging
└── README.md
```

## Security Architecture

This proxy is designed for a **browser-based SPA** (React/TypeScript frontend) that authenticates via **JWT tokens**. The EasyWeaver platform uses dual-mode JWT (standalone + admin-panel bridge).

### Authentication Flow

```
Browser → Apigee (/easyweaver/v1/*) → Backend (/api/v1/*)
   │                                        │
   ├─ POST /auth/login ────────────────────► Backend issues JWT
   ├─ POST /auth/register ─────────────────► Backend creates user + JWT
   ├─ GET /sources (with JWT) ──► JWT verified by Apigee ──► Backend
   └─ POST /auth/refresh ──────────────────► Backend refreshes JWT
```

### PreFlow Request Pipeline

Every request passes through these steps in order:

| Step | Policy | Condition | Purpose |
|------|--------|-----------|---------|
| 1 | `JS-ValidateCORSOrigin` | Always | Validate Origin header against allowlist |
| 2 | `RF-CORSPreflight` | OPTIONS + valid origin | Return CORS headers, skip remaining steps |
| 3 | `RF-CORSInvalidOrigin` | OPTIONS + invalid origin | Reject with 403 |
| 4 | `KVM-Get-Credentials` | Non-OPTIONS, non-infra | Load hostname + JWT secret from encrypted KVM |
| 5 | `SA-RateLimit` | Non-infra | Spike arrest at 100 req/sec per client IP |
| 6 | `JS-ExtractCookieToken` | Protected endpoints | Extract JWT from httpOnly cookie if no Authorization header |
| 7 | `JWT-VerifyAccessToken` | Protected endpoints | Verify HS256 JWT from Authorization header |
| 8 | `Q-EnforceQuota` | Authenticated, non-auth | 10,000 req/month per user_id |

### JWT-Exempt Endpoints (public access)

These endpoints skip JWT verification:

| Path | Purpose |
|------|---------|
| `POST /auth/login` | User login |
| `POST /auth/register` | User registration |
| `POST /auth/refresh` | Refresh expired access token |
| `GET /health` | Health check |
| `GET /health/*` | Liveness / readiness probes |
| `GET /info` | Application info |
| `OPTIONS *` | CORS preflight requests |

### Quota-Exempt Endpoints

These endpoints skip quota enforcement:

- `/health`, `/health/*`, `/info` — Infrastructure probes
- `/auth/*` — All authentication endpoints
- `OPTIONS` — CORS preflight

### Response Security Headers

Applied to all responses via PostFlow:

- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `X-Request-Id: {messageid}`

## Endpoints

Base path: `/easyweaver/v1`

### Authentication (`/auth`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | User login |
| POST | `/auth/register` | User registration |
| POST | `/auth/refresh` | Refresh JWT access token |
| GET | `/auth/me` | Get current user profile |

### Sources (`/sources`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/sources` | List all data sources |
| POST | `/sources` | Create a new data source |
| POST | `/sources/upload` | Upload a file-based data source |
| GET | `/sources/{id}` | Get source by ID |
| PUT | `/sources/{id}` | Update source |
| DELETE | `/sources/{id}` | Delete source |
| GET | `/sources/{id}/schema` | Get source schema (all tables) |
| GET | `/sources/{id}/schema/{table}` | Get schema for a specific table |
| GET | `/sources/{id}/preview/{table}` | Preview data from a table |
| GET | `/sources/{id}/tables/{table}/columns/{col}/distinct` | Get distinct column values |
| POST | `/sources/{id}/test` | Test source connection |

### Queries (`/queries`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/queries/execute` | Execute a query (validates payload) |
| POST | `/queries/join-results` | Join query results |
| GET | `/queries/runs/{run_id}` | Get query run status |
| GET | `/queries/runs/{run_id}/results` | Get query run results |
| GET | `/queries/runs/{run_id}/export` | Export query run results |
| POST | `/queries/runs/{run_id}/cancel` | Cancel a running query |
| WS | `/queries/run/ws` | WebSocket for query execution* |
| WS | `/queries/ws/{run_id}` | WebSocket for run progress* |

### Processes (`/processes`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/processes` | List all saved processes |
| POST | `/processes` | Create a new process configuration |
| GET | `/processes/{id}` | Get process configuration |
| PUT | `/processes/{id}` | Update process configuration |
| DELETE | `/processes/{id}` | Delete process configuration |
| POST | `/processes/{id}/run` | Run a process |
| GET | `/processes/{id}/runs` | List runs for a process |
| POST | `/processes/{id}/refresh-credentials` | Refresh credentials |
| GET | `/processes/runs/{run_id}` | Get run status |
| GET | `/processes/runs/{run_id}/results` | Get run results |
| GET | `/processes/runs/{run_id}/preview/{key}` | Preview a dataset |
| POST | `/processes/runs/{run_id}/save-results` | Save run results |
| POST | `/processes/runs/{run_id}/reload` | Reload a run |
| WS | `/processes/{id}/run/ws` | WebSocket for process execution* |

### Dashboard (`/dashboard`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard/configs` | List dashboard configurations |
| POST | `/dashboard/configs` | Create dashboard configuration |
| GET | `/dashboard/configs/{id}` | Get dashboard config |
| PUT | `/dashboard/configs/{id}` | Update dashboard config |
| DELETE | `/dashboard/configs/{id}` | Delete dashboard config |
| GET | `/dashboard/configs/{id}/stats` | Get config statistics |
| GET | `/dashboard/configs/{id}/history` | Get config history |
| POST | `/dashboard/configs/{id}/refresh` | Refresh config data |

### Lookups (`/lookups`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/lookups/{process_id}` | Get lookup data for a process |
| POST | `/lookups/{process_id}/refresh` | Refresh lookup data |

### Health & Info
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/health/*` | Sub-health endpoints (live, ready) |
| GET | `/info` | Application info |

*\*WebSocket endpoints require special handling. See WebSocket Support section below.*

## WebSocket Support

The following WebSocket endpoints are defined in the API but **require special handling** at the gateway level:

- `WS /queries/run/ws` — Real-time query execution progress
- `WS /queries/ws/{run_id}` — Run-specific progress stream
- `WS /processes/{config_id}/run/ws` — Process execution progress

Standard Apigee proxies do not natively support WebSocket protocol upgrades. Options:

1. **Apigee X with Envoy**: Use Envoy sidecar or upstream proxy for WS
2. **Separate WS gateway**: Route WS traffic through NGINX/Envoy directly to backend
3. **Direct backend access**: Bypass Apigee for WS connections (ensure auth is validated backend-side)

The proxy flows are defined for documentation purposes and will pass through HTTP requests, but the WebSocket upgrade handshake requires additional infrastructure.

## Deployment

### Prerequisites

1. Apigee Edge or Apigee X account
2. `apigeecli` or Apigee Management API access
3. Target servers configured in Apigee

### 1. Configure Target Servers

```bash
# Create backend target servers
apigeecli targetservers create \
  --name easyweaver-backend-1 \
  --host your-easyweaver-host-1.com \
  --port 443 \
  --ssl true \
  --org YOUR_ORG \
  --env YOUR_ENV

apigeecli targetservers create \
  --name easyweaver-backend-2 \
  --host your-easyweaver-host-2.com \
  --port 443 \
  --ssl true \
  --org YOUR_ORG \
  --env YOUR_ENV
```

### 2. Configure KVM (Key-Value Map)

Store the backend hostname and JWT secret in an encrypted KVM:

```bash
# Create the encrypted KVM
apigeecli kvms create --name easyweaver-proxykvm --encrypted true --org YOUR_ORG --env YOUR_ENV

# Store the backend hostname
apigeecli kvms entries create \
  --map easyweaver-proxykvm \
  --key hostname \
  --value "your-easyweaver-hostname.com" \
  --org YOUR_ORG \
  --env YOUR_ENV

# Store the JWT secret (MUST match backend EASYWEAVER_AUTH_SECRET_KEY)
apigeecli kvms entries create \
  --map easyweaver-proxykvm \
  --key secretKey \
  --value "your-jwt-secret-key" \
  --org YOUR_ORG \
  --env YOUR_ENV
```

> The JWT secret (`secretKey`) must match exactly the `EASYWEAVER_AUTH_SECRET_KEY` used by the backend to sign tokens.

### 3. Deploy the Proxy

```bash
# Package and deploy
cd easyweaver-api/apigee/apiproxy
zip -r ../easyweaver-aggregator-api.zip .

apigeecli apis create \
  --name easyweaver-aggregator-api \
  --file ../easyweaver-aggregator-api.zip \
  --org YOUR_ORG

apigeecli apis deploy \
  --name easyweaver-aggregator-api \
  --rev 1 \
  --env YOUR_ENV \
  --org YOUR_ORG
```

Or using the Management API:

```bash
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/YOUR_ORG/apis?action=import&name=easyweaver-aggregator-api" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@easyweaver-aggregator-api.zip"
```

### 4. Updating After Changes

```bash
cd easyweaver-api/apigee/apiproxy
zip -r ../easyweaver-aggregator-api.zip .

apigeecli apis update \
  --name easyweaver-aggregator-api \
  --file ../easyweaver-aggregator-api.zip \
  --org YOUR_ORG

apigeecli apis deploy \
  --name easyweaver-aggregator-api \
  --rev LATEST \
  --env YOUR_ENV \
  --org YOUR_ORG
```

## Error Responses

All errors follow a consistent JSON format:

```json
{
  "error": "Error Type",
  "message": "Human-readable description",
  "code": "ERROR_CODE",
  "status": 401,
  "timestamp": "2026-04-30T12:00:00.000Z",
  "requestId": "abc-123-def-456"
}
```

| Code | Status | Trigger |
|------|--------|---------|
| `INVALID_TOKEN` | 401 | Missing, expired, or invalid JWT token |
| `QUOTA_EXCEEDED` | 429 | Monthly quota limit reached |
| `RATE_LIMIT_EXCEEDED` | 429 | Spike arrest triggered |
| `SERVICE_UNAVAILABLE` | 503 | Backend unreachable |
| `BAD_GATEWAY` | 502 | Error communicating with backend |
| `INTERNAL_ERROR` | 500 | Unexpected proxy error |
| `RESOURCE_NOT_FOUND` | 404 | Unknown route |

## Target Configuration

The target endpoint uses a **RoundRobin** load balancer with two backend servers:

- Algorithm: RoundRobin (alternates requests evenly)
- Path rewrite: Apigee base `/easyweaver/v1` maps to backend `/api/v1`
- Connection timeout: 30s
- I/O timeout: 60s
- Keepalive: 60s
- Max failures before circuit break: 3
- Retry enabled: yes

## Request Validation

### Query Execute (`POST /queries/execute`)

The `JS-ValidateQueryRequest` policy validates the request payload:

- **Required fields**: `type` (simple|join|multi_join), `left` (with `source_id` and `table`)
- **Join queries**: additionally require `right` (with `source_id` and `table`) and `join_config`
- **max_rows**: if provided, must be a positive integer not exceeding 1,000,000

## JWT Verification

| Claim | Flow Variable | Purpose |
|-------|---------------|---------|
| `user_id` | `jwt.user_id` | Identifies the authenticated user |
| `email` | `jwt.email` | User's email address |
| `roles` | `jwt.roles` (array) | User's roles for authorization |

The policy rejects requests (401) if:
1. No Authorization header present
2. Signature doesn't match the secret key
3. Token is expired (beyond 60s grace)
4. `sub` is not `access_token`
5. `iss` is not `easyweaver-auth`
6. `aud` is not `easyweaver-api`
7. Required claims (`user_id`, `email`, `roles`) are missing
