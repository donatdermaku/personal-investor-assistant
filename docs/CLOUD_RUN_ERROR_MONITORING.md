# Cloud Run Error Monitoring

This project emits structured error logs with the prefix `ERROR_EVENT` from `src/api/server.py`.

## Error Signals Implemented

- `UNHANDLED_EXCEPTION`: unexpected runtime exceptions in request middleware.
- `HTTP_5XX`: responses with status `>=500`.

Each event includes:
- `timestamp`
- `path`
- `method`
- `status`
- `error_code`
- `message`

## Cloud Logging Query Examples

Replace `<service-name>` and `<project-id>`.

```text
resource.type="cloud_run_revision"
resource.labels.service_name="<service-name>"
severity>=ERROR
textPayload:"ERROR_EVENT"
```

Filter by status:

```text
resource.type="cloud_run_revision"
resource.labels.service_name="<service-name>"
textPayload:"ERROR_EVENT"
textPayload:"\"status\": 500"
```

Tail in CLI:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="<service-name>" AND textPayload:"ERROR_EVENT"' \
  --project "<project-id>" \
  --limit 50 \
  --format json
```

## Local/Admin Inspection

Use:

- `GET /admin/error-events`

Query params:
- `limit` (default `20`, max `200`)
- `status_min` (default `500`)

Header:
- `x-admin-key: <ADMIN_WARMUP_KEY>`
