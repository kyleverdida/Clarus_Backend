# Clarus Backend

FastAPI backend for the Clarus Flutter application. It provides authenticated
diabetic-retinopathy screening, history, follow-up monitoring, referrals,
clinical notes, analytics, notifications, audit logging, and account
management.

## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure secrets before starting the API:

```powershell
$env:CLARUS_AUTH_SECRET = "use-a-long-random-secret"
$env:CLARUS_BOOTSTRAP_SECRET = "use-a-separate-bootstrap-secret"
```

`CLARUS_AUTH_SECRET` signs bearer tokens. `CLARUS_BOOTSTRAP_SECRET` is used
only when creating the first administrator and should be rotated or removed
after bootstrap. Never commit either secret.

## Run

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at
`http://localhost:8000/docs`.

## Roles and authorization

The stable roles are:

- `health_worker`
- `specialist`
- `ophthalmologist_optometrist`

The BUCAS administrator is one Specialist account with `is_admin = true`.
There is no separate user-facing Admin role. Only that account can create,
activate, or deactivate users and view center-wide analytics.

Health Workers can create screenings and access their own screening and
follow-up plans. Follow-up plans are visible and editable only by their creator
or an administrator. Clinical users can access referrals assigned to them. Referral
creators can also access the referrals they created. The BUCAS Specialist/Admin
can access all referrals and center-wide analytics.

## Authentication endpoints

- `POST /auth/bootstrap-admin` creates the first Specialist administrator.
  Requires the `X-Bootstrap-Secret` header and refuses to run once a user
  exists.
- `POST /auth/login` accepts an email and password and returns a bearer token.
- `GET /auth/me` returns the authenticated user.
- Repeated failed login attempts are temporarily throttled.
- `GET /auth/users` lists accounts for the administrator.
- `POST /auth/users` creates Health Worker, Specialist, or
  Ophthalmologist/Optometrist accounts.
- `PATCH /auth/users/{user_id}/active` activates or deactivates an account.

Passwords are stored as scrypt hashes. Tokens expire and inactive accounts
cannot authenticate. Configure `CLARUS_CORS_ORIGINS` as a comma-separated
allowlist for browser clients; wildcard origins are not enabled.

## Screening and history endpoints

- `POST /predict` accepts a retinal image plus `worker_name` and `patient_id`.
  It requires an authenticated Health Worker or Specialist.
- `GET /history` returns authorized integrated encounters only.
- `GET /comparison` returns integrated and baseline records for research and
  requires administrator access.
- `GET /model-info` requires authentication.
- `GET /gradcam/{filename}` serves a Grad-CAM image only to an authorized
  encounter owner, assigned clinician, or administrator, and records a view
  audit event.

Accepted images create an integrated encounter for the application and a
baseline record for research comparison. Rejected images are not saved.

## Referral workflow

- `POST /referrals` creates a referral only for an integrated `Refer` encounter,
  requires patient consent, and assigns it to a Specialist or
  Ophthalmologist/Optometrist.
- `GET /referrals` lists referrals authorized for the current user.
- `GET /referrals/{referral_id}` returns the encounter and clinical notes.
- `PATCH /referrals/{referral_id}` updates referral status.
- `GET /referrals/{referral_id}/notes` lists notes.
- `POST /referrals/{referral_id}/notes` adds a clinical note and optional
  recommendation.

Supported referral statuses are `assigned`, `in_review`, `reviewed`,
`completed`, and `cancelled`. Referral viewing, creation, note changes, and
status changes are audit logged.

## Follow-ups and reminders

- `POST /follow-ups` creates a consented plan.
- `GET /follow-ups` lists plans created by the authenticated user, or all plans
  for an administrator.
- `PATCH /follow-ups/{follow_up_id}` changes status or return date.
- `POST /reminders/send` manually triggers the administrator-only reminder
  check.

Follow-up plans accept `patient_id`, optional `encounter_id`, `contact_method`
(`Email`), `contact_value`, `consent_given`, and a future `return_date`.
Gmail reminders require:

```powershell
$env:CLARUS_EMAIL_USER = "your-email@gmail.com"
$env:CLARUS_EMAIL_APP_PASSWORD = "your-gmail-app-password"
```

Never commit an email app password.

## Analytics and notifications

- `GET /analytics/summary` returns center-wide triage counts and requires the
  BUCAS Specialist/Admin.
- `GET /notifications` lists the current user's in-app notifications.
- `POST /notifications/{notification_id}/read` marks a notification as read.

New referral assignments notify the assigned clinician. Referral status
changes notify the other workflow participant. Email notifications for
referrals are not currently enabled.

## Audit and privacy controls

- `POST /audit/report-share` records a report-sharing event.
- Referral access and changes are recorded in the audit log.
- Grad-CAM access is authenticated and audited.
- Patient consent is required before referrals and follow-up reminders.
- SQLite stores application data in `clarus.db`; the database path can be
  overridden with `CLARUS_DB_PATH`. Protect this file and
  define backup and retention procedures before production use.

Use HTTPS, strong secrets, restricted database access, and a production
authentication secret outside local development.

## Implementation status

| Component | Status |
|---|---|
| Image-quality check | Real OpenCV blur/brightness detection |
| Classifier | Integrated through `app/classifier.py`; verify the configured model package before production |
| Triage mapping | Implemented logic; requires clinical validation |
| Grad-CAM | Generated and served through an authenticated endpoint |
| Authentication | Bearer tokens, scrypt passwords, role checks, and inactive-account rejection |
| Database | SQLite with users, encounters, follow-ups, referrals, notes, notifications, and audit logs. Integrated and baseline screening results are rows in `encounters`; Grad-CAM PNGs are files, not database blobs. |

## Connecting the Flutter app

1. Start this backend on port `8000`.
2. Keep `useMockApi = false` in
   `Clarus_App/clarus_app/lib/services/api_service.dart`.
3. Use `http://10.0.2.2:8000` for an Android emulator,
   `http://127.0.0.1:8000` for desktop/web on the same computer, or the
   computer's local network IP for a physical device.
4. Bootstrap the first BUCAS Specialist/Admin, then create the remaining
   accounts from the admin workflow.

## Validation

```powershell
python -m compileall -q app
```

The Flutter project validates its integration with:

```powershell
flutter analyze
flutter test
```
