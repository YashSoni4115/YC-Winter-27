CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS commands (
  idempotency_key TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
  status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, command_key TEXT NOT NULL,
  action_id TEXT NOT NULL UNIQUE, timestamp_utc TEXT NOT NULL,
  policy TEXT NOT NULL, state_json TEXT NOT NULL, forecast_json TEXT,
  proposal_json TEXT NOT NULL, response_json TEXT NOT NULL,
  outcome_json TEXT, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS telemetry (
  sequence INTEGER PRIMARY KEY, timestamp_utc TEXT NOT NULL, state_json TEXT NOT NULL
);
PRAGMA user_version = 1;
