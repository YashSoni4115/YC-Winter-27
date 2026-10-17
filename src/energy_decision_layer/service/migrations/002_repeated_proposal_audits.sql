-- A proposal may be submitted repeatedly; every rejected attempt needs its own audit.
-- Idempotency is enforced by commands, and stale-state safety prevents re-actuation.
CREATE TABLE decisions_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT, command_key TEXT NOT NULL,
  action_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL,
  policy TEXT NOT NULL, state_json TEXT NOT NULL, forecast_json TEXT,
  proposal_json TEXT NOT NULL, response_json TEXT NOT NULL,
  outcome_json TEXT, status TEXT NOT NULL
);
INSERT INTO decisions_v2 SELECT * FROM decisions;
DROP TABLE decisions;
ALTER TABLE decisions_v2 RENAME TO decisions;
CREATE INDEX decisions_action_id ON decisions(action_id);
PRAGMA user_version = 2;
