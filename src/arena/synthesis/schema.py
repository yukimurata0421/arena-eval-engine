from __future__ import annotations

CLAIMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    source_path TEXT NOT NULL,
    claim TEXT,
    claim_type TEXT,
    basis TEXT,
    raw_text TEXT,
    evidence_refs TEXT,
    priority_hint TEXT,
    created_at TEXT,
    topic TEXT,
    topic_confidence TEXT,
    topic_method TEXT,
    topic_reason TEXT,
    baseline_label TEXT,
    baseline_type TEXT,
    baseline_confidence TEXT,
    baseline_method TEXT,
    baseline_reason TEXT,
    exploration_axes_json TEXT,
    fixed_conditions_json TEXT,
    variable_conditions_json TEXT,
    validity_scope TEXT,
    review_status TEXT,
    ingested_at TEXT NOT NULL
);
"""

# Backward-compatible alias for existing imports/tests.
HYPOTHESES_TABLE_SQL = CLAIMS_TABLE_SQL


CLAIM_REQUIRED_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS claim_required_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    priority TEXT,
    reason TEXT,
    required_for TEXT,
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE
);
"""


CLAIM_REQUIRED_FILES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_claim_required_files_claim_id
ON claim_required_files(claim_id);
"""


INGESTED_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingested_files (
    source_path TEXT PRIMARY KEY,
    source_sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    source_date TEXT,
    record_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
"""


BASELINE_CLUSTERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS baseline_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL UNIQUE,
    type TEXT,
    confidence TEXT,
    cluster_reason TEXT,
    created_at TEXT NOT NULL
);
"""


CLAIM_BASELINE_LINKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS claim_baseline_links (
    claim_id INTEGER NOT NULL,
    baseline_cluster_id INTEGER NOT NULL,
    link_confidence TEXT,
    PRIMARY KEY (claim_id, baseline_cluster_id),
    FOREIGN KEY(claim_id) REFERENCES claims(id) ON DELETE CASCADE,
    FOREIGN KEY(baseline_cluster_id) REFERENCES baseline_clusters(id) ON DELETE CASCADE
);
"""


CLAIM_BASELINE_LINKS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_claim_baseline_links_cluster_id
ON claim_baseline_links(baseline_cluster_id);
"""


REVIEW_STATUS_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS review_status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    change_reason TEXT,
    FOREIGN KEY(hypothesis_id) REFERENCES claims(id) ON DELETE CASCADE
);
"""


REVIEW_STATUS_HISTORY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_review_status_history_hypothesis_id
ON review_status_history(hypothesis_id);
"""


PROPOSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS propositions (
    proposition_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    target TEXT NOT NULL,
    convergence TEXT NOT NULL CHECK(convergence IN ('convergent','partial','divergent')),
    convergence_detail TEXT NOT NULL,
    resolved_type TEXT NOT NULL CHECK(resolved_type IN ('supported','supported_with_caveat','negative','unresolved','insufficient_data')),
    caveat TEXT,
    exploration_axes TEXT,
    priority_hint TEXT NOT NULL CHECK(priority_hint IN ('high','medium','low')),
    created_at TEXT
);
"""


CLAIM_PROPOSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS claim_propositions (
    claim_id TEXT NOT NULL,
    proposition_id TEXT NOT NULL,
    source_ai TEXT NOT NULL,
    proposition_question TEXT,
    PRIMARY KEY (claim_id, proposition_id, source_ai),
    FOREIGN KEY (proposition_id) REFERENCES propositions(proposition_id)
);
"""


CLAIM_PROPOSITIONS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_claim_propositions_proposition_id
ON claim_propositions(proposition_id);
"""


CLAIM_PROPOSITIONS_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS claim_propositions_view AS
SELECT
    cp.source_ai,
    cp.claim_id,
    cp.proposition_id,
    p.question AS proposition_question,
    p.target AS proposition_target,
    p.convergence,
    p.resolved_type,
    p.caveat,
    p.priority_hint
FROM claim_propositions cp
JOIN propositions p
  ON p.proposition_id = cp.proposition_id;
"""


PROPOSITION_TRIAGE_RECORDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS proposition_triage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposition_id TEXT NOT NULL,
    triage_run_id TEXT NOT NULL,
    review_status_suggested TEXT NOT NULL CHECK(review_status_suggested IN ('auto_accept_candidate','human_review_required','hold','reject_candidate')),
    validity_assessment TEXT NOT NULL CHECK(validity_assessment IN ('strong','moderate','weak','conflicting')),
    support_strength REAL NOT NULL,
    decision_risk REAL NOT NULL,
    review_priority_score REAL NOT NULL,
    consensus_class TEXT NOT NULL CHECK(consensus_class IN ('cross_model_supported','same_model_repeated','cross_model_conflicting','weak_sparse')),
    observation_metrics_json TEXT NOT NULL,
    key_issues_json TEXT NOT NULL,
    evidence_coverage_json TEXT NOT NULL,
    consistency_check_json TEXT NOT NULL,
    action_suggestion_json TEXT NOT NULL,
    rewrite_suggestion TEXT,
    next_data_needed_json TEXT NOT NULL,
    precheck_flags_json TEXT NOT NULL,
    source_provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (proposition_id, triage_run_id),
    FOREIGN KEY (proposition_id) REFERENCES propositions(proposition_id)
);
"""


PROPOSITION_TRIAGE_RECORDS_PROP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prop_triage_records_proposition_id
ON proposition_triage_records(proposition_id);
"""


PROPOSITION_TRIAGE_RECORDS_PRIORITY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prop_triage_records_priority
ON proposition_triage_records(review_priority_score DESC, decision_risk DESC);
"""


PROPOSITION_REVIEW_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS proposition_review_state (
    proposition_id TEXT PRIMARY KEY,
    current_status TEXT NOT NULL CHECK(current_status IN ('pending','triaged','human_review_required','human_confirmed','human_corrected','human_rejected','on_hold')),
    triage_record_id INTEGER,
    updated_at TEXT NOT NULL,
    note TEXT,
    FOREIGN KEY (proposition_id) REFERENCES propositions(proposition_id),
    FOREIGN KEY (triage_record_id) REFERENCES proposition_triage_records(id)
);
"""


PROPOSITION_REVIEW_STATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prop_review_state_status
ON proposition_review_state(current_status);
"""


PROPOSITION_REVIEW_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS proposition_review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposition_id TEXT NOT NULL,
    triage_record_id INTEGER,
    old_status TEXT,
    new_status TEXT NOT NULL,
    change_reason TEXT,
    force_applied INTEGER NOT NULL DEFAULT 0,
    changed_at TEXT NOT NULL,
    FOREIGN KEY (proposition_id) REFERENCES propositions(proposition_id),
    FOREIGN KEY (triage_record_id) REFERENCES proposition_triage_records(id)
);
"""


PROPOSITION_REVIEW_HISTORY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_prop_review_history_proposition_id
ON proposition_review_history(proposition_id);
"""


HYPOTHESES_COMPAT_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS hypotheses AS
SELECT
    id,
    model,
    source_path,
    claim,
    claim_type,
    basis,
    raw_text,
    evidence_refs,
    priority_hint,
    created_at,
    topic,
    topic_confidence,
    topic_method,
    topic_reason,
    baseline_label,
    baseline_type,
    baseline_confidence,
    baseline_method,
    baseline_reason,
    exploration_axes_json,
    fixed_conditions_json,
    variable_conditions_json,
    validity_scope,
    review_status,
    ingested_at
FROM claims;
"""


HYPOTHESES_COMPAT_INSERT_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS trg_hypotheses_compat_insert
INSTEAD OF INSERT ON hypotheses
BEGIN
    INSERT INTO claims (
        id,
        model,
        source_path,
        claim,
        claim_type,
        basis,
        raw_text,
        evidence_refs,
        priority_hint,
        created_at,
        topic,
        topic_confidence,
        topic_method,
        topic_reason,
        baseline_label,
        baseline_type,
        baseline_confidence,
        baseline_method,
        baseline_reason,
        exploration_axes_json,
        fixed_conditions_json,
        variable_conditions_json,
        validity_scope,
        review_status,
        ingested_at
    )
    VALUES (
        NEW.id,
        NEW.model,
        NEW.source_path,
        NEW.claim,
        NEW.claim_type,
        NEW.basis,
        NEW.raw_text,
        NEW.evidence_refs,
        NEW.priority_hint,
        NEW.created_at,
        NEW.topic,
        NEW.topic_confidence,
        NEW.topic_method,
        NEW.topic_reason,
        NEW.baseline_label,
        NEW.baseline_type,
        NEW.baseline_confidence,
        NEW.baseline_method,
        NEW.baseline_reason,
        NEW.exploration_axes_json,
        NEW.fixed_conditions_json,
        NEW.variable_conditions_json,
        NEW.validity_scope,
        NEW.review_status,
        NEW.ingested_at
    );
END;
"""


HYPOTHESES_COMPAT_UPDATE_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS trg_hypotheses_compat_update
INSTEAD OF UPDATE ON hypotheses
BEGIN
    UPDATE claims
    SET
        model = NEW.model,
        source_path = NEW.source_path,
        claim = NEW.claim,
        claim_type = NEW.claim_type,
        basis = NEW.basis,
        raw_text = NEW.raw_text,
        evidence_refs = NEW.evidence_refs,
        priority_hint = NEW.priority_hint,
        created_at = NEW.created_at,
        topic = NEW.topic,
        topic_confidence = NEW.topic_confidence,
        topic_method = NEW.topic_method,
        topic_reason = NEW.topic_reason,
        baseline_label = NEW.baseline_label,
        baseline_type = NEW.baseline_type,
        baseline_confidence = NEW.baseline_confidence,
        baseline_method = NEW.baseline_method,
        baseline_reason = NEW.baseline_reason,
        exploration_axes_json = NEW.exploration_axes_json,
        fixed_conditions_json = NEW.fixed_conditions_json,
        variable_conditions_json = NEW.variable_conditions_json,
        validity_scope = NEW.validity_scope,
        review_status = NEW.review_status,
        ingested_at = NEW.ingested_at
    WHERE id = OLD.id;
END;
"""


HYPOTHESES_COMPAT_DELETE_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS trg_hypotheses_compat_delete
INSTEAD OF DELETE ON hypotheses
BEGIN
    DELETE FROM claims WHERE id = OLD.id;
END;
"""
