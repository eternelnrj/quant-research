# scripts/audit_data.py
from qer.data.loader import DataLoader
from qer.diagnostics.audit_data import run_all_audits

if __name__ == "__main__":
    loader = DataLoader()
    run_all_audits(loader)
