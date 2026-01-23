import sys
import yaml
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from storage import db, repo, models
from storage.datamanager import data_manager

def migrate():
    print("Starting migration to SQLite...")
    
    # Initialize DB tables
    engine = db.init_db()
    models.Base.metadata.create_all(bind=engine)
    
    # 1. User & Portfolio
    user = repo.get_or_create_default_user()
    portfolio_id = repo.get_default_portfolio_id(user.id)
    print(f"User ID: {user.id}, Portfolio ID: {portfolio_id}")
    
    # 2. Watchlist
    watchlist_path = ROOT / "watchlist.yml"
    if watchlist_path.exists():
        try:
            data = yaml.safe_load(watchlist_path.read_text())
            tickers = data.get("tickers", [])
            repo.replace_watchlist(user.id, tickers)
            print(f"Migrated {len(tickers)} watchlist items.")
        except Exception as e:
            print(f"Error reading watchlist.yml: {e}")
            
    # 3. Transactions
    tx_path = ROOT / "data" / "user_uploads" / "transactions.csv"
    if tx_path.exists():
        try:
            df = pd.read_csv(tx_path)
            # Map CSV columns to Trade model
            # Expected CSV cols: date, ticker, amount (shares/price can be inferred or raw)
            # This logic mimics current loading but writes to DB
            trades = []
            for _, row in df.iterrows():
                # Infer action if missing
                action = row.get("action", "BUY").upper()
                
                trade = {
                    "date": pd.to_datetime(row["date"]),
                    "ticker": row["ticker"],
                    "action": action,
                    "shares": float(row.get("shares", 0.0)), # Should be present if it's the raw ledger?
                    # Or is the user_uploads/transactions.csv the User's broker file or the Normalized file?
                    # Assuming normalized for now as per schema in P11.0-A step 1 where we defined 'trades'
                    # But prompt says "detect current source-of-truth files".
                    # Let's import what we have, gracefully handling missing cols.
                    "amount": float(row.get("amount", 0.0)), 
                    "price": float(row.get("price", 0.0)),
                    "fees": float(row.get("fees", 0.0)),
                    "notes": str(row.get("notes", ""))
                }
                
                # Basic fill if shares missing but amount present (and vice versa)? 
                # For migration, let's just copy what's there.
                
                trades.append(trade)
            
            repo.replace_trades(portfolio_id, trades)
            print(f"Migrated {len(trades)} transactions.")
        except Exception as e:
            print(f"Error migrating transactions: {e}")

    # 4. Holdings Snapshot (if exists)
    snap_path = ROOT / "data" / "user_uploads" / "holdings.csv"
    if snap_path.exists():
        try:
            df = pd.read_csv(snap_path)
            snaps = []
            for _, row in df.iterrows():
                snap = {
                    "as_of_date": pd.to_datetime("today"), # Snapshots usually lack date in simple CSVs?
                    "ticker": row["ticker"],
                    "shares": float(row["shares"]),
                    "cost_basis": float(row.get("cost_basis", 0.0))
                }
                snaps.append(snap)
            repo.replace_snapshot(portfolio_id, snaps)
            print(f"Migrated {len(snaps)} holdings snapshot items.")
        except Exception as e:
            print(f"Error migrating holdings: {e}")

    print("Migration complete.")

if __name__ == "__main__":
    migrate()
