# Migration Guide: File to DB

Phase 11 introduces a SQLite database (`user.db`) to store your Watchlist, Transactions, and Portfolio data. This replaces the loose files (`watchlist.yml`, `transactions.csv`, etc.) as the primary source of truth, although we currently run in "Hybrid Mode" where files act as a fallback.

## How to Migrate

1.  **Backup your data**:
    ```bash
    cp -r data data_backup
    cp watchlist.yml watchlist.yml.bak
    ```

2.  **Run the migration script**:
    ```bash
    python scripts/migrate_files_to_db.py
    ```
    This script will:
    *   Create `data/user.db` if it doesn't exist.
    *   Read `watchlist.yml` and insert items into the DB.
    *   Read `data/user_uploads/transactions.csv` and insert trades.
    *   Read `data/user_uploads/holdings.csv` and insert snapshots.

## Verification
You can check if the migration worked by inspecting the database (using any SQLite viewer) or by setting `STORAGE_MODE=db` and running the app.

## Rollback
If something goes wrong, simply delete the database file:
```bash
rm data/user.db
```
The app will automatically fall back to reading your original files (Hybrid Mode).
