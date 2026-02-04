# Best Practices Guide
## Personal Investment Assistant Platform

**Version:** 2.0  
**Last Updated:** 2026-02-04  
**Target Audience:** Cross-functional development teams

---

## Table of Contents

1. [Team Structure & Roles](#1-team-structure--roles)
2. [Frontend Developer Best Practices](#2-frontend-developer-best-practices)
3. [Backend Developer Best Practices](#3-backend-developer-best-practices)
4. [DevOps Engineer Best Practices](#4-devops-engineer-best-practices)
5. [QA Engineer Best Practices](#5-qa-engineer-best-practices)
6. [Solutions Architect Best Practices](#6-solutions-architect-best-practices)
7. [Data Engineer Best Practices](#7-data-engineer-best-practices)
8. [Security Engineer Best Practices](#8-security-engineer-best-practices)
9. [Product Manager Best Practices](#9-product-manager-best-practices)
10. [Cross-Team Collaboration](#10-cross-team-collaboration)

---

## 1. Team Structure & Roles

### Recommended Team Composition

In a **startup/small team** (5-10 people), roles often overlap:

```
Product Manager (PM)
  ├── Frontend Engineer (React/TypeScript)
  ├── Backend Engineer (Python/FastAPI)
  ├── DevOps Engineer (Docker/CI/CD)
  ├── QA Engineer (Testing/Automation)
  └── Solutions Architect (Tech decisions)
```

**Role Overlap Guidelines:**
- Backend Engineer often handles Data Engineering tasks
- DevOps Engineer initially covers Security responsibilities  
- Frontend Engineer may handle UI/UX if no dedicated designer
- Solutions Architect role can be part-time / advisory

---

## 2. Frontend Developer Best Practices

### Core Technology Stack

```
React 19 + TypeScript 5.x
Next.js 15 OR Vite 5
TailwindCSS 4
TanStack Query (React Query)
Zustand (global state)
```

### Project Structure

```
web/src/
├── components/
│   ├── atoms/          # Button, Input, Badge
│   ├── molecules/      # SearchBar, PriceDisplay
│   ├── organisms/      # PortfolioCard, TradeForm
│   └── layouts/        # Page templates
├── hooks/              # Custom React hooks
├── lib/                # API clients, utilities
├── types/              # TypeScript definitions
└── app/                # Routes (Next.js) or pages/
```

### Best Practice: Type-Safe Components

**Always use TypeScript interfaces:**

```typescript
// components/molecules/StockTicker.tsx
import { FC } from 'react';

interface StockTickerProps {
  symbol: string;
  price: number;
  change: number;
  loading?: boolean;
}

export const StockTicker: FC<StockTickerProps> = ({ 
  symbol, 
  price, 
  change,
  loading = false 
}) => {
  if (loading) return <SkeletonTicker />;
  
  const changeColor = change >= 0 ? 'text-green-600' : 'text-red-600';
  
  return (
    <div className="flex items-center gap-2">
      <span className="font-semibold">{symbol}</span>
      <span className="text-lg">${price.toFixed(2)}</span>
      <span className={changeColor}>
        {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(2)}%
      </span>
    </div>
  );
};
```

### Best Practice: Server State Management

**Use TanStack Query for all API calls:**

```typescript
// hooks/usePortfolio.ts
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api-client';

export const usePortfolio = (portfolioId: string) => {
  return useQuery({
    queryKey: ['portfolio', portfolioId],
    queryFn: () => api.getPortfolio(portfolioId),
    staleTime: 30_000,      // Data fresh for 30s
    refetchInterval: 60_000, // Auto-refetch every minute
    retry: 3,
  });
};

// Usage in component
function PortfolioDashboard() {
  const { data, isLoading, error } = usePortfolio('default');
  
  if (error) return <ErrorState error={error} />;
  if (isLoading) return <PortfolioSkeleton />;
  
  return <PortfolioView portfolio={data} />;
}
```

### Best Practice: Chart Performance

**Decision matrix for choosing chart libraries:**

| Data Points | Chart Type | Library | Reason |
|-------------|------------|---------|--------|
| < 100 | Pie, Bar | Recharts | Beautiful API, easy customization |
| 100-1000 | Line, Area | Recharts | Still performant |
| 1000+ | Candlestick, Time Series | Lightweight Charts | WebGL-accelerated, handles 100k+ points |

### Frontend Checklist

- ✅ All components have TypeScript types
- ✅ Use TanStack Query for server state
- ✅ Error boundaries for crash protection
- ✅ Loading states (skeleton screens, not spinners)
- ✅ Accessibility: keyboard navigation, ARIA labels
- ✅ Test IDs on interactive elements (`data-testid`)

---

## 3. Backend Developer Best Practices

### Core Technology Stack

```
FastAPI 0.115+
Pydantic 2.x
SQLAlchemy 2.x
PostgreSQL 16
Redis 7
Celery (background jobs)
```

### Project Structure

```
src/
├── api/
│   ├── routes/         # Endpoint definitions
│   ├── dependencies.py # Dependency injection
│   └── middleware.py   # Auth, CORS, logging
├── services/           # Business logic layer
│   ├── portfolio_service.py
│   └── market_data_service.py
├── models/             # SQLAlchemy ORM models
├── schemas/            # Pydantic request/response models
├── repositories/       # Data access layer
└── core/
    ├── config.py       # Settings management
    ├── security.py     # Auth, encryption
    └── database.py     # Session management
```

### Best Practice: Separation of Concerns

**Keep routes thin, logic in services:**

```python
# ❌ BAD: Logic in route handler
@app.post("/portfolio/rebalance")
def rebalance_portfolio(portfolio_id: str, db: Session = Depends(get_db)):
    portfolio = db.query(Portfolio).filter_by(id=portfolio_id).first()
    # 50 lines of rebalancing math...
    # Database updates...
    return result

# ✅ GOOD: Route delegates to service
@app.post("/portfolio/rebalance")
def rebalance_portfolio(
    portfolio_id: str,
    service: PortfolioService = Depends(get_portfolio_service)
):
    return service.rebalance(portfolio_id)
```

### Best Practice: Strict Input Validation

**Use Pydantic for schema validation:**

```python
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from datetime import date
from typing import Literal

class TradeCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    action: Literal['BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL']
    quantity: Decimal = Field(ge=0, decimal_places=6)
    price: Decimal = Field(gt=0, decimal_places=2)
    trade_date: date
    fees: Decimal = Field(default=Decimal('0'), ge=0)
    
    @validator('ticker')
    def ticker_uppercase(cls, v):
        return v.upper().strip()
    
    class Config:
        json_encoders = {Decimal: str}  # Serialize Decimal as string
```

### Best Practice: External API Resilience

**Circuit breaker pattern for external APIs:**

```python
from circuitbreaker import circuit
import aiohttp

class YahooFinanceClient:
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def get_quote(self, symbol: str) -> dict:
        try:
            async with self.session.get(
                f"{self.base_url}/quote",
                params={"symbols": symbol},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise MarketDataUnavailableError(f"Failed to fetch {symbol}")
```

### Best Practice: Financial Precision

**ALWAYS use Decimal for currency calculations:**

```python
from decimal import Decimal, ROUND_HALF_UP

# ✅ CORRECT - no rounding errors
def calculate_portfolio_value(holdings: list[Holding]) -> Decimal:
    total = Decimal('0.00')
    for holding in holdings:
        value = holding.quantity * holding.price
        total += value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return total

# ❌ WRONG - will have floating point errors!
bad_value = sum(float(h.quantity) * float(h.price) for h in holdings)
```

### Backend Checklist

- ✅ All money amounts use `Decimal` type
- ✅ Routes are thin, logic in services
- ✅ Pydantic validation on all inputs
- ✅ Circuit breakers for external APIs
- ✅ Structured logging (JSON format)
- ✅ OpenAPI docs auto-generated at `/docs`

---

## 4. DevOps Engineer Best Practices

### Core Technology Stack

```
Docker + Docker Compose
GitHub Actions (CI/CD)
Cloud: GCP Cloud Run / AWS ECS / Railway
PostgreSQL (Cloud SQL / RDS)
Redis (Memorystore / Elasticache)
Monitoring: Prometheus + Grafana
```

### Best Practice: Multi-Stage Docker Builds

**Optimize image size and security:**

```dockerfile
# Backend Dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY src/ ./src/
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["gunicorn", "src.api.server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300", \
     "--graceful-timeout", "300", \
     "--bind", "0.0.0.0:8000"]
```

### Best Practice: CI/CD Pipeline

**GitHub Actions workflow:**

```yaml
name: Backend CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Lint & Type Check
        run: |
          ruff check src/
          black --check src/
          mypy src/
      
      - name: Run Tests
        run: pytest --cov=src --cov-report=xml
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/test
      
      - name: Security Scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: investor-api
          image: gcr.io/${{ secrets.GCP_PROJECT }}:${{ github.sha }}
```

### Best Practice: Local Development Environment

**docker-compose.yml:**

```yaml
version: '3.9'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/investor
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./backend/src:/app/src  # Hot reload
  
  frontend:
    build: ./web
    ports:
      - "3000:80"
    depends_on:
      - backend
  
  db:
    image: timescale/timescaledb:latest-pg16
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=investor
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Best Practice: Observability

**Structured logging:**

```python
import structlog

logger = structlog.get_logger()

# In your code
logger.info(
    "portfolio_computed",
    portfolio_id=portfolio_id,
    duration_ms=elapsed * 1000,
    tickers_count=len(tickers),
    user_id=current_user.id
)
```

**Grafana dashboard panels:**
- API request rate (requests/sec by endpoint)
- Error rate (4xx, 5xx)
- P95 latency
- Database connection pool usage
- Celery queue length

### DevOps Checklist

- ✅ Multi-stage Docker builds
- ✅ CI/CD pipeline with tests
- ✅ Secrets in environment variables (never committed)
- ✅ Monitoring/alerting configured
- ✅ Staging environment mirrors production
- ✅ Rollback procedure documented

---

## 5. QA Engineer Best Practices

### Testing Pyramid

```
     /\
    /  \     E2E Tests (5%) - Critical user flows
   /────\    
  /──────\   Integration Tests (30%) - API contracts
 /────────\  
/──────────\ Unit Tests (65%) - Logic, calculations
```

### Best Practice: Rigorous Unit Testing

**Financial calculations must be tested thoroughly:**

```python
# tests/unit/test_portfolio_math.py
import pytest
from decimal import Decimal
from src.services.portfolio_service import calculate_twr

def test_time_weighted_return_basic():
    """Test TWR calculation with no cash flows."""
    prices = [Decimal('100.00'), Decimal('110.00')]  # 10% gain
    cash_flows = []
    
    result = calculate_twr(prices, cash_flows)
    
    assert result == Decimal('0.10')  # 10% return

def test_time_weighted_return_with_deposit():
    """TWR should be unaffected by deposits."""
    prices = [
        Decimal('100.00'),  # Day 0
        Decimal('110.00'),  # Day 1: 10% gain
        Decimal('165.00'),  # Day 2: $50 deposit + 10% gain on new total
    ]
    cash_flows = [(1, Decimal('50.00'))]
    
    result = calculate_twr(prices, cash_flows)
    assert result == pytest.approx(Decimal('0.10'), abs=Decimal('0.001'))

def test_division_by_zero_protection():
    """Ensure no crash when price is zero."""
    from src.portfolio import align_benchmark
    
    benchmark = pd.DataFrame({'adj_close': [0.0]})  # Invalid
    portfolio = pd.Series([10000], index=pd.to_datetime(['2023-01-01']))
    
    result = align_benchmark(benchmark, portfolio)
    assert result.empty  # Should return empty, not crash
```

### Best Practice: Integration Testing

**Test API endpoints with real database:**

```python
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_create_portfolio_complete_flow():
    # Create portfolio
    response = client.post("/portfolios", json={"name": "Test Portfolio"})
    assert response.status_code == 201
    portfolio_id = response.json()["id"]
    
    # Add transaction
    response = client.post(
        f"/portfolios/{portfolio_id}/transactions",
        json={
            "ticker": "AAPL",
            "action": "BUY",
            "quantity": "10.5",
            "price": "150.00",
            "trade_date": "2024-01-15"
        }
    )
    assert response.status_code == 201
    
    # Verify portfolio value
    response = client.get(f"/portfolios/{portfolio_id}/summary")
    assert response.status_code == 200
    assert Decimal(response.json()["total_value"]) == Decimal("1575.00")
```

### Best Practice: E2E Critical Flows

**Playwright for browser testing:**

```typescript
// tests/e2e/upload-portfolio.spec.ts
import { test, expect } from '@playwright/test';

test('upload CSV and view dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000');
  
  // Login
  await page.fill('[data-testid="email"]', 'test@example.com');
  await page.fill('[data-testid="password"]', 'password123');
  await page.click('[data-testid="login-button"]');
  
  // Wait for dashboard
  await expect(page.locator('h1')).toContainText('Dashboard');
  
  // Upload file
  await page.click('[data-testid="upload-button"]');
  await page.setInputFiles('[data-testid="file-input"]', 'fixtures/trades.csv');
  await page.click('[data-testid="submit-upload"]');
  
  // Wait for processing (max 60s)
  await page.waitForSelector('[data-testid="processing-complete"]', { 
    timeout: 60000 
  });
  
  // Verify TWR metric displayed
  const twrElement = page.locator('[data-testid="metric-twr"]');
  await expect(twrElement).toBeVisible();
  await expect(twrElement).toContainText('%');
  
  // Screenshot for visual regression
  await page.screenshot({ path: 'screenshots/dashboard.png', fullPage: true });
});
```

### QA Checklist

- ✅ Unit tests cover edge cases (zero, negative, extreme values)
- ✅ Integration tests use real database
- ✅ E2E tests cover critical user journeys
- ✅ Test data fixtures maintained
- ✅ Coverage > 80%
- ✅ Tests run in CI/CD pipeline

---

## 6. Solutions Architect Best Practices

### Best Practice: Architecture Decision Records

**Template for major technical decisions:**

```markdown
# ADR-001: Use PostgreSQL with TimescaleDB

**Status:** Accepted  
**Date:** 2026-02-01  
**Deciders:** Tech Lead, Backend Lead, Solutions Architect

## Context
Need to store millions of market data points (OHLC per ticker per day).
Queries: "Get price history for AAPL from 2020-2024"

## Decision
Use PostgreSQL 16 with TimescaleDB extension.

## Alternatives Considered
1. **MongoDB** - NoSQL, flexible schema
   - Rejected: Weak transaction support, less mature for financial data
2. **InfluxDB** - Purpose-built time-series DB
   - Rejected: Separate DB for different data types adds complexity

## Consequences
**Positive:**
- Single database for relational + time-series data
- ACID transactions for financial integrity
- Automatic partitioning via TimescaleDB hypertables

**Negative:**
- Learning curve for team
- Slightly more complex than vanilla Postgres
```

### Best Practice: Multi-Level Caching

**Cache architecture:**

```
Client
  ↓
CDN (CloudFlare) → Static assets (HTML, JS, CSS)
  ↓
App Server
  ↓
Redis (L2 Cache, 30s TTL) → Market prices, session data
  ↓
PostgreSQL (L3, Source of Truth) → All persistent data
```

**Implementation:**

```python
async def get_stock_price(symbol: str, redis: Redis, db: Session) -> Decimal:
    # L2: Redis cache
    cached = await redis.get(f"price:{symbol}")
    if cached:
        logger.info("cache_hit", symbol=symbol, source="redis")
        return Decimal(cached)
    
    # L3: Database
    price = db.query(MarketQuote)\
        .filter_by(symbol=symbol)\
        .order_by(desc(MarketQuote.time))\
        .first()
    
    if price:
        await redis.setex(f"price:{symbol}", 30, str(price.close))
        logger.info("cache_miss", symbol=symbol, source="database")
        return price.close
    
    # Miss: Fetch from external API
    price = await fetch_from_yahoo(symbol)
    await redis.setex(f"price:{symbol}", 30, str(price))
    db.add(MarketQuote(symbol=symbol, close=price, time=datetime.now()))
    db.commit()
    logger.info("cache_miss", symbol=symbol, source="external_api")
    return price
```

### Best Practice: Database Schema Design

**Double-entry bookkeeping for financial integrity:**

```sql
-- Accounts table
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(20) CHECK (type IN ('ASSET', 'LIABILITY', 'EQUITY', 'INCOME', 'EXPENSE')),
    name VARCHAR(100) NOT NULL,
    currency CHAR(3) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Journal entries (transactions)
CREATE TABLE journal_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Postings (splits)
CREATE TABLE postings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    journal_entry_id UUID REFERENCES journal_entries(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id),
    amount NUMERIC(20, 6) NOT NULL,
    direction VARCHAR(6) CHECK (direction IN ('DEBIT', 'CREDIT'))
);

-- CRITICAL: Ensure balanced entries (debits = credits)
CREATE FUNCTION check_balanced_entry() RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT SUM(CASE WHEN direction = 'DEBIT' THEN amount ELSE -amount END)
        FROM postings WHERE journal_entry_id = NEW.journal_entry_id) != 0 THEN
        RAISE EXCEPTION 'Journal entry not balanced - debits must equal credits';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_balanced_entry
AFTER INSERT OR UPDATE ON postings
FOR EACH ROW EXECUTE FUNCTION check_balanced_entry();
```

### Architect Checklist

- ✅ ADRs for all major decisions
- ✅ Architecture diagrams (C4 model)
- ✅ Database schema reviewed for normalization
- ✅ Caching strategy documented
- ✅ Scalability plan defined
- ✅ Non-functional requirements (SLAs) specified

---

## 7. Data Engineer Best Practices

### ETL Pipeline Architecture

```
External APIs → Ingestion Job → Raw Layer → Transformation → Curated Layer
(Yahoo Finance)   (Scheduled)    (Postgres)   (Cleaning)    (Analytics-Ready)
```

### Best Practice: Data Quality Checks

**Validate before loading:**

```python
def validate_market_data(df: pd.DataFrame) -> bool:
    """Run data quality checks before loading to database."""
    checks = [
        ("No nulls in price columns", 
         df[['open', 'high', 'low', 'close']].notna().all().all()),
        ("High >= Low", 
         (df['high'] >= df['low']).all()),
        ("Close within OHLC range", 
         ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all()),
        ("Positive volume", 
         (df['volume'] >= 0).all()),
        ("No duplicate timestamps", 
         df['time'].is_unique),
        ("Valid symbols", 
         df['symbol'].str.match(r'^[A-Z]{1,5}$').all()),
    ]
    
    for check_name, passed in checks:
        if not passed:
            logger.error("data_quality_check_failed", check=check_name)
            raise ValueError(f"Data quality check failed: {check_name}")
    
    logger.info("data_quality_checks_passed", checks_count=len(checks))
    return True
```

### Best Practice: Scheduled ETL (Airflow)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_market_data_ingestion',
    default_args=default_args,
    schedule_interval='0 18 * * 1-5',  # 6 PM weekdays after market close
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    fetch_prices = PythonOperator(
        task_id='fetch_stock_prices',
        python_callable=fetch_all_tickers_from_yahoo
    )
    
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=run_data_quality_checks
    )
    
    load_to_db = PythonOperator(
        task_id='load_to_postgres',
        python_callable=bulk_insert_prices
    )
    
    notify_success = PythonOperator(
        task_id='notify_completion',
        python_callable=send_slack_notification
    )
    
    fetch_prices >> validate_data >> load_to_db >> notify_success
```

### Data Engineer Checklist

- ✅ Data pipelines scheduled and monitored
- ✅ Quality checks before loading
- ✅ Failed pipelines alert on-call
- ✅ Data lineage documented
- ✅ Backfill procedures defined

---

## 8. Security Engineer Best Practices

### Best Practice: Authentication & Authorization

**OAuth2 + JWT implementation:**

```python
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# In route - check authorization
@app.get("/portfolios/{portfolio_id}")
async def get_portfolio(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter_by(id=portfolio_id).first()
    
    # Authorization check
    if portfolio.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this portfolio")
    
    return portfolio
```

### Best Practice: Input Validation (Prevent Injection)

```python
# ❌ NEVER EVER do this - SQL injection vulnerability!
def get_user_by_email(email: str):
    query = f"SELECT * FROM users WHERE email = '{email}'"
    return db.execute(query)

# ✅ ALWAYS use ORM or parameterized queries
def get_user_by_email(email: str):
    return db.query(User).filter(User.email == email).first()

# ✅ Pydantic validation catches malicious input
from pydantic import EmailStr, validator

class EmailInput(BaseModel):
    email: EmailStr  # Validates email format
    
    @validator('email')
    def sanitize_email(cls, v):
        return v.lower().strip()
```

### Best Practice: Secrets Management

```python
# ❌ BAD - secrets in code
API_KEY = "sk-1234567890abcdef"

# ✅ GOOD - environment variables
import os
API_KEY = os.getenv("YAHOO_FINANCE_API_KEY")
if not API_KEY:
    raise ValueError("YAHOO_FINANCE_API_KEY environment variable not set")

# ✅ BEST - Cloud Secret Manager
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

SUPABASE_KEY = get_secret("supabase-secret-key")
```

### Best Practice: Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/auth/login")
@limiter.limit("5/minute")  # Max 5 login attempts per minute
async def login(request: Request, credentials: LoginCredentials):
    # Login logic...
    pass

@app.get("/api/prices/{symbol}")
@limiter.limit("100/hour")  # Prevent API abuse
async def get_price(request: Request, symbol: str):
    # Fetch price...
    pass
```

### Security Checklist

- ✅ All passwords hashed (bcrypt, Argon2)
- ✅ JWT tokens with short expiry (< 15 min)
- ✅ HTTPS enforced (HSTS header)
- ✅ Rate limiting on all endpoints
- ✅ Input validation on all user input
- ✅ Secrets in Secret Manager, not env vars
- ✅ Security headers (CSP, X-Frame-Options)
- ✅ Regular dependency audits (`pip-audit`, `npm audit`)

---

## 9. Product Manager Best Practices

### Best Practice: User Story Format

**Template:**

```
As a [persona]
I want [action]
So that [benefit]

Acceptance Criteria:
- [ ] Criterion 1 (must be testable)
- [ ] Criterion 2

Definition of Done:
- [ ] Code reviewed by 2 engineers
- [ ] Unit tests written (coverage > 80%)
- [ ] Integration tests pass
- [ ] Deployed to staging
- [ ] QA sign-off
- [ ] Documentation updated
```

**Example:**

```
As a retail investor
I want to see my portfolio's time-weighted return (TWR)
So that I can evaluate my investment performance independent of deposits/withdrawals

Acceptance Criteria:
- [ ] TWR displayed on dashboard as percentage with 2 decimal places
- [ ] Calculation excludes impact of cash flows (deposits/withdrawals)
- [ ] Tooltip explains difference between TWR and MWR
- [ ] Metric updates in real-time when new transaction is added
- [ ] Shows "N/A" if insufficient data (< 2 data points)

Technical Notes:
- Formula: TWR = ∏(1 + r_i) - 1 where r_i is return for period i
- Backend endpoint: GET /portfolio/{id}/metrics
- Frontend component: MetricCard.tsx
```

### Best Practice: Feature Prioritization (RICE)

**Score = (Reach × Impact × Confidence) / Effort**

| Feature | Reach (users/mo) | Impact (1-3) | Confidence (%) | Effort (person-days) | RICE Score |
|---------|------------------|--------------|----------------|----------------------|------------|
| TWR/MWR calculation | 1000 | 3 | 100% | 5 | **600** |
| Tax loss harvesting | 500 | 2 | 50% | 13 | 38 |
| Portfolio rebalancing | 800 | 3 | 80% | 8 | 240 |
| Dividend tracking | 700 | 2 | 90% | 3 | **420** |

**Prioritize:** TWR/MWR (600), Dividend tracking (420), Rebalancing (240), Tax (38)

### Product Manager Checklist

- ✅ User stories written in standard format
- ✅ Acceptance criteria are testable
- ✅ Features prioritized using data (RICE/MoSCoW)
- ✅ Mockups/wireframes provided for UI changes
- ✅ Success metrics defined
- ✅ Stakeholders aligned on roadmap

---

## 10. Cross-Team Collaboration

### Git Workflow (Gitflow)

```
main ─────────────────────────────▶ Production (deployed)
      ↑               ↑
      merge           merge
develop ──────────────┴────────────▶ Staging (auto-deploy)
    ↑      ↑      ↑
    │      │      │
feature/ feature/ bugfix/
login    api      calc-error
```

**Branch naming conventions:**
- `feature/TICKET-123-short-description`
- `bugfix/TICKET-456-fix-calculation`
- `hotfix/critical-security-patch`

### Pull Request Guidelines

**PR Template:**

```markdown
## Description
Brief summary of what this PR does

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing Done
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Tested on staging

## Screenshots (if UI change)
[Paste screenshots here]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings/errors
```

**Review Requirements:**
- 2 approvals minimum
- CI/CD pipeline must pass (all tests green)
- No merge conflicts
- Branch up-to-date with target

### Communication Channels

**Slack Channels:**
- `#engineering` - General eng discussion
- `#frontend` - Frontend-specific
- `#backend` - Backend-specific
- `#devops` - Infrastructure
- `#qa-testing` - Test results, bugs
- `#incidents` - Production issues (urgent)

**Meetings:**
- **Daily Standup** (15 min): Yesterday, today, blockers
- **Sprint Planning** (2h, biweekly): Prioritize work for next sprint
- **Retrospective** (1h, biweekly): What went well, what didn't, improvements

### Documentation Standards

**README.md must include:**
```markdown
# Project Name

## Overview
Brief description

## Tech Stack
- Frontend: React + TypeScript
- Backend: FastAPI + Python
- Database: PostgreSQL 16

## Setup
1. Clone repo
2. Install dependencies
3. Run docker-compose up

## Architecture
[Link to architecture diagram]

## Contributing
[Link to CONTRIBUTING.md]

## License
```

**API Documentation:**
- Auto-generated OpenAPI/Swagger at `/docs`
- Example requests/responses
- Error codes table

---

## Conclusion

These best practices represent **production-grade standards** for building financial software in 2026. 

### Implementation Approach

**Don't try to adopt everything at once.** Follow this priority:

**Phase 1: Critical (Week 1-2)**
1. Security: Input validation, secrets management
2. Testing: Basic unit tests (>50% coverage)
3. Git: PR process, branch naming

**Phase 2: Important (Week 3-4)**
4. Backend: Service layer separation
5. Frontend: TypeScript types, TanStack Query
6. DevOps: CI/CD pipeline

**Phase 3: Nice-to-Have (Month 2+)**
7. Monitoring: Prometheus + Grafana
8. Documentation: ADRs, architecture diagrams
9. Advanced: E2E tests, load testing

### Remember

- **Practices serve the team**, not the other way around
- **Document deviations** - if you skip a practice, explain why
- **Review quarterly** - update this guide as technology evolves
- **Adapt to context** - smaller teams can combine roles

**The goal is sustainable, high-quality software delivery** - not perfect adherence to every practice.
