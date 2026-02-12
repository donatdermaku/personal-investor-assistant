# NEXUS Responsive Testing Checklist

## Breakpoint Rules Implemented

- `<768px` phone: single column content, bottom nav, Mission Control as bottom sheet (FAB trigger), holdings as expandable cards.
- `768px-1023px` tablet portrait: collapsible nav drawer (hamburger), content full width, Mission Control right drawer.
- `1024px-1199px` tablet landscape: 2-column shell (nav + content), Mission Control right drawer, charts may stack by page.
- `1200px-1439px` desktop: 3-column shell `200px | fluid | 320px`, Mission Control persistent.
- `>=1440px` desktop large: 3-column shell `220px | fluid | 360px`, Mission Control persistent.

## Spacing and Card Consistency

- Base spacing tokens: `8, 16, 24, 32, 48`.
- Card internal padding: `24px` desktop/tablet, `16px` phone.
- Grid gap: minimum `20px` (implemented as `20-24px` via responsive tokens).
- Touch targets: minimum `44x44`.

## Page-by-Page Layout Expectations

### Overview

- Metrics render in a 4-card grid on desktop, 2-up on tablet, stack on phone.
- Equity Curve is dominant and wider than the companion stats card on desktop.
- Holdings snapshot and diagnostics render full width below primary charts.

### Performance

- Top metrics in a 3-card row on tablet/desktop, stacked on phone.
- Primary row uses `Portfolio Value (2/3)` and `Drawdowns (1/3)` on desktop.
- Secondary row uses `Monthly Returns` and `Attribution` in 2-up.

### Risk

- Top metrics in 4-card grid desktop, reduced columns on smaller screens.
- Correlation Matrix is a large primary block.
- Bottom row uses `Rolling Volatility (2/3)` and `Risk Contribution (1/3)` on desktop.

### Holdings

- Desktop/tablet: table layout with sticky header.
- Tablet: price column hidden before desktop width and merged as subtext in value cell.
- Phone: expandable holding cards with key metrics first, details on expand.

## Device Width Verification Matrix

- `375px` (iPhone SE): phone behavior confirmed.
- `393px` (iPhone 14 Pro): phone behavior confirmed.
- `768px` (iPad Mini): tablet portrait behavior confirmed.
- `1024px` (iPad Pro): tablet landscape behavior confirmed.
- `1440px` (MacBook Air 13"): desktop large behavior confirmed.
- `1920px` desktop: desktop large behavior confirmed.
- `2560px` desktop: desktop large behavior confirmed.
- `3440px` ultrawide: desktop large behavior confirmed.

## Known Validation Caveats in This Environment

- `npm run build` fails in sandbox due blocked access to Google Fonts (`Outfit`, `Space Grotesk`), not due TS/JS compile errors in modified files.
- `npm run lint` currently fails due pre-existing issues in `web/src/lib/mock-data.ts` unrelated to this layout work.

## Performance Pass (Second Iteration)

- Added chart lazy mounting with `IntersectionObserver` to defer chart rendering until near viewport:
  - `web/src/components/nexus/LazyChart.tsx`
- Added responsive chart downsampling utility for lighter rendering on phones/tablets:
  - `web/src/lib/chartPerformance.ts`
  - `web/src/hooks/useMediaQuery.ts`
- Applied lazy + downsample behavior in:
  - `web/src/app/(routes)/overview/page.tsx`
  - `web/src/app/(routes)/performance/page.tsx`
  - `web/src/app/(routes)/risk/page.tsx`
- Added holdings virtualization path for large tables (`>140` rows), including overscan windowing and spacer rows:
  - `web/src/app/(routes)/holdings/page.tsx`
