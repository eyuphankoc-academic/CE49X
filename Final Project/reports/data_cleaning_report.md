# Data Cleaning Report

## FIRMS Thermal Data
- Raw rows processed: 5,988,583
- Clean rows retained: 5,988,583
- Daily region summary rows: 8,183
- Date range: 2024-01-01 to 2026-05-24

## News Data
- Raw rows processed: 92,062
- Clean rows retained: 85,697
- Date range: 2024-01-01 to 2026-05-24

## Modeling Summary
- Daily monitoring rows: 8,183
- Days with nearby conflict news: 4,073

## Cleaning Choices
- Removed low-confidence FIRMS detections (`l` / `low`) and invalid geographic, date, brightness, or FRP values.
- Combined standard processed FIRMS data with near-real-time FIRMS data for April-May 2026.
- Removed duplicate news URLs and duplicate same-region/same-date titles.
- Created daily region-level summaries to make notebook analysis and machine learning manageable.