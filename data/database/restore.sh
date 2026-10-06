#!/usr/bin/env bash
# Restores the Build4All research database into PostgreSQL.
# Usage:  ./restore.sh "postgresql://user:password@host:5432/dbname"
set -euo pipefail
URL="${1:?give a PostgreSQL connection URL}"
cd "$(dirname "$0")"
psql "$URL" -v ON_ERROR_STOP=1 -f schema.sql
for t in categories customers products shipping_methods coupons tax_rules orders order_items payments product_events; do
  psql "$URL" -v ON_ERROR_STOP=1 -c "\copy build4all_research.$t FROM '$t.csv' WITH (FORMAT csv, HEADER true)"
done
psql "$URL" -c "SELECT 'orders' t, count(*) FROM build4all_research.orders UNION ALL SELECT 'order_items', count(*) FROM build4all_research.order_items UNION ALL SELECT 'product_events', count(*) FROM build4all_research.product_events;"
