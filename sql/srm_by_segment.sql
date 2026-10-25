-- Assigned users per variant within each app version, with each variant's share of its segment.
-- An overall SRM says something is wrong; the segment split usually says where.
SELECT
    app_version,
    variant,
    COUNT(*) AS users,
    ROUND(1.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY app_version), 4) AS share
FROM assignments
GROUP BY app_version, variant
ORDER BY app_version, variant;
