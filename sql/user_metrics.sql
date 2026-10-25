-- One row per assigned user, with experiment-period and pre-period totals.
-- Days 0-13 are the experiment; days -14 to -1 are the pre-period used for CUPED.
-- LEFT JOINs keep assigned users who never opened the app. Dropping them (an INNER JOIN)
-- would overstate saves per user and can bias the comparison if treatment changes who returns.
WITH experiment AS (
    SELECT
        user_id,
        COUNT(*)                                       AS sessions,
        SUM(impressions)                               AS impressions,
        SUM(saves)                                     AS saves,
        SUM(hides)                                     AS hides,
        SUM(CASE WHEN day < 7  THEN saves ELSE 0 END)  AS saves_week1,
        SUM(CASE WHEN day >= 7 THEN saves ELSE 0 END)  AS saves_week2
    FROM sessions
    WHERE day BETWEEN 0 AND 13
    GROUP BY user_id
),
pre AS (
    SELECT
        user_id,
        COUNT(*)    AS pre_sessions,
        SUM(saves)  AS pre_saves,
        SUM(hides)  AS pre_hides
    FROM sessions
    WHERE day BETWEEN -14 AND -1
    GROUP BY user_id
)
SELECT
    a.user_id,
    a.variant,
    a.app_version,
    COALESCE(e.sessions, 0)     AS sessions,
    COALESCE(e.impressions, 0)  AS impressions,
    COALESCE(e.saves, 0)        AS saves,
    COALESCE(e.hides, 0)        AS hides,
    COALESCE(e.saves_week1, 0)  AS saves_week1,
    COALESCE(e.saves_week2, 0)  AS saves_week2,
    COALESCE(p.pre_sessions, 0) AS pre_sessions,
    COALESCE(p.pre_saves, 0)    AS pre_saves,
    COALESCE(p.pre_hides, 0)    AS pre_hides
FROM assignments AS a
LEFT JOIN experiment AS e ON e.user_id = a.user_id
LEFT JOIN pre        AS p ON p.user_id = a.user_id;
