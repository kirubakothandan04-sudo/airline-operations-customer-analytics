-- Airline Customer Segmentation Analysis
-- Step 1: Satisfaction by travel class and travel purpose

SELECT
    travel_class,
    travel_type,
    COUNT(*) AS total_passengers,
    SUM(
        CASE
            WHEN satisfaction = 'satisfied' THEN 1
            ELSE 0
        END
    ) AS satisfied_passengers,
    ROUND(
        AVG(satisfaction = 'satisfied') * 100,
        2
    ) AS satisfaction_rate
FROM passengers
GROUP BY travel_class, travel_type
ORDER BY satisfaction_rate ASC;