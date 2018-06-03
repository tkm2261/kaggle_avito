--avito.dmt_item_periods
SELECT
item_id,
AVG(EXTRACT(DAYOFWEEK from activation_date)) as avg_activation_date_weekday,
AVG(EXTRACT(DAYOFWEEK from date_from)) as avg_date_from_weekday,
AVG(EXTRACT(DAYOFWEEK from date_to)) as avg_date_to_weekday,

AVG(DATE_DIFF(date_from, activation_date, DAY)) as avg_delay,
MIN(DATE_DIFF(date_from, activation_date, DAY)) as min_delay,
MAX(DATE_DIFF(date_from, activation_date, DAY)) as max_delay,
STDDEV(DATE_DIFF(date_from, activation_date, DAY)) as std_delay,
SUM(DATE_DIFF(date_from, activation_date, DAY)) as sum_delay,

AVG(DATE_DIFF(date_to, date_from, DAY)) as avg_duration,
MIN(DATE_DIFF(date_to, date_from, DAY)) as min_duration,
MAX(DATE_DIFF(date_to, date_from, DAY)) as max_duration,
STDDEV(DATE_DIFF(date_to, date_from, DAY)) as std_duration,
SUM(DATE_DIFF(date_to, date_from, DAY)) as sum_duration
FROM
(
  SELECT
*
FROM
`avito.periods_train` as p
UNION ALL
SELECT
*
FROM
`avito.periods_test` as p
)
GROUP BY
item_id

-- avito.train_test
SELECT
t.*,
avg_activation_date_weekday,
avg_date_from_weekday,
avg_date_to_weekday,

avg_delay,
min_delay,
max_delay,
std_delay,
sum_delay,

avg_duration,
min_duration,
max_duration,
std_duration,
sum_duration,

DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id order by t.activation_date), DAY) as avg_item_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id order by t.activation_date), DAY) as avg_user_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.region order by t.activation_date), DAY) as avg_region_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.city order by t.activation_date), DAY) as avg_city_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.parent_category_name order by t.activation_date), DAY) as avg_parent_category_name_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.category_name order by t.activation_date), DAY) as avg_category_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_seq_number order by t.activation_date), DAY) as avg_isn_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_type order by t.activation_date), DAY) as avg_usertype_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by CAST(t.image_top_1 as STRING) order by t.activation_date), DAY) as avg_it1_activation_span,

DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.region order by t.activation_date), DAY) as avg_user_region_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.city order by t.activation_date), DAY) as avg_user_city_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.parent_category_name order by t.activation_date), DAY) as avg_user_pcat_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.category_name order by t.activation_date), DAY) as avg_user_category_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.item_seq_number order by t.activation_date), DAY) as avg_user_isn_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, t.user_type order by t.activation_date), DAY) as avg_user_usertype_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.user_id, CAST(t.image_top_1 as STRING) order by t.activation_date), DAY) as avg_user_it1_activation_span,

DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.region order by t.activation_date), DAY) as avg_item_region_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.city order by t.activation_date), DAY) as avg_item_city_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.parent_category_name order by t.activation_date), DAY) as avg_item_pcat_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.category_name order by t.activation_date), DAY) as avg_item_category_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.item_seq_number order by t.activation_date), DAY) as avg_item_isn_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, t.user_type order by t.activation_date), DAY) as avg_item_usertype_activation_span,
DATE_DIFF(t.activation_date, LAG(t.activation_date, 1) OVER(partition by t.item_id, CAST(t.image_top_1 as STRING) order by t.activation_date), DAY) as avg_item_it1_activation_span
FROM
(
SELECT
data_id, deal_probability, t.item_id,user_id,region,city,parent_category_name,category_name,param_1,param_2,param_3,price,item_seq_number,t.activation_date,user_type,image,image_top_1,
EXTRACT(DAYOFWEEK from t.activation_date) as activation_weekday
FROM
`avito.train_notext` as t
UNION ALL
SELECT
data_id, null as deal_probability, t.item_id,user_id,region,city,parent_category_name,category_name,param_1,param_2,param_3,price,item_seq_number,t.activation_date,user_type,image,image_top_1,
EXTRACT(DAYOFWEEK from t.activation_date) as activation_weekday
FROM
`avito.test_notext` as t
) as t
LEFT OUTER JOIN
`avito.dmt_item_periods` as p
ON
t.item_id = p.item_id

-- avito.dmt_user
SELECT
  user_id,
  SUM(deal_probability) as sum_user_deal_probability,
  AVG(deal_probability) as avg_user_deal_probability,
  count(1) as cnt_user,
  count(distinct item_id) as uq_user_item,
  count(distinct region) as uq_user_region,
  count(distinct city) as uq_user_city,
  count(distinct parent_category_name) as uq_user_parent_category_name,
  count(distinct category_name) as uq_user_category_name,
  count(distinct param_1) as uq_user_param_1,
  count(distinct param_2) as uq_user_param_2,
  count(distinct param_3) as uq_user_param_3,
  AVG(price) as avg_user_price,
  MIN(price) as min_user_price,
  MAX(price) as max_user_price,
  STDDEV(price) as std_user_price,
  AVG(log(price + 1)) as avglog_user_price,

  count(distinct item_seq_number) as uq_user_item_seq_number,
  AVG(item_seq_number) as avg_user_item_seq_number,
  MIN(item_seq_number) as min_user_item_seq_number,
  MAX(item_seq_number) as max_user_item_seq_number,
  STDDEV(item_seq_number) as std_user_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_user_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_user_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_user_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_user_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_user_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_user_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_user_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_user_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_user_activation_7,

  count(distinct user_type) as uq_user_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_user_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_user_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_user_user_type_shop,

  count(distinct image_top_1) as uq_image_top_1,
  AVG(image_top_1) as avg_user_image_top_1,
  MIN(image_top_1) as min_user_image_top_1,
  MAX(image_top_1) as max_user_image_top_1,
  STDDEV(image_top_1) as std_user_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_user_image_top_1,


  AVG(avg_delay) as avg_user_delay,
  MIN(min_delay) as min_user_delay,
  MAX(max_delay) as max_user_delay,
  STDDEV(std_delay) as avg_user_std_delay,
  SUM(sum_delay) as sum_user_delay,

  AVG(avg_duration) as avg_user_duration,
  MIN(min_duration) as min_user_duration,
  MAX(max_duration) as max_user_duration,
  STDDEV(std_duration) as avg_user_std_duration,
  SUM(sum_duration) as sum_user_duration
FROM
  avito.train_test
GROUP BY
  user_id

-- avito.dmt_item
SELECT
  item_id,
  SUM(deal_probability) as sum_item_deal_probability,
  AVG(deal_probability) as avg_item_deal_probability,
  count(1) as cnt_item,
  count(distinct user_id) as uq_item_user,
  count(distinct region) as uq_item_region,
  count(distinct city) as uq_item_city,
  count(distinct parent_category_name) as uq_item_parent_category_name,
  count(distinct category_name) as uq_item_category_name,
  count(distinct param_1) as uq_item_param_1,
  count(distinct param_2) as uq_item_param_2,
  count(distinct param_3) as uq_item_param_3,
  AVG(price) as avg_item_price,
  MIN(price) as min_item_price,
  MAX(price) as max_item_price,
  STDDEV(price) as std_item_price,
  AVG(log(price + 1)) as avglog_item_price,

  count(distinct item_seq_number) as uq_item_item_seq_number,
  AVG(item_seq_number) as avg_item_item_seq_number,
  MIN(item_seq_number) as min_item_item_seq_number,
  MAX(item_seq_number) as max_item_item_seq_number,
  STDDEV(item_seq_number) as std_item_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_item_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_item_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_item_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_item_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_item_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_item_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_item_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_item_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_item_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_item_activation_span,

  count(distinct user_type) as uq_item_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_item_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_item_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_item_user_type_shop,
  count(distinct image_top_1) as uq_item_image_top_1,
  AVG(image_top_1) as avg_item_image_top_1,
  MIN(image_top_1) as min_item_image_top_1,
  MAX(image_top_1) as max_item_image_top_1,
  STDDEV(image_top_1) as std_item_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_item_image_top_1,

  AVG(avg_delay) as avg_item_delay,
  MIN(min_delay) as min_item_delay,
  MAX(max_delay) as max_item_delay,
  STDDEV(std_delay) as avg_item_std_delay,
  SUM(sum_delay) as sum_item_delay,

  AVG(avg_duration) as avg_item_duration,
  MIN(min_duration) as min_item_duration,
  MAX(max_duration) as max_item_duration,
  STDDEV(std_duration) as avg_item_std_duration,
  SUM(sum_duration) as sum_item_duration
FROM
  avito.train_test
GROUP BY
    item_id

-- avito.dmt_parent_category_name
SELECT
  parent_category_name,
  SUM(deal_probability) as sum_pcat_deal_probability,
  AVG(deal_probability) as avg_pcat_deal_probability,
  count(1) as cnt_pcat,
  count(distinct user_id) as uq_pcat_user,
  count(distinct item_id) as uq_pcat_item,
  count(distinct region) as uq_pcat_region,
  count(distinct city) as uq_pcat_city,
  count(distinct parent_category_name) as uq_pcat_parent_category_name,
  count(distinct category_name) as uq_pcat_category_name,
  count(distinct param_1) as uq_pcat_param_1,
  count(distinct param_2) as uq_pcat_param_2,
  count(distinct param_3) as uq_pcat_param_3,
  AVG(price) as avg_pcat_price,
  MIN(price) as min_pcat_price,
  MAX(price) as max_pcat_price,
  STDDEV(price) as std_pcat_price,
  AVG(log(price + 1)) as avglog_pcat_price,

  count(distinct item_seq_number) as uq_pcat_item_seq_number,
  AVG(item_seq_number) as avg_pcat_item_seq_number,
  MIN(item_seq_number) as min_pcat_item_seq_number,
  MAX(item_seq_number) as max_pcat_item_seq_number,
  STDDEV(item_seq_number) as std_pcat_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_pcat_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_pcat_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_pcat_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_pcat_activation_span,

  count(distinct user_type) as uq_pcat_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_pcat_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_pcat_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_pcat_user_type_shop,
  count(distinct image_top_1) as uq_pcat_image_top_1,
  AVG(image_top_1) as avg_pcat_image_top_1,
  MIN(image_top_1) as min_pcat_image_top_1,
  MAX(image_top_1) as max_pcat_image_top_1,
  STDDEV(image_top_1) as std_pcat_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_pcat_image_top_1,

  AVG(avg_delay) as avg_pcat_delay,
  MIN(min_delay) as min_pcat_delay,
  MAX(max_delay) as max_pcat_delay,
  STDDEV(std_delay) as avg_pcat_std_delay,
  SUM(sum_delay) as sum_pcat_delay,

  AVG(avg_duration) as avg_pcat_duration,
  MIN(min_duration) as min_pcat_duration,
  MAX(max_duration) as max_pcat_duration,
  STDDEV(std_duration) as avg_pcat_std_duration,
  SUM(sum_duration) as sum_pcat_duration
FROM
  avito.train_test
GROUP BY
    parent_category_name

-- avito.dmt_category_name
SELECT
  category_name,
  SUM(deal_probability) as sum_category_deal_probability,
  AVG(deal_probability) as avg_category_deal_probability,
  count(1) as cnt_category,
  count(distinct user_id) as uq_category_user,
  count(distinct item_id) as uq_category_item,
  count(distinct region) as uq_category_region,
  count(distinct city) as uq_category_city,
  count(distinct parent_category_name) as uq_category_parent_category_name,
  count(distinct category_name) as uq_category_category_name,
  count(distinct param_1) as uq_category_param_1,
  count(distinct param_2) as uq_category_param_2,
  count(distinct param_3) as uq_category_param_3,
  AVG(price) as avg_category_price,
  MIN(price) as min_category_price,
  MAX(price) as max_category_price,
  STDDEV(price) as std_category_price,
  AVG(log(price + 1)) as avglog_category_price,

  count(distinct item_seq_number) as uq_category_item_seq_number,
  AVG(item_seq_number) as avg_category_item_seq_number,
  MIN(item_seq_number) as min_category_item_seq_number,
  MAX(item_seq_number) as max_category_item_seq_number,
  STDDEV(item_seq_number) as std_category_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_category_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_category_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_category_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_category_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_category_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_category_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_category_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_category_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_category_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_category_activation_span,

  count(distinct user_type) as uq_category_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_category_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_category_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_category_user_type_shop,
  count(distinct image_top_1) as uq_item_image_top_1,
  AVG(image_top_1) as avg_category_image_top_1,
  MIN(image_top_1) as min_category_image_top_1,
  MAX(image_top_1) as max_category_image_top_1,
  STDDEV(image_top_1) as std_category_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_category_image_top_1,

  AVG(avg_delay) as avg_category_delay,
  MIN(min_delay) as min_category_delay,
  MAX(max_delay) as max_category_delay,
  STDDEV(std_delay) as avg_category_std_delay,
  SUM(sum_delay) as sum_category_delay,

  AVG(avg_duration) as avg_category_duration,
  MIN(min_duration) as min_category_duration,
  MAX(max_duration) as max_category_duration,
  STDDEV(std_duration) as avg_category_std_duration,
  SUM(sum_duration) as sum_category_duration
FROM
  avito.train_test
GROUP BY
    category_name


-- avito.dmt_region
SELECT
  region,
  SUM(deal_probability) as sum_region_deal_probability,
  AVG(deal_probability) as avg_region_deal_probability,
  count(1) as cnt_region,
  count(distinct user_id) as uq_region_user,
  count(distinct item_id) as uq_region_item,
  count(distinct region) as uq_region_region,
  count(distinct city) as uq_region_city,
  count(distinct parent_category_name) as uq_region_parent_category_name,
  count(distinct category_name) as uq_region_category_name,
  count(distinct param_1) as uq_region_param_1,
  count(distinct param_2) as uq_region_param_2,
  count(distinct param_3) as uq_region_param_3,
  AVG(price) as avg_region_price,
  MIN(price) as min_region_price,
  MAX(price) as max_region_price,
  STDDEV(price) as std_region_price,
  AVG(log(price + 1)) as avglog_region_price,

  count(distinct item_seq_number) as uq_region_item_seq_number,
  AVG(item_seq_number) as avg_region_item_seq_number,
  MIN(item_seq_number) as min_region_item_seq_number,
  MAX(item_seq_number) as max_region_item_seq_number,
  STDDEV(item_seq_number) as std_region_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_region_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_region_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_region_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_region_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_region_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_region_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_region_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_region_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_region_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_region_activation_span,

  count(distinct user_type) as uq_region_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_region_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_region_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_region_user_type_shop,
  count(distinct image_top_1) as uq_region_image_top_1,
  AVG(image_top_1) as avg_region_image_top_1,
  MIN(image_top_1) as min_region_image_top_1,
  MAX(image_top_1) as max_region_image_top_1,
  STDDEV(image_top_1) as std_region_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_region_image_top_1,

  AVG(avg_delay) as avg_region_delay,
  MIN(min_delay) as min_region_delay,
  MAX(max_delay) as max_region_delay,
  STDDEV(std_delay) as avg_region_std_delay,
  SUM(sum_delay) as sum_region_delay,

  AVG(avg_duration) as avg_region_duration,
  MIN(min_duration) as min_region_duration,
  MAX(max_duration) as max_region_duration,
  STDDEV(std_duration) as avg_region_std_duration,
  SUM(sum_duration) as sum_region_duration
FROM
  avito.train_test
GROUP BY
    region

-- avito.dmt_city
SELECT
  city,
  SUM(deal_probability) as sum_city_deal_probability,
  AVG(deal_probability) as avg_city_deal_probability,
  count(1) as cnt_city,
  count(distinct user_id) as uq_city_user,
  count(distinct item_id) as uq_city_item,
  count(distinct region) as uq_city_region,
  count(distinct city) as uq_city_city,
  count(distinct parent_category_name) as uq_city_parent_category_name,
  count(distinct category_name) as uq_city_category_name,
  count(distinct param_1) as uq_city_param_1,
  count(distinct param_2) as uq_city_param_2,
  count(distinct param_3) as uq_city_param_3,
  AVG(price) as avg_city_price,
  MIN(price) as min_city_price,
  MAX(price) as max_city_price,
  STDDEV(price) as std_city_price,
  AVG(log(price + 1)) as avglog_city_price,

  count(distinct item_seq_number) as uq_city_item_seq_number,
  AVG(item_seq_number) as avg_city_item_seq_number,
  MIN(item_seq_number) as min_city_item_seq_number,
  MAX(item_seq_number) as max_city_item_seq_number,
  STDDEV(item_seq_number) as std_city_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_city_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_city_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_city_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_city_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_city_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_city_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_city_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_city_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_city_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_city_activation_span,

  count(distinct user_type) as uq_city_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_city_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_city_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_city_user_type_shop,
  count(distinct image_top_1) as uq_city_image_top_1,
  AVG(image_top_1) as avg_city_image_top_1,
  MIN(image_top_1) as min_city_image_top_1,
  MAX(image_top_1) as max_city_image_top_1,
  STDDEV(image_top_1) as std_city_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_city_image_top_1,

  AVG(avg_delay) as avg_city_delay,
  MIN(min_delay) as min_city_delay,
  MAX(max_delay) as max_city_delay,
  STDDEV(std_delay) as avg_city_std_delay,
  SUM(sum_delay) as sum_city_delay,

  AVG(avg_duration) as avg_city_duration,
  MIN(min_duration) as min_city_duration,
  MAX(max_duration) as max_city_duration,
  STDDEV(std_duration) as avg_city_std_duration,
  SUM(sum_duration) as sum_city_duration
FROM
  avito.train_test
GROUP BY
    city

-- avito.dmt_item_seq_number
SELECT
  item_seq_number,
  SUM(deal_probability) as sum_isn_deal_probability,
  AVG(deal_probability) as avg_isn_deal_probability,
  count(1) as cnt_isn,
  count(distinct user_id) as uq_isn_user,
  count(distinct item_id) as uq_isn_item,
  count(distinct region) as uq_isn_region,
  count(distinct city) as uq_isn_city,
  count(distinct parent_category_name) as uq_isn_parent_category_name,
  count(distinct category_name) as uq_isn_category_name,
  count(distinct param_1) as uq_isn_param_1,
  count(distinct param_2) as uq_isn_param_2,
  count(distinct param_3) as uq_isn_param_3,
  AVG(price) as avg_isn_price,
  MIN(price) as min_isn_price,
  MAX(price) as max_isn_price,
  STDDEV(price) as std_isn_price,
  AVG(log(price + 1)) as avglog_isn_price,

  count(distinct item_seq_number) as uq_isn_item_seq_number,
  AVG(item_seq_number) as avg_isn_item_seq_number,
  MIN(item_seq_number) as min_isn_item_seq_number,
  MAX(item_seq_number) as max_isn_item_seq_number,
  STDDEV(item_seq_number) as std_isn_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_isn_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_isn_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_isn_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_isn_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_isn_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_isn_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_isn_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_isn_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_isn_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_isn_activation_span,

  count(distinct user_type) as uq_isn_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_isn_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_isn_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_isn_user_type_shop,
  count(distinct image_top_1) as uq_isn_image_top_1,
  AVG(image_top_1) as avg_isn_image_top_1,
  MIN(image_top_1) as min_isn_image_top_1,
  MAX(image_top_1) as max_isn_image_top_1,
  STDDEV(image_top_1) as std_isn_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_isn_image_top_1,

  AVG(avg_delay) as avg_isn_delay,
  MIN(min_delay) as min_isn_delay,
  MAX(max_delay) as max_isn_delay,
  STDDEV(std_delay) as avg_isn_std_delay,
  SUM(sum_delay) as sum_isn_delay,

  AVG(avg_duration) as avg_isn_duration,
  MIN(min_duration) as min_isn_duration,
  MAX(max_duration) as max_isn_duration,
  STDDEV(std_duration) as avg_isn_std_duration,
  SUM(sum_duration) as sum_isn_duration
FROM
  avito.train_test
GROUP BY
    item_seq_number

-- avito.dmt_user_type
SELECT
  user_type,
  SUM(deal_probability) as sum_usertype_deal_probability,
  AVG(deal_probability) as avg_usertype_deal_probability,
  count(1) as cnt_usertype,
  count(distinct user_id) as uq_usertype_user,
  count(distinct item_id) as uq_usertype_item,
  count(distinct region) as uq_usertype_region,
  count(distinct city) as uq_usertype_city,
  count(distinct parent_category_name) as uq_usertype_parent_category_name,
  count(distinct category_name) as uq_usertype_category_name,
  count(distinct param_1) as uq_usertype_param_1,
  count(distinct param_2) as uq_usertype_param_2,
  count(distinct param_3) as uq_usertype_param_3,
  AVG(price) as avg_usertype_price,
  MIN(price) as min_usertype_price,
  MAX(price) as max_usertype_price,
  STDDEV(price) as std_usertype_price,
  AVG(log(price + 1)) as avglog_usertype_price,

  count(distinct item_seq_number) as uq_usertype_item_seq_number,
  AVG(item_seq_number) as avg_usertype_item_seq_number,
  MIN(item_seq_number) as min_usertype_item_seq_number,
  MAX(item_seq_number) as max_usertype_item_seq_number,
  STDDEV(item_seq_number) as std_usertype_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_usertype_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_usertype_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_usertype_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_usertype_activation_span,

  count(distinct user_type) as uq_usertype_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_usertype_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_usertype_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_usertype_user_type_shop,
  count(distinct image_top_1) as uq_usertype_image_top_1,
  AVG(image_top_1) as avg_usertype_image_top_1,
  MIN(image_top_1) as min_usertype_image_top_1,
  MAX(image_top_1) as max_usertype_image_top_1,
  STDDEV(image_top_1) as std_usertype_image_top_1,
  AVG(log(image_top_1 + 1)) as avglog_usertype_image_top_1,

  AVG(avg_delay) as avg_usertype_delay,
  MIN(min_delay) as min_usertype_delay,
  MAX(max_delay) as max_usertype_delay,
  STDDEV(std_delay) as avg_usertype_std_delay,
  SUM(sum_delay) as sum_usertype_delay,

  AVG(avg_duration) as avg_usertype_duration,
  MIN(min_duration) as min_usertype_duration,
  MAX(max_duration) as max_usertype_duration,
  STDDEV(std_duration) as avg_usertype_std_duration,
  SUM(sum_duration) as sum_usertype_duration
FROM
  avito.train_test
GROUP BY
    user_type

-- avito.dmt_image_top_1
SELECT
  image_top_1,
  SUM(deal_probability) as sum_im1_deal_probability,
  AVG(deal_probability) as avg_im1_deal_probability,
  count(1) as cnt_im1,
  count(distinct user_id) as uq_im1_user,
  count(distinct item_id) as uq_im1_item,
  count(distinct region) as uq_im1_region,
  count(distinct city) as uq_im1_city,
  count(distinct parent_category_name) as uq_im1_parent_category_name,
  count(distinct category_name) as uq_im1_category_name,
  count(distinct param_1) as uq_im1_param_1,
  count(distinct param_2) as uq_im1_param_2,
  count(distinct param_3) as uq_im1_param_3,
  AVG(price) as avg_im1_price,
  MIN(price) as min_im1_price,
  MAX(price) as max_im1_price,
  STDDEV(price) as std_im1_price,
  AVG(log(price + 1)) as avglog_im1_price,

  count(distinct item_seq_number) as uq_im1_item_seq_number,
  AVG(item_seq_number) as avg_im1_item_seq_number,
  MIN(item_seq_number) as min_im1_item_seq_number,
  MAX(item_seq_number) as max_im1_item_seq_number,
  STDDEV(item_seq_number) as std_im1_item_seq_number,
  AVG(log(item_seq_number + 1)) as avglog_im1_item_seq_number,

  count(distinct EXTRACT(DAYOFWEEK from activation_date)) as uq_im1_activation_weekday,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 1 THEN 1 ELSE 0 END) as rate_uq_im1_activation_1,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 2 THEN 1 ELSE 0 END) as rate_uq_im1_activation_2,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 3 THEN 1 ELSE 0 END) as rate_uq_im1_activation_3,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 4 THEN 1 ELSE 0 END) as rate_uq_im1_activation_4,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 5 THEN 1 ELSE 0 END) as rate_uq_im1_activation_5,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 6 THEN 1 ELSE 0 END) as rate_uq_im1_activation_6,
  AVG(CASE WHEN EXTRACT(DAYOFWEEK from activation_date) = 7 THEN 1 ELSE 0 END) as rate_uq_im1_activation_7,

  --AVG(DATE_DIFF(activation_date, LAG(activation_date, 1), DAY) as avg_im1_activation_span,

  count(distinct user_type) as uq_im1_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_im1_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_im1_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_im1_user_type_shop,

  AVG(avg_delay) as avg_im1_delay,
  MIN(min_delay) as min_im1_delay,
  MAX(max_delay) as max_im1_delay,
  STDDEV(std_delay) as avg_im1_std_delay,
  SUM(sum_delay) as sum_im1_delay,

  AVG(avg_duration) as avg_im1_duration,
  MIN(min_duration) as min_im1_duration,
  MAX(max_duration) as max_im1_duration,
  STDDEV(std_duration) as avg_im1_std_duration,
  SUM(sum_duration) as sum_im1_duration
FROM
  avito.train_test
GROUP BY
    image_top_1

-- avito.train_test2
SELECT
t.*,
i.*, u.*, c.*, ct.*, isn.*, pc.*, r.*, ut.*, it1.*
FROM
avito.train_test as t
LEFT OUTER JOIN avito.dmt_item_periods as p
ON t.item_id = p.item_id
LEFT OUTER JOIN avito.dmt_item as i
ON t.item_id = i.item_id
LEFT OUTER JOIN avito.dmt_user as u
ON t.user_id = u.user_id
LEFT OUTER JOIN avito.dmt_category_name as c
ON t.category_name = c.category_name
LEFT OUTER JOIN avito.dmt_city as ct
ON t.city = ct.city
LEFT OUTER JOIN avito.dmt_item_seq_number as isn
ON t.item_seq_number = isn.item_seq_number
LEFT OUTER JOIN avito.dmt_parent_category_name as pc
ON t.parent_category_name = pc.parent_category_name
LEFT OUTER JOIN avito.dmt_region as r
ON t.region = r.region
LEFT OUTER JOIN avito.dmt_user_type as ut
ON t.user_type = ut.user_type
LEFT OUTER JOIN avito.dmt_image_top_1 as it1
ON t.image_top_1 = it1.image_top_1

-- avito.dmt_train
SELECT
*
FROM
`avito.train_test2`
WHERE
t_deal_probability is not null

-- avito.dmt_test
SELECT
*
FROM
`avito.train_test2`
WHERE
t_deal_probability is null
