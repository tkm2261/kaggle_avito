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
AVG(DATE_DIFF(date_to, date_from, DAY)) as avg_duration,
MIN(DATE_DIFF(date_to, date_from, DAY)) as min_duration,
MAX(DATE_DIFF(date_to, date_from, DAY)) as max_duration,
STDDEV(DATE_DIFF(date_to, date_from, DAY)) as std_duration
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

-- avito.dmt_user
SELECT
  user_id,
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

  AVG(delay) as avg_user_delay,
  MIN(delay) as min_user_delay,
  MAX(delay) as max_user_delay,
  STDDEV(delay) as std_user_delay,

  AVG(duration) as avg_user_duration,
  MIN(duration) as min_user_duration,
  MAX(duration) as max_user_duration,
  STDDEV(duration) as std_user_duration

FROM
  avito.train_test
GROUP BY
  user_id


-- avito.dmt_item
SELECT
  item_id,
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

  count(distinct user_type) as uq_item_user_type,
  AVG(CASE WHEN user_type = 'Private' THEN 1 ELSE 0 END) as rate_uq_item_user_type_private,
  AVG(CASE WHEN user_type = 'Company' THEN 1 ELSE 0 END) as rate_uq_item_user_type_company,
  AVG(CASE WHEN user_type = 'Shop' THEN 1 ELSE 0 END) as rate_uq_item_user_type_shop,
  count(distinct image_top_1) as uq_item_image_top_1,
  AVG(delay) as avg_item_delay,
  MIN(delay) as min_item_delay,
  MAX(delay) as max_item_delay,
  STDDEV(delay) as std_item_delay,

  AVG(duration) as avg_item_duration,
  MIN(duration) as min_item_duration,
  MAX(duration) as max_item_duration,
  STDDEV(duration) as std_item_duration
FROM
  avito.train_test
GROUP BY
    item_id
