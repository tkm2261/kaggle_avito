-- avito.dmt_user_region
SELECT
  user_id,
  region,
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
  user_id, region


-- avito.dmt_user_city
SELECT
  user_id,
  city,
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
  user_id, city


-- avito.dmt_user_pcat
SELECT
  user_id,
  parent_category_name,
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
  user_id, parent_category_name


-- avito.dmt_user_category
SELECT
  user_id,
  category_name,
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
  user_id, category_name

-- avito.dmt_user_isn
SELECT
  user_id,
  item_seq_number,
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
  user_id, item_seq_number

-- avito.dmt_user_usertype
SELECT
  user_id,
  user_type,
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
  user_id, user_type

-- avito.dmt_user_it1
SELECT
  user_id,
  image_top_1,
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
  user_id, image_top_1

###################
-- avito.dmt_item_region
SELECT
  item_id,
  region,
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
  item_id, region


-- avito.dmt_item_city
SELECT
  item_id,
  city,
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
  item_id, city


-- avito.dmt_item_pcat
SELECT
  item_id,
  parent_category_name,
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
  item_id, parent_category_name


-- avito.dmt_item_category
SELECT
  item_id,
  category_name,
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
  item_id, category_name

-- avito.dmt_item_isn
SELECT
  item_id,
  item_seq_number,
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
  item_id, item_seq_number

-- avito.dmt_item_usertype
SELECT
  item_id,
  user_type,
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
  item_id, user_type

-- avito.dmt_item_it1
SELECT
  item_id,
  image_top_1,
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
  item_id, image_top_1


-- avito.train_test2
SELECT
t.*,
i.*, u.*, c.*, ct.*, isn.*, pc.*, r.*, ut.*, it1.*,
ur.*, uc.*, up.*, uca.*, ui.*, uu.*, uit.*,
ir.*, ic.*, ip.*, ica.*, ii.*, iu.*, iit.*
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

LEFT OUTER JOIN avito.dmt_user_region as ur
ON t.user_id = ur.user_id AND t.region = ur.region
LEFT OUTER JOIN avito.dmt_user_city as uc
ON t.user_id = uc.user_id AND t.city = uc.city
LEFT OUTER JOIN avito.dmt_user_pcat as up
ON t.user_id = up.user_id AND t.parent_category_name = up.parent_category_name
LEFT OUTER JOIN avito.dmt_user_category as uca
ON t.user_id = uca.user_id AND t.category_name = uca.category_name
LEFT OUTER JOIN avito.dmt_user_isn as ui
ON t.user_id = ui.user_id AND t.item_seq_number = ui.item_seq_number
LEFT OUTER JOIN avito.dmt_user_usertype as uu
ON t.user_id = uu.user_id AND t.user_type = uu.user_type
LEFT OUTER JOIN avito.dmt_user_it1 as uit
ON t.user_id = uit.user_id AND t.image_top_1 = uit.image_top_1

LEFT OUTER JOIN avito.dmt_item_region as ir
ON t.item_id = ir.item_id AND t.region = ir.region
LEFT OUTER JOIN avito.dmt_item_city as ic
ON t.item_id = ic.item_id AND t.city = ic.city
LEFT OUTER JOIN avito.dmt_item_pcat as ip
ON t.item_id = ip.item_id AND t.parent_category_name = ip.parent_category_name
LEFT OUTER JOIN avito.dmt_item_category as ica
ON t.item_id = ica.item_id AND t.category_name = ica.category_name
LEFT OUTER JOIN avito.dmt_item_isn as ii
ON t.item_id = ii.item_id AND t.item_seq_number = ii.item_seq_number
LEFT OUTER JOIN avito.dmt_item_usertype as iu
ON t.item_id = iu.item_id AND t.user_type = iu.user_type
LEFT OUTER JOIN avito.dmt_item_it1 as iit
ON t.item_id = iit.item_id AND t.image_top_1 = iit.image_top_1
