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
  count(distinct image_top_1) as uq_image_top_1
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
  count(distinct image_top_1) as uq_item_image_top_1
FROM
  avito.train_test
GROUP BY
    item_id


-- avito.train_test2
SELECT
t.data_id as data_id,
t.deal_probability as deal_probability,
t.item_id as item_id,
t.user_id as user_id,
t.region as region,
t.city as city,
t.parent_category_name as parent_category_name,
t.category_name as category_name,
t.param_1 as param_1,
t.param_2 as param_2,
t.param_3 as param_3,
t.price as price,
t.item_seq_number as item_seq_number,
t.activation_date as activation_date,
t.user_type as user_type,
t.image as image,
t.image_top_1 as image_top_1,
t.activation_weekday as activation_weekday,
t.avg_activation_date_weekday as avg_activation_date_weekday,
t.avg_date_from_weekday as avg_date_from_weekday,
t.avg_date_to_weekday as avg_date_to_weekday,
t.avg_delay as avg_delay,
t.min_delay as min_delay,
t.max_delay as max_delay,
t.std_delay as std_delay,
t.sum_delay as sum_delay,
t.avg_duration as avg_duration,
t.min_duration as min_duration,
t.max_duration as max_duration,
t.std_duration as std_duration,
t.sum_duration as sum_duration,

i.cnt_item,
i.uq_item_user,
i.uq_item_region,
i.uq_item_city,
i.uq_item_parent_category_name,
i.uq_item_category_name,
i.uq_item_param_1,
i.uq_item_param_2,
i.uq_item_param_3,
i.avg_item_price,
i.min_item_price,
i.max_item_price,
i.std_item_price,
i.avglog_item_price,
i.uq_item_item_seq_number,
i.avg_item_item_seq_number,
i.min_item_item_seq_number,
i.max_item_item_seq_number,
i.std_item_item_seq_number,
i.avglog_item_item_seq_number,
i.uq_item_activation_weekday,
i.rate_uq_item_activation_1,
i.rate_uq_item_activation_2,
i.rate_uq_item_activation_3,
i.rate_uq_item_activation_4,
i.rate_uq_item_activation_5,
i.rate_uq_item_activation_6,
i.rate_uq_item_activation_7,
i.uq_item_user_type,
i.rate_uq_item_user_type_private,
i.rate_uq_item_user_type_company,
i.rate_uq_item_user_type_shop,
i.uq_item_image_top_1,
u.cnt_user,
u.uq_user_item,
u.uq_user_region,
u.uq_user_city,
u.uq_user_parent_category_name,
u.uq_user_category_name,
u.uq_user_param_1,
u.uq_user_param_2,
u.uq_user_param_3,
u.avg_user_price,
u.min_user_price,
u.max_user_price,
u.std_user_price,
u.avglog_user_price,
u.uq_user_item_seq_number,
u.avg_user_item_seq_number,
u.min_user_item_seq_number,
u.max_user_item_seq_number,
u.std_user_item_seq_number,
u.avglog_user_item_seq_number,
u.uq_user_activation_weekday,
u.rate_uq_user_activation_1,
u.rate_uq_user_activation_2,
u.rate_uq_user_activation_3,
u.rate_uq_user_activation_4,
u.rate_uq_user_activation_5,
u.rate_uq_user_activation_6,
u.rate_uq_user_activation_7,
u.uq_user_user_type,
u.rate_uq_user_user_type_private,
u.rate_uq_user_user_type_company,
u.rate_uq_user_user_type_shop,
u.uq_image_top_1,
p.avg_activation_date_weekday,
p.avg_date_from_weekday,
p.avg_date_to_weekday,
p.avg_delay,
p.min_delay,
p.max_delay,
p.std_delay,
p.avg_duration,
p.min_duration,
p.max_duration,
p.std_duration

FROM
'avito.train_test' as t
LEFT OUTER JOIN `avito.dmt_item_periods` as p
ON t.item_id = p.item_id
LEFT OUTER JOIN `avito.dmt_item` as i
ON t.item_id = i.item_id
LEFT OUTER JOIN `avito.dmt_user` as u
ON t.user_id = i.user_id

-- avito.dmt_train
SELECT
*
FROM
`avito.train_test2`
WHERE
deal_probability is not null

-- avito.dmt_test
SELECT
*
FROM
`avito.train_test2`
WHERE
deal_probability is null
