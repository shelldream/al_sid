-- 以下脚本是KNN防碰撞的脚本（对最后一级码本ID，此处为3级码本demo，防碰撞数目设置在5）
-- 如果要进行随机防碰撞：将sorted_index_lv3_col 的生成方式改为随机即可，如bigint(rand()*8192 % 8192))


-- 存储物品与代码簿的映射及索引信息（结果表）
CREATE TABLE IF NOT EXISTS item_codebook_info
(
    item_id          BIGINT    comment '商品id'
    ,origin_codebook STRING    comment '原始码本ID'
    ,codebook        STRING    comment '防碰撞后码本ID'
    ,index           BIGINT    comment '防碰撞后的编号（如果防碰撞设置为5，则是1~5）'
)
LIFECYCLE 360
;

-- ========================================--
-- 1.当得到了新的模型，先进行对全局的推理，得到raw_data表，字段包括item_id和codebook_index
-- 如果是运行了'infer_SID.py'的代码，输出结果的两个字段就是代表item_id和codebook_index
-- 具体的数据会例如：['15615', '111,222,333']

-- 2.进行初次防碰撞
INSERT OVERWRITE TABLE item_codebook_info
SELECT  item_id
        ,codebook_index
        ,codebook_index
        ,ROW_NUMBER() OVER (PARTITION BY codebook_index ORDER BY rand() ) AS num
FROM    raw_data
;

-- 3.截断保留前5个索引的数据（即每个语义ID保留随机五个，剩下的商品开始进行防碰撞）
INSERT OVERWRITE TABLE item_codebook_info
SELECT  *
FROM    item_codebook_info
WHERE   index <= 5
;

-- ========================================--
-- 0.上述未分配的商品，需要计算它们的KNN排序sorted_index

-- 包含item和对应3级码本ID序的表 item_id, codebook_index, sorted_index, priority
-- 1.对三级排序数据，去掉当前已经分配过的商品
INSERT OVERWRITE TABLE sorted_index_lv3
SELECT  *
FROM    sorted_index_lv3
LEFT ANTI JOIN  (
                    SELECT  item_id
                    FROM    item_codebook_info
                ) b
ON      a.item_id = b.item_id
;


-- 2.生成三级排序索引表
INSERT OVERWRITE TABLE sorted_index_lv3_col
SELECT  TRANS_ARRAY(2,',',a.item_id,codebook_index,SPLIT_PART(sorted_index,',',2,201),'1,2,3,...,200') AS (item_id,codebook_index,sorted_index,priority)
FROM    sorted_index_lv3 a
;

-- 循环开始
-- 3.过滤已满载的代码簿（逻辑：去掉已分配商品、保留未满5个物品的代码簿）
INSERT OVERWRITE TABLE sorted_index_lv3_col
SELECT  a.item_id
        ,a.codebook_index
        ,a.sorted_index
        ,a.priority
FROM    sorted_index_lv3_col a
LEFT ANTI JOIN  (
                    SELECT  item_id
                    FROM    item_codebook_info
                ) b
ON      a.item_id = b.item_id
LEFT ANTI JOIN  (
                    SELECT  codebook
                    FROM    (
                                SELECT  codebook
                                        ,COUNT(*) AS c
                                FROM    item_codebook_info
                                GROUP BY codebook
                            )
                    WHERE   c >= 5
                ) b
ON      a.sorted_index = b.codebook
;

-- 4.按优先级进行分配
INSERT OVERWRITE TABLE candidate_assignment
SELECT  item_id
        ,codebook_index AS origin_cate
        ,sorted_index AS assigned_cate
        ,priority
FROM    (
            SELECT  item_id
                    ,codebook_index
                    ,sorted_index
                    ,priority
                    ,ROW_NUMBER() OVER (PARTITION BY codebook_index,priority ORDER BY rand(int(priority)) ) AS num --对于每个待分配语义ID，根据对应商品的分配优先级来截断5个
            FROM    sorted_index_lv3_col
        )
WHERE   num <= 5
;

-- 5.生成最终分配结果
INSERT OVERWRITE TABLE final_assignment
SELECT  assigned_cate
        ,origin_cate
        ,item_id
        ,priority
        ,rank
FROM    (
            SELECT  assigned_cate
                    ,origin_cate
                    ,BIGINT(item_id) AS item_id
                    ,BIGINT(priority) AS priority
                    ,ROW_NUMBER() OVER (PARTITION BY assigned_cate ORDER BY rank ) AS rank --对于每个待分配语义ID，保留优先级最高的5个商品
            FROM    (
                        SELECT  assigned_cate
                                ,origin_cate
                                ,item_id
                                ,priority
                                ,5 + ROW_NUMBER() OVER (PARTITION BY assigned_cate ORDER BY priority,random(item_id) ) AS rank --优先级大于5，必定低于已分配商品
                        FROM    candidate_assignment
                        UNION ALL
                        SELECT  codebook
                                ,origin_codebook
                                ,item_id
                                ,0 AS priority
                                ,BIGINT(index) AS rank
                        FROM    item_codebook_info
                    )
        )
WHERE   rank <= 5
;

-- 6.最终结果回写到主表
INSERT OVERWRITE TABLE item_codebook_info
SELECT  item_id
        ,origin_cate
        ,assigned_cate
        ,rank
FROM    (
            SELECT  item_id
                    ,origin_cate
                    ,assigned_cate
                    ,rank
                    ,ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY priority ) AS rn -- 对于每个商品，保留第一优先级的选择（不会影响到已分配品）
            FROM    final_assignment
        )
WHERE   rn = 1
;
-- 重复上述循环，从第3步开始
