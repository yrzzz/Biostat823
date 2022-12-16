library(tidyverse)
library(splitTools)
library(caret)
library(mlbench)

# splitting testing and training data
image = read_rds("cache/image.rds")
#make this example reproducible
set.seed(1)
image$id = 1:nrow(image)
#Use 70% of dataset as training set and remaining 30% as testing set
train = image %>% dplyr::sample_frac(0.7)
test_block  = dplyr::anti_join(image, train, by = 'id')

#find the coordinate of each block
find_blocks = function(x, y, cols ,rows ) {
  x_min = min(x)
  x_max = max(x)
  x_step = (x_max-x_min)/cols

  y_min = min(y)
  y_max = max(y)
  y_step = (y_max-y_min)/rows

  block = tibble()

  for (i in seq_len(rows)) {
    # make top and bottom
    top = floor(y_min + (i-1) * y_step)
    bot = floor(y_max - (rows-i) * y_step )
    if (i != rows)
      bot = bot - 1

    for (j in seq_len(cols)){
      # make left and right
      left = floor(x_min + (j-1) * x_step)
      right = floor(x_max - (cols-j) * x_step )
      if (j != cols)
        right = right - 1

      block %>%
        bind_rows(data.frame(top, bot, left, right)) -> block
    }
  }
  names(block) = c("top", "bottom", "left", "right")
  return(block)
}

# Split blocks into training blocks and validation blocks
split_blocks = function(block, cols , rows , train_num_blocks , val_num_blocks ) {
  set.seed(1)
  train_index = sample(seq_len(cols*rows),train_num_blocks)
  set.seed(11)
  val_index = sample(setdiff(seq_len(cols*rows),train_index),val_num_blocks)

  train_blocks = block[train_index,]
  val_blocks = block[val_index,]
  return(list(train = train_blocks, val = val_blocks))
}

# Split data into based on given blocks
split_data = function(data, block) {
  final_data = tibble()
  for (i in seq_len(nrow(block))) {
    coords = block[i,]

    data %>%
      filter(x_coordinate >= coords$left, x_coordinate <= coords$right,
             y_coordinate >= coords$top, y_coordinate <= coords$bottom) -> filtered_data

    final_data = rbind(final_data, filtered_data)
  }

  return(final_data)
}

# Final function split data into blocks
split_blocks_main = function(df, cols, rows , train_num_blocks, val_num_blocks){
  blocks = find_blocks(df$x_coordinate, df$y_coordinate,cols, rows)
  block_indices = split_blocks(blocks, cols, rows,
                                train_num_blocks, val_num_blocks)

  val = split_data(df, block_indices$val)
  train = split_data(df, block_indices$train)
  return(list(val = val, train = train))
}

#use k-means to cluster data and split them based on cluster
kmeans_splitting = function(df, blocks , train_num_blocks, val_num_blocks){
  set.seed(1)
  train_index = sample(seq_len(blocks),train_num_blocks)
  val_index = setdiff(seq_len(blocks),train_index)
  clustering_img1 <- kmeans(
    df%>%
      filter(expert_label!=0,class == "image1")%>%
      select(x_coordinate, y_coordinate) ,
    centers = 10,
    nstart = 20
  )
  clustering_img2 <- kmeans(
    df%>%
      filter(expert_label!=0,class == "image2")%>%
      select(x_coordinate, y_coordinate) ,
    centers = 10,
    nstart = 20
  )
  clustering_img3 <- kmeans(
    df%>%
      filter(expert_label!=0,class == "image3")%>%
      select(x_coordinate, y_coordinate) ,
    centers = 10,
    nstart = 20
  )

  final_data = bind_rows((df%>%
               filter(expert_label!=0,class == "image1")%>%
                 mutate(block = clustering_img1$cluster)),
            ((df%>%
                filter(expert_label!=0,class == "image2")%>%
                mutate(block = clustering_img2$cluster))),
            ((df%>%
                filter(expert_label!=0,class == "image3")%>%
                mutate(block = clustering_img3$cluster)))
  )
  val = final_data%>%filter(block %in% val_index)
  train = final_data%>%filter(block %in% train_index)
  return(list(val = val, train = train))

}


# Split train data using block split and kmeans split under different setting
image_dfs_1 = split_blocks_main(train, cols = 2, rows = 5, train_num_blocks = 8, val_num_blocks = 2)
image_dfs_2 = split_blocks_main(train, cols = 1, rows = 10, train_num_blocks = 8, val_num_blocks = 2)
image_dfs_3 = kmeans_splitting(train, blocks = 10, train_num_blocks = 8, val_num_blocks = 2)
#method1
train_block_2_5 = image_dfs_1$train
val_block_2_5 = image_dfs_1$val
#method2
train_block_1_10 = image_dfs_2$train
val_block_1_10 = image_dfs_2$val
#kmeans
train_kmeans = image_dfs_3$train
val_kmeans = image_dfs_3$val

# Baseline model for different split methods
#method1:block 2*5
train_block_2_5 %>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

val_block_2_5%>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

#method2: block 1*10
# Split data into partitions
train_block_1_10 %>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

val_block_1_10%>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

#method3: k-means, k = 10
# Split data into partitions
train_kmeans %>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

val_kmeans%>%
  filter(expert_label != 0) %>%
  summarise(acc = mean(expert_label == -1))

#k features by importance using the caret r packageR
# ensure results are repeatable
set.seed(1)
# Select features we will measure of block spliting method
x_train_block <- train_block_2_5%>%
  filter(expert_label != 0)%>%
  dplyr::select(NDAI:Rad_AN)%>%
  mutate("log(SD)" = log(SD))


y_train_block <-train_block_2_5%>%
  filter(expert_label != 0)%>%
  dplyr::select(expert_label)

# calculate feature importance
roc_imp_block <- filterVarImp(x = x_train_block, y = y_train_block$expert_label)
roc_imp_block <- data.frame(cbind(variable = rownames(roc_imp_block), score = roc_imp_block[,1]))
roc_imp_block$score <- as.double(roc_imp_block$score)

roc_imp_block[order(roc_imp_block$score,decreasing = TRUE),]
roc_imp_block$set = "Block"

## Select features we will measure of block horizontal block
x_train_block_h <- train_block_1_10%>%
  filter(expert_label != 0)%>%
  dplyr::select(NDAI:Rad_AN)%>%
  mutate("log(SD)" = log(SD))


y_train_block_h <-train_block_1_10%>%
  filter(expert_label != 0)%>%
  dplyr::select(expert_label)


# calculate feature importance
roc_imp_block_h <- filterVarImp(x = x_train_block_h, y = y_train_block_h$expert_label)
roc_imp_block_h <- data.frame(cbind(variable = rownames(roc_imp_block_h), score = roc_imp_block_h[,1]))
roc_imp_block_h$score <- as.double(roc_imp_block_h$score)

roc_imp_block_h[order(roc_imp_block_h$score,decreasing = TRUE),]
roc_imp_block_h$set = "Horizontal Block"

# Select features we will measure of kmeans spliting method
x_train_kmeans <- train_kmeans%>%
  filter(expert_label != 0)%>%
  dplyr::select(NDAI:Rad_AN)%>%
  mutate("log(SD)" = log(SD))


y_train_kmeans <-train_kmeans%>%
  filter(expert_label != 0)%>%
  dplyr::select(expert_label)

#calculate feature importance
roc_imp_kmeans <- filterVarImp(x = x_train_kmeans, y = y_train_kmeans$expert_label)
roc_imp_kmeans <- data.frame(cbind(variable = rownames(roc_imp_kmeans), score = roc_imp_kmeans[,1]))
roc_imp_kmeans$score <- as.double(roc_imp_kmeans$score)

roc_imp_kmeans[order(roc_imp_kmeans$score,decreasing = TRUE),]
roc_imp_kmeans$set = "K-Means"

# bind the result
df_roc_imp = bind_rows(roc_imp_block_h,roc_imp_kmeans)

# plot feature importance 
varimp = ggplot(df_roc_imp, aes(x=reorder(variable, score), y=score,fill=set)) +
#  geom_point() +
  geom_bar(stat = 'identity', position = position_dodge(0.5),width = 0.5)+
  scale_fill_manual(values=c('black','darkgray','lightgray'))+
  ylab("Variable Importance") +
  xlab("Variable Name") +
  coord_flip()+  theme_bw()
ggsave(
  "graphs/varimp_image.png",
  varimp,
  width = 15,
  height = 14,
  units = "cm"
)
 
# subset labelled train and test data of different split methods
train = train%>%
  filter(expert_label != 0)%>%
  write_rds("cache/02_train.rds")
test_block %>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_test.rds")

train_block_2_5 %>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_train_block_2_5.rds")
val_block_2_5 %>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_val_block_2_5.rds")
train_block_1_10 %>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_train_block_1_10.rds")
val_block_1_10 %>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_val_block_1_10.rds")
train_kmeans %>%select(-block)%>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_train_kmeans.rds")
val_kmeans %>%select(-block)%>%
  filter(expert_label != 0) %>%
  write_rds("cache/02_val_kmeans.rds")


