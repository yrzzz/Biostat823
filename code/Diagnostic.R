library(tidyverse)
library(caret)
library(magrittr)
library(ggpubr)

# Read metric data
metric_rf_block = read_rds("cache/metric_rf_block.rds")
metric_rf_horizon = read_rds("cache/metric_rf_horizon.rds")
metric_rf_kmeans = read_rds("cache/metric_rf_kmeans.rds")

#Read models
model_rf_block = read_rds("cache/model_rf_block.rds")
model_rf_horizon = read_rds("cache/model_rf_horizon.rds")
model_rf_kmeans = read_rds("cache/model_rf_kmeans.rds")

#feature importance of random forest
#feature importance for different spliting method
rfimp_block = varImp(model_rf_block, scale = FALSE)
rfimp_horizon = varImp(model_rf_horizon, scale = FALSE)
rfimp_kmeans = varImp(model_rf_kmeans, scale = FALSE)

#block
rfimp_block <- data.frame(cbind(variable = rownames(rfimp_block$importance), score = rfimp_block$importance[,1]))
rfimp_block$score <- as.double(rfimp_block$score)

rfimp_block[order(rfimp_block$score,decreasing = TRUE),]
rfimp_block$set = "Block"
#horizon
rfimp_horizon <- data.frame(cbind(variable = rownames(rfimp_horizon$importance), score = rfimp_horizon$importance[,1]))
rfimp_horizon$score <- as.double(rfimp_horizon$score)

rfimp_horizon[order(rfimp_horizon$score,decreasing = TRUE),]
rfimp_horizon$set = "Horizon"
#kmeans
rfimp_kmeans <- data.frame(cbind(variable = rownames(rfimp_kmeans$importance), score = rfimp_kmeans$importance[,1]))
rfimp_kmeans$score <- as.double(rfimp_kmeans$score)

rfimp_kmeans[order(rfimp_kmeans$score,decreasing = TRUE),]
rfimp_kmeans$set = "Kmeans"

df_roc_imp = bind_rows(rfimp_horizon,rfimp_kmeans)

# Plot feature importance
varimp = ggplot(df_roc_imp, aes(x=reorder(variable, score), y=score,fill=set)) +
  #  geom_point() +
  geom_bar(stat = 'identity', position = position_dodge(0.5),width = 0.5)+
  scale_fill_manual(values=c('black','lightgray','darkgray'))+
  #geom_segment(aes(x=variable,xend=variable,y=0,yend=score,color = set,alpha = 0.7,linewidth = 5)) +
  ylab("Variable Importance") +
  xlab("Variable Name") +
  coord_flip()+  theme_bw()
ggsave(
  "graphs/rf_varimp_image.png",
  varimp,
  width = 15,
  height = 12,
  units = "cm"
)

#confusion matrix
#read test data
test = read_rds("cache/02_test.rds")%>%
  mutate('log(SD)' = log(SD))%>%
  select(-x_coordinate,-y_coordinate,-class,-id,-SD)%>%
  mutate(across(expert_label, factor))
colnames(test) <- make.names(colnames(test))
# block
y_pred_prob = predict(model_rf_block, newdata = test, type ="prob")
roc_obj=roc(test[["expert_label"]] ~ y_pred_prob[,"1"], smoothed=TRUE, plot=FALSE)
# Loss computation
thres = as.numeric(coords(roc_obj, "best", "threshold")['threshold'])
y_pred <- as.factor(ifelse(y_pred_prob[,"1"] > thres, "1", "-1"))

test_set = data.frame(obs = as.factor(c(test[, "expert_label"])), pred =as.factor( c(y_pred)))
block_confusion = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")
# horizon
y_pred_prob = predict(model_rf_horizon, newdata = test, type ="prob")
roc_obj=roc(test[["expert_label"]] ~ y_pred_prob[,"1"], smoothed=TRUE, plot=FALSE)
# Loss computation
thres = as.numeric(coords(roc_obj, "best", "threshold")['threshold'])
y_pred <- as.factor(ifelse(y_pred_prob[,"1"] > thres, "1", "-1"))

test_set = data.frame(obs = as.factor(c(test[, "expert_label"])), pred =as.factor( c(y_pred)))
horizon_confusion = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")

#Plot prediction graph

train = read_rds("cache/image.rds")
train_set = train%>%
  mutate('log(SD)' = log(SD))%>%
  select(-x_coordinate,-y_coordinate,-class,-SD)%>%
  mutate(across(expert_label, factor))
colnames(train_set) <- make.names(colnames(train_set))

# Plot misclassification pixel plot of Kmeans cluster
#kmeans
y_pred_kmeans = predict(model_rf_kmeans, newdata = train_set, type ="prob")
roc_kmeans=roc(train_set[["expert_label"]] ~ y_pred_kmeans[,"1"], smoothed=TRUE, plot=FALSE)
# Loss computation
thres = as.numeric(coords(roc_kmeans, "best", "threshold")['threshold'])
y_pred_b <- as.factor(ifelse(y_pred_kmeans[,"1"] > thres, "1", "-1"))

#preparation for graph
train$pred_label = y_pred_b
rf_pixel_kmeans = train%>%
  select(expert_label,pred_label,x_coordinate,y_coordinate,class)%>%
  mutate(pred_label = case_when(
    pred_label == expert_label ~ expert_label,
    expert_label == 0 ~ 0 ,
    TRUE ~ 2
  ))%>%
  mutate(
    across(
      pred_label,
      ~ case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled",
        .x == 2 ~ "Misclassified"
      )
    )
  )%>%ggplot(aes(x = x_coordinate, y = y_coordinate, color = factor(pred_label))) +
  geom_point() +
  scale_color_manual(values = c( "Cloud" = "#F4EDCA","Unlabelled" = "#C4961A","No cloud" = "#FFDB6D",
                                 "Misclassified" = "red"),
                     name = "pred_label") +
  labs(x = "X Coordinate", y = "Y Coordinate") +
  theme_bw()  + facet_grid(~class)
#save image
ggsave(
  "graphs/04_rf_pixel_kmeans.png",
  rf_pixel_kmeans,
  width = 18,
  height = 7,
  units = "cm"
)

# Plot boxplot of misclassification of Kmeans cluster split
box_data_kmeans = train%>%
  select(expert_label,pred_label,x_coordinate,y_coordinate,class)%>%
  mutate(pred_label = case_when(
    pred_label == expert_label ~ expert_label,
    expert_label == 0 ~ 0 ,
    TRUE ~ 2
  ))%>%mutate(prediction = case_when(
    pred_label == 2 ~ "misclassification",
    TRUE ~ "correct classification"
  ))%>%bind_cols(train%>%select(-expert_label,-pred_label,-x_coordinate,-y_coordinate,-class))%>%
  select(-expert_label,-pred_label,-x_coordinate,-y_coordinate,-class)

p1 = ggplot( box_data_kmeans, aes(prediction, NDAI))+
  geom_boxplot()
p2 = ggplot( box_data_kmeans, aes(prediction, SD))+
  geom_boxplot()
p3 = ggplot( box_data_kmeans, aes(prediction, CORR))+
  geom_boxplot()
p4 = ggplot( box_data_kmeans, aes(prediction, Rad_CF))+
  geom_boxplot()
p5 = ggplot( box_data_kmeans, aes(prediction, Rad_BF))+
  geom_boxplot()
p6 = ggplot( box_data_kmeans, aes(prediction, Rad_AF))+
  geom_boxplot()
p7 = ggplot( box_data_kmeans, aes(prediction, Rad_AN))+
  geom_boxplot()
p8 = ggplot( box_data_kmeans, aes(prediction, Rad_DF))+
  geom_boxplot()

boxplot_kmeans = ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,ncol=4,nrow=2)

ggsave(
  "graphs/04_boxplot_kmeans.png",
  boxplot_kmeans,
  width = 30,
  height = 15,
  units = "cm"
)

# Plot misclassification pixel plot of horizontal block
#horizon
y_pred_horizon = predict(model_rf_horizon, newdata = train_set, type ="prob")
roc_horizon=roc(train_set[["expert_label"]] ~ y_pred_horizon[,"1"], smoothed=TRUE, plot=FALSE)
# Loss computation
thres = as.numeric(coords(roc_horizon, "best", "threshold")['threshold'])
y_pred_h <- as.factor(ifelse(y_pred_horizon[,"1"] > thres, "1", "-1"))

#preparation for graph
train$pred_label = y_pred_h
rf_pixel_horizon = train%>%
  select(expert_label,pred_label,x_coordinate,y_coordinate,class)%>%
  mutate(pred_label = case_when(
    pred_label == expert_label ~ expert_label,
    expert_label == 0 ~ 0 ,
    TRUE ~ 2
  ))%>%
  mutate(
    across(
      pred_label,
      ~ case_when(
        .x == 1 ~ "Cloud",
        .x == -1 ~ "No cloud",
        .x == 0 ~ "Unlabelled",
        .x == 2 ~ "Misclassified"
      )
    )
  )%>%ggplot(aes(x = x_coordinate, y = y_coordinate, color = factor(pred_label))) +
  geom_point() +
  scale_color_manual(values = c( "Cloud" = "#F4EDCA","Unlabelled" = "#C4961A","No cloud" = "#FFDB6D",
                                 "Misclassified" = "red"),
                     name = "pred_label") +
  labs(x = "X Coordinate", y = "Y Coordinate") +
  theme_bw()  + facet_grid(~class)
#save image
ggsave(
  "graphs/04_rf_pixel_horizon.png",
  rf_pixel_horizon,
  width = 18,
  height = 7,
  units = "cm"
)

# plot boxplot of misclassification of horizontal block
box_data_horizon = train%>%
  select(expert_label,pred_label,x_coordinate,y_coordinate,class)%>%
  mutate(pred_label = case_when(
    pred_label == expert_label ~ expert_label,
    expert_label == 0 ~ 0 ,
    TRUE ~ 2
  ))%>%mutate(prediction = case_when(
    pred_label == 2 ~ "misclassification",
    TRUE ~ "correct classification"
  ))%>%bind_cols(train%>%select(-expert_label,-pred_label,-x_coordinate,-y_coordinate,-class))%>%
  # bind_cols(train%>%mutate('log(SD)' = log(SD))%>%
  #                  select(-expert_label,-pred_label,-x_coordinate,-y_coordinate,-class,-SD))%>%
  select(-expert_label,-pred_label,-x_coordinate,-y_coordinate,-class)

p1 = ggplot( box_data_horizon, aes(prediction, NDAI))+
  geom_boxplot()
p2 = ggplot( box_data_horizon, aes(prediction, SD))+
  geom_boxplot()
p3 = ggplot( box_data_horizon, aes(prediction, CORR))+
  geom_boxplot()
p4 = ggplot( box_data_horizon, aes(prediction, Rad_CF))+
  geom_boxplot()
p5 = ggplot( box_data_horizon, aes(prediction, Rad_BF))+
  geom_boxplot()
p6 = ggplot( box_data_horizon, aes(prediction, Rad_AF))+
  geom_boxplot()
p7 = ggplot( box_data_horizon, aes(prediction, Rad_AN))+
  geom_boxplot()
p8 = ggplot( box_data_horizon, aes(prediction, Rad_DF))+
  geom_boxplot()

boxplot_horizon = ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,ncol=4,nrow=2)

ggsave(
  "graphs/04_boxplot_horizon.png",
  boxplot_horizon,
  width = 30,
  height = 15,
  units = "cm"
)