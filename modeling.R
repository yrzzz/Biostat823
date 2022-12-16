library(tidyverse)
library(caret)
library(pROC)
#read cv models
lr_model_block = read_rds("cache/model_lr_model_block.rds")
lr_model_horizon = read_rds("cache/model_lr_model_horizon.rds")
lr_model_kmeans = read_rds("cache/model_lr_model_kmeans.rds")

rf_model_block = read_rds("cache/model_rf_model_block.rds")
rf_model_horizon = read_rds("cache/model_rf_model_horizon.rds")
rf_model_kmeans = read_rds("cache/model_rf_model_kmeans.rds")

dt_model_block = read_rds("cache/model_dt_model_block.rds")
dt_model_horizon = read_rds("cache/model_dt_model_horizon.rds")
dt_model_kmeans = read_rds("cache/model_dt_model_kmeans.rds")

qda_model_block = read_rds("cache/model_qda_model_block.rds")
qda_model_horizon = read_rds("cache/model_qda_model_horizon.rds")
qda_model_kmeans = read_rds("cache/model_qda_model_kmeans.rds")
lda_model_block = read_rds("cache/model_lda_model_block.rds")
lda_model_horizon = read_rds("cache/model_lda_model_horizon.rds")
lda_model_kmeans = read_rds("cache/model_lda_model_kmeans.rds")
#find_best_parameter
#logistic regression  tune_param = c(0,0.0005, seq(0.001,0.01,0.001))
#random forest tune_param = seq(1, 8, by=1)
#decision tree tune_param =c(0,0.0005, seq(0.001,0.03,0.001))

# Function to find best parameters of the model
find_best_parameter = function(model, tune_param) {
  accuracy_mean = colMeans(model$accuracy)
  best_parameter = tune_param[which.max(accuracy_mean)]
  return(best_parameter)
}

# Find the best parameters
params_lr = list(block = find_best_parameter(lr_model_block,c(0,0.0005, seq(0.001,0.01,0.001))),
                 horizon = find_best_parameter(lr_model_horizon,c(0,0.0005, seq(0.001,0.01,0.001))),
                 kmeans = find_best_parameter(lr_model_kmeans,c(0,0.0005, seq(0.001,0.01,0.001))))
params_rf = list(block = find_best_parameter(rf_model_block,seq(1, 8, by=1)),
                 horizon = find_best_parameter(rf_model_horizon,seq(1, 8, by=1)),
                 kmeans = find_best_parameter(rf_model_kmeans,seq(1, 8, by=1)))
params_dt = list(block = find_best_parameter(dt_model_block,c(0,0.0005, seq(0.001,0.03,0.001))),
                 horizon = find_best_parameter(dt_model_horizon,c(0,0.0005, seq(0.001,0.03,0.001))),
                 kmeans = find_best_parameter(dt_model_kmeans,c(0,0.0005, seq(0.001,0.03,0.001))))

# A fuction to train model under best paramneters for different methods
model_train = function(trainData, label_col = "expert_label", classifier="qda",best_params) {
  set.seed(1)
  caretctrl = trainControl(method = "none")
  # Training using Caret Wrapper
  trainData=trainData%>%
    mutate('log(SD)' = log(SD))%>%
    select(-x_coordinate,-y_coordinate,-class,-id,-SD)%>%
    mutate(across(expert_label, factor))
  colnames(trainData) <- make.names(colnames(trainData))
  if (classifier == "glmnet") {
    return(
      train(
        expert_label ~ .,
        data = trainData,
        method = classifier,
        family = "binomial",
        preProcess = c("center", "scale"),
        tuneLength = 1,
        tuneGrid = data.frame(alpha = 1, lambda = best_params),
        trControl = caretctrl
      )
    )
  } else if (classifier == "rf") {
    return(
      train(
        expert_label ~ .,
        data = trainData,
        method = classifier,
        tuneLength = 1,
        tuneGrid = data.frame(mtry = best_params),
        trControl = caretctrl
      )
    )
  } else if (classifier == "rpart") {
    return(
      cvModel = train(
        expert_label ~ .,
        data = trainData,
        method = classifier,
        tuneLength = 1,
        tuneGrid = data.frame(cp = best_params),
        trControl = caretctrl
      )
    )
  } else {
    # for lda and qda, which do not have hyper parameter
    return(
      train(
        expert_label ~ .,
        data = trainData,
        method = classifier,
        preProcess = c("center", "scale"),
        tuneLength = 1,
        trControl = caretctrl
      )
    )
  }
}
#training model using best parameter and training data under different split methods
#read training data
train_block = read_rds("cache/02_train_block_2_5.rds")
train_horizon = read_rds("cache/02_train_block_1_10.rds")
train_kmeans = read_rds("cache/02_train_kmeans.rds")
#block
model_lr_block = model_train(trainData = train_block,classifier = 'glmnet',best_params = params_lr$block)
model_rf_block = model_train(trainData = train_block,classifier = 'rf',best_params = params_rf$block)
model_dt_block = model_train(trainData = train_block,classifier = 'rpart',best_params = params_dt$block)
model_qda_block = model_train(trainData = train_block,classifier = 'qda')
model_lda_block = model_train(trainData = train_block,classifier = 'lda')

#horizon
model_lr_horizon = model_train(trainData = train_horizon,classifier = 'glmnet',best_params = params_lr$horizon)
model_rf_horizon = model_train(trainData = train_horizon,classifier = 'rf',best_params = params_rf$horizon)
model_dt_horizon = model_train(trainData = train_horizon,classifier = 'rpart',best_params = params_dt$horizon)
model_qda_horizon = model_train(trainData = train_horizon,classifier = 'qda')
model_lda_horizon = model_train(trainData = train_horizon,classifier = 'lda')

#kmeans
model_lr_kmeans = model_train(trainData = train_kmeans,classifier = 'glmnet',best_params = params_lr$kmeans)
model_rf_kmeans = model_train(trainData = train_kmeans,classifier = 'rf',best_params = params_rf$kmeans)
model_dt_kmeans = model_train(trainData = train_kmeans,classifier = 'rpart',best_params = params_dt$kmeans)
model_qda_kmeans = model_train(trainData = train_kmeans,classifier = 'qda')
model_lda_kmeans = model_train(trainData = train_kmeans,classifier = 'lda')

write_rds(model_rf_block,"cache/model_rf_block.rds")
write_rds(model_rf_horizon,"cache/model_rf_horizon.rds")
write_rds(model_rf_kmeans,"cache/model_rf_kmeans.rds")

#use test data to get other metrics
test = read_rds("cache/02_test.rds")%>%
  mutate('log(SD)' = log(SD))%>%
  select(-x_coordinate,-y_coordinate,-class,-id,-SD)%>%
  mutate(across(expert_label, factor))
colnames(test) <- make.names(colnames(test))

#Get metrics
get_metrics = function(testData,model,thres){
  y_pred_prob = predict(model, newdata = testData, type ="prob")
  roc_obj=roc(testData[["expert_label"]] ~ y_pred_prob[,"1"], smoothed=TRUE, plot=FALSE)
  #get the "best" point
  x_best = coords(roc_obj, "b", ret="fpr", best.method="closest.toplef")[[1]]
  y_best = coords(roc_obj, "b", ret="tpr", best.method="closest.toplef")[[1]]

  # Loss computation
  # thres = as.numeric(coords(roc_obj, "best", "threshold")['threshold'])
  y_pred <- as.factor(ifelse(y_pred_prob[,"1"] > thres, "1", "-1"))

  testData = data.frame(obs = as.factor(c(testData[, "expert_label"])), pred =as.factor( c(y_pred)))
  #accuracy,precision,recall,f1,auc
  acc = confusionMatrix(data = testData$pred, reference = testData$obs, mode = "prec_recall")$overall[["Accuracy"]]
  pre = confusionMatrix(data = testData$pred, reference = testData$obs, mode = "prec_recall")$byClass[["Precision"]]
  rec = confusionMatrix(data = testData$pred, reference = testData$obs, mode = "prec_recall")$byClass[["Recall"]]
  f1 = confusionMatrix(data = testData$pred, reference = testData$obs, mode = "prec_recall")$byClass[["F1"]]
  auc = round(auc(roc_obj)[1], 3)
  list(roc = roc_obj, accuracy = acc, precision = pre, recall = rec, f1score = f1, AUC = auc,x_best = x_best, y_best = y_best,threshold = thres )
}

# A function to find best cutoff point
find_best_thres = function(model){
  accuracy_mean = colMeans(model$accuracy)
  thres = mean(model$threshold[,which.max(accuracy_mean)])
  return(thres)
}

# Find best cutoff points
thress_lr = list(block = find_best_thres(lr_model_block),
                 horizon = find_best_thres(lr_model_horizon),
                 kmeans = find_best_thres(lr_model_kmeans))
thress_rf = list(block = find_best_thres(rf_model_block),
                 horizon = find_best_thres(rf_model_horizon),
                 kmeans = find_best_thres(rf_model_kmeans))
thress_dt = list(block = find_best_thres(dt_model_block),
                 horizon = find_best_thres(dt_model_horizon),
                 kmeans = find_best_thres(dt_model_kmeans))
thress_qda = list(block = find_best_thres(qda_model_block),
                  horizon = find_best_thres(qda_model_horizon),
                  kmeans = find_best_thres(qda_model_kmeans))
thress_lda = list(block = find_best_thres(lda_model_block),
                  horizon = find_best_thres(lda_model_horizon),
                  kmeans = find_best_thres(lda_model_kmeans))

# get and save metric result for different models and different split methods
# horizon
metric_lr_horizon = get_metrics(test,model_lr_horizon,thress_lr$horizon) %>%
  write_rds("cache/metric_lr_horizon.rds")

metric_rf_horizon = get_metrics(test,model_rf_horizon,thress_rf$horizon) %>%
  write_rds("cache/metric_rf_horizon.rds")

metric_dt_horizon  = get_metrics(test,model_dt_horizon,thress_dt$horizon) %>%
  write_rds("cache/metric_dt_horizon.rds")

metric_qda_horizon = get_metrics(test,model_qda_horizon,thress_qda$horizon) %>%
  write_rds("cache/metric_qda_horizon.rds")

metric_lda_horizon = get_metrics(test,model_lda_horizon,thress_lda$horizon) %>%
  write_rds("cache/metric_lda_horizon.rds")

#kmeans
metric_lr_kmeans = get_metrics(test,model_lr_kmeans,thress_lr$kmeans) %>%
  write_rds("cache/metric_lr_kmeans.rds")

metric_rf_kmeans = get_metrics(test,model_rf_kmeans,thress_rf$kmeans) %>%
  write_rds("cache/metric_rf_kmeans.rds")

metric_dt_kmeans  = get_metrics(test,model_dt_kmeans,thress_dt$kmeans) %>%
  write_rds("cache/metric_dt_kmeans.rds")

metric_qda_kmeans = get_metrics(test,model_qda_kmeans,thress_qda$kmeans) %>%
  write_rds("cache/metric_qda_kmeans.rds")

metric_lda_kmeans = get_metrics(test,model_lda_kmeans,thress_lda$kmeans) %>%
  write_rds("cache/metric_lda_kmeans.rds")

# calculate auc-roc for different model and different split methods
roc.list_horizon <- list("Logistic Regression" = metric_lr_horizon$roc,
                        "Random Forest" = metric_rf_horizon$roc,
                        "Decision Tree" = metric_dt_horizon$roc,
                        "QDA" = metric_qda_horizon$roc,
                        "LDA" = metric_lda_horizon$roc
                        )
roc.list_kmeans <- list("Logistic Regression" = metric_lr_kmeans$roc,
                        "Random Forest" = metric_rf_kmeans$roc,
                        "Decision Tree" = metric_dt_kmeans$roc,
                        "QDA" = metric_qda_kmeans$roc,
                        "LDA" = metric_lda_kmeans$roc
)

cut_off_horizon = data.frame(x = c(metric_lr_horizon$x_best,
                                   metric_rf_horizon$x_best,
                                   metric_dt_horizon$x_best,
                                   metric_qda_horizon$x_best,
                                   metric_lda_horizon$x_best),
                             y = c(metric_lr_horizon$y_best,
                                   metric_rf_horizon$y_best,
                                   metric_dt_horizon$y_best,
                                   metric_qda_horizon$y_best,
                                   metric_lda_horizon$y_best)
)
cut_off_kmeans = data.frame(x = c(metric_lr_kmeans$x_best,
                                  metric_rf_kmeans$x_best,
                                  metric_dt_kmeans$x_best,
                                  metric_qda_kmeans$x_best,
                                  metric_lda_kmeans$x_best),
                            y = c(metric_lr_kmeans$y_best,
                                  metric_rf_kmeans$y_best,
                                  metric_dt_kmeans$y_best,
                                  metric_qda_kmeans$y_best,
                                  metric_lda_kmeans$y_best)
)

# plot roc curve for different models under different split methods
horizon_ROC = ggroc(roc.list_horizon, aes= "color",legacy.axes = TRUE)+
  geom_point(data = cut_off_block,aes(x=x,y=y),color = "black") +
  scale_colour_brewer(palette="Paired",name = 'Model Type')+
  theme_bw()+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="darkgrey", linetype="dashed")

kmean_ROC= ggroc(roc.list_kmeans, aes= "color",legacy.axes = TRUE)+
  geom_point(data = cut_off_block,aes(x=x,y=y),color = "black") +
  scale_colour_brewer(palette="Paired",name = 'Model Type')+
  theme_bw()+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="darkgrey", linetype="dashed")



ggsave(
  "graphs/horizon_ROC.png",
  horizon_ROC,
  width = 18,
  height = 15,
  units = "cm"
)

ggsave(
  "graphs/kmean_ROC.png",
  kmean_ROC,
  width = 18,
  height = 15,
  units = "cm"
)

#get confusion matrix
confusion_rf = function(data,model,thres){

  y_pred_prob = predict(model, newdata = data, type ="prob")
  y_pred <- as.factor(ifelse(y_pred_prob[,"1"] > thres, "1", "-1"))

  test = data.frame(obs = as.factor(c(data[, "expert_label"])), pred =as.factor( c(y_pred)))

  return(confusionMatrix(data = test$pred, reference = test$obs, mode = "prec_recall"))
}
confusion_rf(test,model_rf_kmeans,thress_rf$kmeans)
