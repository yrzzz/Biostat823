library(caret)
library(pROC)
library(tidyverse)

# Similar to the block split method in Data Preparation. R
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



split_data = function(data, cols ,rows ) {
  final_data = tibble()
  block = find_blocks(data$x_coordinate,data$y_coordinate,cols,rows)
  for (i in seq_len(nrow(block))) {
    coords = block[i,]
    data %>%
      filter(x_coordinate >= coords$left, x_coordinate <= coords$right,
             y_coordinate >= coords$top, y_coordinate <= coords$bottom) -> filtered_data
    filtered_data$block = i
    final_data = rbind(final_data, filtered_data)
  }

  return(final_data)
}

# Similar to the kmeans split method in Data Preparation. R
kmeans_splitting = function(df, blocks){
  set.seed(1)
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
  return(final_data)

}

# A function that conduct block cross validation
cvMaster = function(trainData,method = 'Block', label_col = "expert_label", folds= 10, classifier="qda", metric = 'ALL',tune_param = c(-1), verbose = T) {
 set.seed(1)
  #use training data from data splitting
  if(method == 'Block'){
    if(folds %%2 == 0){
      trainData = split_data(trainData, cols = 2,rows = folds/2 )
    }
    else{
      stop("Please input correct folds")
    }
  }else if(method == 'Horizontal'){
    trainData = split_data(trainData, cols = 1,rows = folds)
  }else if(method == 'Kmeans'){
    trainData = kmeans_splitting(trainData, blocks = 10)
  }else{
    stop("Please input correct method")
  }
  # Select features that will be used
 trainData=trainData%>%
    mutate('log(SD)' = log(SD))%>%
    select(-x_coordinate,-y_coordinate,-class,-id,-SD)
 # create variables to store result
  accuracy_vector = matrix(0, folds, length(tune_param))
  precision_vector = matrix(0, folds, length(tune_param))
  recall_vector = matrix(0, folds, length(tune_param))
  F1_vector = matrix(0, folds, length(tune_param))
  auc_vector = matrix(0, folds, length(tune_param))
  threshold = matrix(0, folds, length(tune_param))
  # loop block and set them to training set and validation set
 for (i in 1:folds) {
    # build train, val dataset
   # i = 1
    val_set = trainData%>%filter(block == i)%>%select(-block)
    train_set = trainData%>%filter(block != i)%>%select(-block)

    colnames(train_set) <- make.names(colnames(train_set))
    colnames(val_set) <- make.names(colnames(val_set))
    # Training using Caret Wrapper
   for (j in 1:length(tune_param)) {
     # j = 1
      tp = tune_param[j]
      fm = as.formula(paste0(label_col, " ~ ."))
      caretctrl = trainControl(method = "none")
      # since different model we use intake different preprocess option and different hyper parameter,
      # here we use if-else to specify the training process
      if (classifier == "glmnet") {
        tune=data.frame(alpha=1,lambda=tp)
        cvModel = train(
          form = fm,
          data = train_set,
          method = classifier,
          family = "binomial",
          preProcess = c("center","scale"),
          tuneLength = 1,
          tuneGrid = tune,
          metric = metric,
          trControl = caretctrl
        )
      }else if (classifier == "rf") {
        tune=data.frame(mtry=tp)
        cvModel = train(
          form = fm,
          data = train_set,
          method = classifier,
          tuneLength = 1,
          tuneGrid = tune,
          metric = metric,
          trControl = caretctrl
        )
      }else if (classifier == "rpart") {
        tune=data.frame(cp=tp)
        cvModel = train(
          form = fm,
          data = train_set,
          method = classifier,
          tuneLength = 1,
          tuneGrid = tune,
          metric = metric,
          trControl = caretctrl
        )
      }else { # for lda and qda, which do not have hyper parameter
        cvModel = train(
          form = fm,
          data = train_set,
          method = classifier,
          preProcess = c("center","scale"),
          tuneLength = 1,
          metric = metric,
          trControl = caretctrl
        )
      }
      y_pred_prob = predict(cvModel, newdata = val_set, type ="prob")
      roc_obj=roc(val_set[[label_col]] ~ y_pred_prob[,"1"], smoothed=TRUE, plot=FALSE)

      #get the "best" "threshold"
      thres = as.numeric(coords(roc_obj, "best", "threshold")['threshold'])
      # Loss computation
      y_pred <- as.factor(ifelse(y_pred_prob[,"1"] > thres, "1", "-1"))

      test_set = data.frame(obs = as.factor(c(val_set[, label_col])), pred =as.factor( c(y_pred)))

      acc = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$overall[["Accuracy"]]
      pre = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$byClass[["Precision"]]
      rec = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$byClass[["Recall"]]
      f1 = confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$byClass[["F1"]]

      test_set = data.frame(obs = as.numeric(c(val_set[, label_col])), pred = as.numeric(c(y_pred)))
      # create roc curve
      roc_object <- roc( test_set$obs, test_set$pred)
      # calculate area under curve
      auc = round(auc(roc_object)[1], 3)
      #plot(roc_object ,main ="ROC curve -- Logistic Regression ")


      if (verbose) {
        print(sprintf("The average accuracy of %s classifier with %f tunning parameter at %f fold is %f", classifier, tp, i, acc))
      }
      threshold[i,j] = thres
      accuracy_vector[i,j] = acc
      precision_vector[i,j] = pre
      recall_vector[i,j] = rec
      F1_vector[i,j] = f1
      auc_vector[i,j] = auc
   }

 }
  if(metric == 'ALL'){
    return (list("accuracy" = accuracy_vector,"precision" = precision_vector,
                 "recall" = recall_vector,"F1_score" = F1_vector,
                 "AUC" = auc_vector, "threshold" = threshold))
  }else if (metric == 'Accuracy') {
    return(list("accuracy" = accuracy_vector))
  }
  else if (metric == 'Precision') {
    return(list("precision" = precision_vector))
  }else if (metric == 'Recall') {
    return(list("recall" = recall_vector))
  }else if (metric == 'F1_score') {
    return(list("F1_score" = F1_vector))
  }else if (metric == 'AUC') {
    return(list("AUC" = auc_vector))
  }else if (metric == 'Threshold') {
    return(list("threshold" = threshold))
  }else{
    stop("Please input correct metric")
  }
}
# Block cross validation for different methods
train = read_rds("cache/02_train.rds")
train$expert_label=factor(train$expert_label)
#logistic regression
lr_model_block = cvMaster(train,method = 'Block', fold = 10, classifier="glmnet", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.01,0.001)))
lr_model_horizon = cvMaster(train,method = 'Horizontal', fold = 10, classifier="glmnet", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.01,0.001)))
lr_model_kmeans = cvMaster(train,method = 'Kmeans', fold = 10, classifier="glmnet", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.01,0.001)))

#random forest
rf_model_block =  cvMaster(train,method = 'Block', classifier="rf", verbose=T, tune_param = seq(1, 8, by=1))
rf_model_horizon =  cvMaster(train,method = 'Horizontal', fold = 10, classifier="rf", verbose=T, tune_param = seq(1, 8, by=1))
rf_model_kmeans =  cvMaster(train,method = 'Kmeans', fold = 10, classifier="rf", verbose=T, tune_param = seq(1, 8, by=1))

#decision tree
dt_model_block =  cvMaster(train,method = 'Block', classifier="rpart", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.03,0.001)))
dt_model_horizon =  cvMaster(train,method = 'Horizontal', fold = 10, classifier="rpart", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.03,0.001)))
dt_model_kmeans =  cvMaster(train,method = 'Kmeans', fold = 10, classifier="rpart", verbose=T, tune_param = c(0,0.0005, seq(0.001,0.03,0.001)))


#qda
qda_model_block = cvMaster(train, method = 'Block', classifier="qda",  verbose=T)
qda_model_horizon = cvMaster(train, method = 'Horizontal', fold = 10, classifier="qda",  verbose=T)
qda_model_kmeans = cvMaster(train, method = 'Kmeans', fold = 10, classifier="qda",  verbose=T)

#lda
lda_model_block = cvMaster(train, method = 'Block', classifier="lda",  verbose=T)
lda_model_horizon = cvMaster(train, method = 'Horizontal', fold = 10, classifier="lda",  verbose=T)
lda_model_kmeans = cvMaster(train, method = 'Kmeans', fold = 10, classifier="lda",  verbose=T)


#save models
#logistic regression
lr_model_block%>%write_rds("cache/model_lr_model_block.rds")
lr_model_horizon%>%write_rds("cache/model_lr_model_horizon.rds")
lr_model_kmeans%>%write_rds("cache/model_lr_model_kmeans.rds")

#random forest
rf_model_block%>%write_rds("cache/model_rf_model_block.rds")
rf_model_horizon%>%write_rds("cache/model_rf_model_horizon.rds")
rf_model_kmeans%>%write_rds("cache/model_rf_model_kmeans.rds")

#decision tree
dt_model_block%>%write_rds("cache/model_dt_model_block.rds")
dt_model_horizon%>%write_rds("cache/model_dt_model_horizon.rds")
dt_model_kmeans%>%write_rds("cache/model_dt_model_kmeans.rds")

#qda
qda_model_block%>%write_rds("cache/model_qda_model_block.rds")
qda_model_horizon%>%write_rds("cache/model_qda_model_horizon.rds")
qda_model_kmeans%>%write_rds("cache/model_qda_model_kmeans.rds")

#lda
lda_model_block%>%write_rds("cache/model_lda_model_block.rds")
lda_model_horizon%>%write_rds("cache/model_lda_model_horizon.rds")
lda_model_kmeans%>%write_rds("cache/model_lda_model_kmeans.rds")










