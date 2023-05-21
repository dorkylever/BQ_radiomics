#if (!require(readr))install.packages(c('readr','dplyr','tidyr', 'ggplot2', 'stringr', 'gridExtra'),
#                                      type="binary")

library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(stringr)
library(gridExtra)
library(grid)

args <- commandArgs(trailingOnly = TRUE)

# Check if the expected number of arguments is provided
if (length(args) != 2) {
  stop("Invalid number of arguments. Usage: Rscript plot.R --input <input_file>")
}

f2 <- read_csv(args[2])

cv_dir <- dirname(args[2])


p1 <- ggplot(f2, aes(x=`iterations`))+
  geom_line(aes(y=`train-Logloss-mean`, col='train'))+
  geom_ribbon(aes(y=`train-Logloss-mean`, ymin=`train-Logloss-mean`-`train-Logloss-std`,ymax=`train-Logloss-mean`+`train-Logloss-std`, col='train'), alpha=0.1)+
  geom_line(aes(y=`test-Logloss-mean`, col='test'))+
  geom_ribbon(aes(y=`train-Logloss-mean`, ymin=`test-Logloss-mean`-`test-Logloss-std`,ymax=`test-Logloss-mean`+`test-Logloss-std`, col='test'), alpha=0.1)+
  facet_wrap(~nfeats)+
  ylab("Mean Logloss Score")



p2 <- ggplot(f2, aes(x=`iterations`))+
  geom_line(aes(y=`train-Accuracy-mean`, col='train'))+
  geom_ribbon(aes(y=`train-Accuracy-mean`, ymin=`train-Accuracy-mean`-`train-Accuracy-std`,ymax=`train-Accuracy-mean`+`train-Accuracy-std`, col='train'), alpha=0.1)+
  geom_line(aes(y=`test-Accuracy-mean`, col='test'))+
  geom_ribbon(aes(y=`train-Accuracy-mean`, ymin=`test-Accuracy-mean`-`test-Accuracy-std`,ymax=`test-Accuracy-mean`+`test-Accuracy-std`, col='test'), alpha=0.1)+
  facet_wrap(~nfeats)+
  ylab("Mean Accuracy Score")




f2_logmean <- f2 %>% group_by(nfeats) %>% filter(iterations>100) %>% summarise(val=mean(`test-Logloss-mean`), val_std = sd(`test-Logloss-mean`),
                                                    train_mean = mean(`train-Logloss-mean`), train_std = sd(`train-Logloss-mean`))

lowest_val_feats <- f2_logmean[which.min(f2_logmean$val),]$nfeats

lowest_val_mean <-f2_logmean[which.min(f2_logmean$val),]$val


p3 <- ggplot(f2_logmean, aes(x=nfeats))+
  geom_line(aes(y=train_mean, col='train'))+
  geom_ribbon(aes(y=train_mean, ymin=train_mean-train_std,ymax=train_mean+train_std, col='train'), alpha=0.1)+
  geom_line(aes(y=val, col='test'))+
  geom_ribbon(aes(y=train_mean, ymin=val-val_std,ymax=val+val_std, col='test'), alpha=0.1)+
  geom_vline(xintercept=lowest_val_feats, linetype="dashed")+
  geom_text(aes(lowest_val_feats,0,label=lowest_val_feats, hjust=-1))+
  geom_hline(yintercept=lowest_val_mean, linetype="dashed")+
  geom_text(aes(1,lowest_val_mean, label=round(lowest_val_mean, 3), vjust=-1))+
  ylab("Mean Logloss Score")





f2_acc <- f2 %>% group_by(nfeats) %>% filter(iterations>100) %>% summarise(val=mean(`test-Accuracy-mean`), val_std = sd(`test-Accuracy-mean`),
                                                                               train_mean = mean(`train-Accuracy-mean`), train_std = sd(`train-Accuracy-mean`))

highest_val_feats <- f2_acc[which.max(f2_acc$val),]$nfeats

highest_val_mean <-f2_acc[which.max(f2_acc$val),]$val


p4 <- ggplot(f2_acc, aes(x=nfeats))+
  geom_line(aes(y=train_mean, col='train'))+
  geom_ribbon(aes(y=train_mean, ymin=train_mean-train_std,ymax=train_mean+train_std, col='train'), alpha=0.1)+
  geom_line(aes(y=val, col='test'))+
  geom_ribbon(aes(y=train_mean, ymin=val-val_std,ymax=val+val_std, col='test'), alpha=0.1)+
  geom_vline(xintercept=highest_val_feats, linetype="dashed")+
  geom_text(aes(highest_val_feats,0,label=highest_val_feats, hjust=-1))+
  geom_hline(yintercept=highest_val_mean, linetype="dashed")+
  geom_text(aes(1,highest_val_mean, label=round(highest_val_mean, 3), vjust=-1))+
  ylab("Mean Accuracy Score")

new_filepath <- file.path(cv_dir, "CatBoostTrainingData.pdf")

pdf(new_filepath, onefile = T, paper="a4r", width=13, height=10)

grid.arrange(grobs=list(p3, p4), ncol=2, nrow=1,
             top = textGrob(stringr::str_to_title("Training and Validation Data")),  gp=gpar(fontsize=28,font=8))

grid.arrange(grobs=list(p1, p2), ncol=2, nrow=1,
             top = textGrob(stringr::str_to_title("Training and Validation Data Subplots")),  gp=gpar(fontsize=28,font=8))


# Close the graphics device explicitly
graphics.off()




result <- paste("[",lowest_val_feats, ",", lowest_val_mean, ",", highest_val_feats, ",", highest_val_mean, "]")
cat(result)


