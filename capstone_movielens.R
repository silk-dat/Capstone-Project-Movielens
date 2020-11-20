#loading packages
library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(stringr)
library(knitr)
library(data.table)

#downloading and splitting dataset
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#remove unneccessary objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#data exploration
head(edx)
nrow(edx)
any(is.na(edx))

#unique ratings
sort(unique(edx$rating))

#number of distinct movies and users
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

#plot distribution of ratings per movie
edx %>% 
  group_by(movieId) %>% 
  summarise(n_ratings = n()) %>%
  ggplot(aes(n_ratings)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#plot distribution of movie ratings
edx %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_continuous(limit = c(1.5, 5)) + 
  ggtitle("Movies") 

#plot distribution of ratings per movie
edx %>% 
  group_by(userId) %>% 
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_continuous(limit = c(1.5, 5)) +
  ggtitle("Users")

#plot distribution of the number of ratings per movie
edx %>% 
  group_by(userId) %>% 
  summarise(n_ratings = n()) %>%
  ggplot(aes(n_ratings)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

#plot the release year vs avg_rating
#extract year from title
edx <- edx %>% 
  mutate(year = str_extract(edx$title, "\\((\\d{4}\\))"))
edx$year <- as.numeric(gsub("\\(|\\)", "", edx$year))

edx %>% 
  group_by(year) %>% 
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(year, avg_rating)) + 
  geom_point() + 
  geom_smooth()

#plot distribution of genres
edx %>% 
  group_by(genres) %>% 
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  top_n(15) %>%
  ggplot(aes(reorder(genres, -n), n)) + 
  geom_col() +
  theme(axis.text.x=element_text(angle = 90, hjust = 0)) +
  ggtitle("Distribution of Genres") + 
  labs(x = "Genres")

#average rating per genre, top 10
edx %>% 
  group_by(genres) %>%
  summarise(avg = mean(rating)) %>%
  arrange(desc(avg)) %>%
  top_n(10) 

#average rating per genre, low 10
edx %>% 
  group_by(genres) %>%
  summarise(avg = mean(rating)) %>%
  arrange(desc(avg)) %>%
  top_n(-10) 


#modelling

#define loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#splitting edx set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test_set set are also in train_set set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set set back into train_set set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)


#proportion predicting

#count the ratings in training_set
ratings <- train_set %>% group_by(rating) %>% summarise(n = n())
ratings <- as.data.frame(ratings)
pred <- c()

#create function for calculating predictions in test_set
predict <- function(x) {
  prop <- ratings$n[x]/sum(ratings$n)
  pred <- c(pred, rep(ratings$rating[x], nrow(test_set)*prop))
}
t <- 1:nrow(ratings)
predictions <- predict(t)

#calculating RMSE
prop_rmse <- RMSE(test_set$rating, predictions)

#create a table for storing the RMSE results of all models
rmse_results <- tibble(method = "Proportions", RMSE = prop_rmse)
prop_rmse


#predict the same rating regardless of other variables
mu <- mean(train_set$rating)

#calulating RMSE
avg_rmse <- RMSE(test_set$rating, mu)

#adding to the table
rmse_results <- add_row(rmse_results, method = "Just the Average", RMSE = avg_rmse)
avg_rmse


#addong movie bias 
#calculating movie bias for each movie
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#making predictions
predictions <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_rmse <- RMSE(test_set$rating, predictions)
rmse_results <- add_row(rmse_results, method = "Avg + Movie Bias", RMSE = movie_rmse)

movie_rmse


#adding user bias
#calculating user bias for each user
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#making predictions
predictions <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
user_rmse <- RMSE(test_set$rating, predictions)
rmse_results <- add_row(rmse_results, method = "Avg + Movie Bias + User Bias", RMSE = user_rmse)
user_rmse


#adding year bias
#extract year from title
train_set <- train_set %>% 
  mutate(year = str_extract(train_set$title, "\\((\\d{4}\\))"))
train_set$year <- as.numeric(gsub("\\(|\\)", "", train_set$year))

test_set <- test_set %>% 
  mutate(year = str_extract(test_set$title, "\\((\\d{4}\\))"))
test_set$year <- as.numeric(gsub("\\(|\\)", "", test_set$year))

#calculate the year bias
year_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

#make predictions
predictions <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)
year_rmse <- RMSE(test_set$rating, predictions)
rmse_results <- add_row(rmse_results, method = "Avg + Movie Bias + User Bias + Year Bias", RMSE = year_rmse)
year_rmse


#adding genre bias
#calculating genre bias
genre_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "year") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_y))

#making predictions
predictions <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by = "year") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)
genre_rmse <- RMSE(test_set$rating, predictions)
rmse_results <- add_row(rmse_results, method = "Avg + Movie Bias + User Bias + Year Bias + Genre Bias", RMSE = genre_rmse)
genre_rmse


#finding optimal lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - mu - b_u)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu - b_u - b_y)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

reg_rmse <- rmses[which.min(rmses)]
rmse_results <- add_row(rmse_results, method = "Regularized Model", RMSE = reg_rmse)
reg_rmse
lambdas[which.min(rmses)]


#testing with validation set
#defining the average
mu <- mean(train_set$rating)

#defining lambda for regulization
l <- lambdas[which.min(rmses)]

#calculating regulized movie bias
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))

#calculating regulized user bias
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#defining year bias
#adding column year to validation set
validation <- validation %>% 
  mutate(year = str_extract(validation$title, "\\((\\d{4}\\))"))
validation$year <- as.numeric(gsub("\\(|\\)", "", validation$year))

#calculating the bias
year_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - b_i - mu - b_u)/(n()+l))

#calculating regulized genre bias
genre_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "year") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - mu - b_u - b_y)/(n()+l))

#calculating the prediction
predicted_ratings <- 
  validation %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "year") %>%
  left_join(genre_avgs, by = "genres") %>% 
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)

#calculating rmse
RMSE(validation$rating, predicted_ratings)


#show rmse table
as.data.frame(rmse_results)

