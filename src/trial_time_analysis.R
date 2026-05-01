library(conflicted)
library(lme4)
library(lmerTest)
library(tidyverse)
library(patchwork)
library(emmeans)
library(report) # For easystats reporting
library(performance) # For model diagnostics
library(see) # For visualizations
library(parameters) # For effect sizes
library(moments) # for skewness()
library(feather)
conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::lag)
conflicts_prefer(lme4::lmer)
conflicts_prefer(moments::skewness)

source_data <- "data/hopscotch_data_no_floor2.feather"

all_trial_data <- read_feather(source_data)

# set factor levels for condition and obstacles and subject
all_trial_data <- all_trial_data %>%
    mutate(
        condition = factor(condition),
        obstacles = factor(obstacles),
        subject = factor(subject)
    )

all_trial_data <- all_trial_data %>% arrange(subject, condition, obstacles, Time)

# see if there are any NaN / missing values
missing_values <- all_trial_data %>%
    summarise_all(~ sum(is.na(.)))

trial_times_by_subj_cond <- all_trial_data %>%
    group_by(subject, condition, obstacles) %>%
    summarise(
        trial_time = max(Time),
        subject = first(subject),
        condition = first(condition),
        obstacles = first(obstacles)
    ) %>%
    ungroup()
# rename conditions
# h -> extrinsic
# s -> intrinsic
# k -> control
trial_times_by_subj_cond <- trial_times_by_subj_cond %>%
    mutate(condition = recode(condition, "h" = "extrinsic", "s" = "intrinsic", "k" = "control"))

# plot distribution curve colored by condition
ggplot(trial_times_by_subj_cond, aes(x = trial_time, fill = condition)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of Trial Durations by Condition", x = "Trial Duration (seconds)", y = "Density") +
    theme_minimal() +
    scale_fill_manual(values = c(intrinsic="lightblue", extrinsic="salmon", control="lightgreen")) +
    theme(text = element_text(size = 14)) +
    theme(legend.title = element_blank())

ggsave("analysis/trial_time_distribution.png", width = 8, height = 6, dpi = 300)

trial_times_by_subj_cond %>% arrange(desc(trial_time))


# Fit linear mixed model with subject as random effect
model <- lmer(trial_time ~ condition * obstacles + (1 | subject), data = trial_times_by_subj_cond)
summary(model)
report(model, estimator = "ML")

# Q-Q plot for model residuals
qqnorm(residuals(model))
qqline(residuals(model))

# Model diagnostics
check_model(model)

# Model log
trial_times_by_subj_cond_log <- trial_times_by_subj_cond %>%
    mutate(log_trial_time = log(trial_time))
model_log <- lmer(log_trial_time ~ condition * obstacles + (1 | subject), data = trial_times_by_subj_cond_log)
summary(model_log)
report(model_log, estimator = "ML")

# Q-Q plot for model residuals
qqnorm(residuals(model_log))
qqline(residuals(model_log))

check_model(model_log)

# Transform results back to original scale for interpretation



# Now create an accuracy model:
accuracy_csv <- "data/Additional data incl. demographics and trial combinations final.csv"
accuracy_data <- read_delim(accuracy_csv, delim = ";", locale = locale(decimal_mark = ","))

# accuracy columns are: k0_acc;k1_acc;h0_acc;h1_acc;s0_acc;s1_acc;
# participant id is in: participant_no
accuracy_data <- accuracy_data %>%
    select(participant_no, k0_acc, k1_acc, h0_acc, h1_acc, s0_acc, s1_acc) %>%
    pivot_longer(cols = -participant_no, names_to = "condition_obstacles", values_to = "accuracy") %>%
    separate(condition_obstacles, into = c("na", "condition", "obstacles"), sep = "", extra = "drop") %>%
    select(-na) %>%
    mutate(
        condition = recode(condition, "h" = "extrinsic", "s" = "intrinsic", "k" = "control"),
        obstacles = factor(obstacles),
        participant_no = factor(participant_no)
    )

accuracy_data$log_accuracy <- log(accuracy_data$accuracy + 1e-5) # Add small constant to avoid log(0)
accuracy_model <- lmer(log_accuracy ~ condition * obstacles + (1 | participant_no), data = accuracy_data)
summary(accuracy_model)
report(accuracy_model, estimator = "ML")

accuracy_model_nolog <- lmer(accuracy ~ condition * obstacles + (1 | participant_no), data = accuracy_data)

qqnorm(residuals(accuracy_model))
qqline(residuals(accuracy_model))

qqnorm(residuals(accuracy_model_nolog))
qqline(residuals(accuracy_model_nolog))

# plot distribution of accuracy
ggplot(accuracy_data, aes(x = accuracy, fill = condition)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of Accuracy by Condition", x = "Accuracy", y = "Density") +
    theme_minimal() +
    scale_fill_manual(values = c(intrinsic="lightblue", extrinsic="salmon", control="lightgreen")) +
    theme(text = element_text(size = 14)) +
    theme(legend.title = element_blank())



ggsave("analysis/accuracy_distribution.png", width = 8, height = 6, dpi = 300)
