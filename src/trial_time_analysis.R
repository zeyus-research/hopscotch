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

# first remove all trials after the first non-sequential frame number
# there is a problem with some where some garbage data
# (max ~10 rows) were appended
all_trial_data_cropped <- all_trial_data %>%
    group_by(subject, condition, obstacles) %>%
    mutate(frame_diff = Frame - lag(Frame))

# find the first non-sequential frame for each subject, condition, and obstacles group
# the garbage frames are not unique in number and may
# be a number that exists earlier in the data
# but because the recorded data is sequential
# the first "out of place" frame is where the garbage
# starts
# in this case non-sequential is = decreasing (some skipped positive changes are fine)
first_non_sequential <- all_trial_data_cropped %>%
    filter(frame_diff < 0 & !is.na(frame_diff)) %>%
    group_by(subject, condition, obstacles) %>%
    summarise(first_non_seq_frame = first(Frame)) %>%
    ungroup()

# now filter out all rows where the frame is greater than or equal to the first non-sequential frame
all_trial_data_cropped <- all_trial_data_cropped %>%
    left_join(first_non_sequential, by = c("subject", "condition", "obstacles")) %>%
    filter(is.na(first_non_seq_frame) | Frame < first_non_seq_frame) %>%
    select(-frame_diff, -first_non_seq_frame)


# print subject condition obstacles for the rows with the least and most frames
all_trial_data_cropped %>% group_by(subject, condition, obstacles) %>% summarise(first(subject), first(condition), first(obstacles), rowcounts=n()) %>% ungroup() %>% arrange(rowcounts) %>% slice(c(1, n()))



# set factor levels for condition and obstacles and subject
all_trial_data <- all_trial_data %>%
    mutate(
        condition = factor(condition),
        obstacles = factor(obstacles),
        subject = factor(subject)
    )

all_trial_data <- all_trial_data %>% arrange(subject, condition, obstacles, trial, Time)

trial_times_by_subj_cond <- all_trial_data %>%
    group_by(subject, condition, obstacles) %>%
    summarise(
        trial_time = max(Time),
        subject = first(subject),
        condition = first(condition),
        obstacles = first(obstacles)
    ) %>%
    ungroup()

# plot distribution curve colored by condition
ggplot(trial_times_by_subj_cond, aes(x = trial_time, fill = condition)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of Trial Times by Condition", x = "Trial Time (seconds)", y = "Density") +
    theme_minimal() +
    scale_fill_manual(values = c("lightblue", "salmon", "lightgreen")) +
    theme(legend.title = element_blank())


trial_times_by_subj_cond %>% arrange(desc(trial_time))
